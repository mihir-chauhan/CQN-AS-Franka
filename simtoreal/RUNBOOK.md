# Real-Robot Runbook — Reproducing Section V of the Paper

This runbook walks through the complete workflow for the real-robot experiments
on the Franka Emika Panda: demonstration capture, training, evaluation, and
cross-method comparison (`CQN-AS`, `ARSQ`, `Pi-ARSQ`). All commands assume the
working directory is `simtoreal/`.

> **Substitutions.** Replace `<IP>` with your robot's IP and the four Orbbec
> serials with those of your camera rig. Use `python check_cameras.py` to
> enumerate connected Orbbec devices.

## Conventions

- `--orbbec-serials` is a JSON dictionary mapping the four canonical viewpoints
  (`front`, `wrist`, `left_shoulder`, `right_shoulder`) to device serials.
  Camera order is positional and must match the encoder configuration.
- All checkpoints, logs, and demos live under `../runs/<experiment>/`.
- `Ctrl+X` aborts the current episode; `Ctrl+C` saves a snapshot and exits.

```bash
ROBOT_IP=192.168.1.41
ORBBEC='{"front":"<SN1>","wrist":"<SN2>","left_shoulder":"<SN3>","right_shoulder":"<SN4>"}'
```

## 1. Capture Demonstrations

### 1.1 Record freedrive waypoint trajectories

```bash
python record_waypoints.py \
    --robot-ip $ROBOT_IP \
    --num-demos 10 --hz 10 \
    --save-dir ../runs/shelf_4cam/waypoints
```

### 1.2 Replay waypoints to capture multi-view camera demos

```bash
python train_real.py \
    --agent arsq \
    --robot-ip $ROBOT_IP \
    --camera-mode orbbec --orbbec-serials "$ORBBEC" \
    --demo-mode waypoint \
    --waypoint-dir ../runs/shelf_4cam/waypoints \
    --demo-dir     ../runs/shelf_4cam/demos \
    --save-dir     ../runs/shelf_4cam
```

## 2. Three-Way Comparison: CQN-AS vs ARSQ vs Pi-ARSQ

All three runs share identical demos, cameras, and hyperparameters; only the
agent and regularizer differ. Logs are piped through `tee` for downstream
plotting.

```bash
mkdir -p ../runs/{compare_cqn_as,compare_arsq_base,compare_arsq_phys}

COMMON=(
    --robot-ip      $ROBOT_IP
    --camera-mode   orbbec
    --orbbec-serials "$ORBBEC"
    --load-demos-only
    --demo-dir      ../runs/shelf_4cam/demos
    --goal-pose     goal_pose.json
    --num-train-steps 10000
    --episode-length 200
    --batch-size 32 --demo-batch-size 32
    --eval-every 2000 --save-every 500
)
```

### 2.1 CQN-AS (distributional RL + action sequences)

```bash
python train_real.py --agent cqn "${COMMON[@]}" \
    --save-dir ../runs/compare_cqn_as \
    2>&1 | tee ../runs/compare_cqn_as/train.log
```

### 2.2 ARSQ (autoregressive Soft Q, no regularization)

```bash
python train_real.py --agent arsq --no-fk-reg "${COMMON[@]}" \
    --save-dir ../runs/compare_arsq_base \
    2>&1 | tee ../runs/compare_arsq_base/train.log
```

### 2.3 Pi-ARSQ (ARSQ + geometric regularizer)

```bash
python train_real.py --agent arsq --use-fk-reg \
    --fk-weight 1.0 --kappa 0.1 --nu 0.01 --num-walks 10 \
    "${COMMON[@]}" \
    --save-dir ../runs/compare_arsq_phys \
    2>&1 | tee ../runs/compare_arsq_phys/train.log
```

### 2.4 Plot training comparison

```bash
python plot_comparison.py \
    --logs ../runs/compare_cqn_as/train.log \
           ../runs/compare_arsq_base/train.log \
           ../runs/compare_arsq_phys/train.log \
    --labels "CQN-AS" "ARSQ" "Pi-ARSQ" \
    --save ../runs/comparison_training.png
```

## 3. Evaluation

Run 10 episodes per checkpoint with identical conditions. Success is detected
automatically by comparing the end-effector pose to `goal_pose.json`.

```bash
EVAL_COMMON=(
    --robot-ip      $ROBOT_IP
    --camera-mode   orbbec
    --orbbec-serials "$ORBBEC"
    --goal-pose     goal_pose.json
    --num-episodes  10
)

python eval_real.py --agent cqn  "${EVAL_COMMON[@]}" \
    --snapshot     ../runs/compare_cqn_as/snapshot.pt \
    --action-stats ../runs/compare_cqn_as/action_stats.npz \
    --results-dir  ../runs/compare_cqn_as \
    2>&1 | tee ../runs/compare_cqn_as/eval.log

python eval_real.py --agent arsq --no-fk-reg "${EVAL_COMMON[@]}" \
    --snapshot     ../runs/compare_arsq_base/snapshot.pt \
    --action-stats ../runs/compare_arsq_base/action_stats.npz \
    --results-dir  ../runs/compare_arsq_base \
    2>&1 | tee ../runs/compare_arsq_base/eval.log

python eval_real.py --agent arsq --use-fk-reg "${EVAL_COMMON[@]}" \
    --snapshot     ../runs/compare_arsq_phys/snapshot.pt \
    --action-stats ../runs/compare_arsq_phys/action_stats.npz \
    --results-dir  ../runs/compare_arsq_phys \
    2>&1 | tee ../runs/compare_arsq_phys/eval.log

python compare_evals.py \
    --results ../runs/compare_cqn_as/eval_results.json \
              ../runs/compare_arsq_base/eval_results.json \
              ../runs/compare_arsq_phys/eval_results.json \
    --labels  "CQN-AS" "ARSQ" "Pi-ARSQ" \
    --save    ../runs/eval_comparison.png
```

## 4. Pick-and-Place Variant

The robot starts at the grasp-ready pose with the object already in the
gripper, then learns to lift, transport, and drop. Required reset waypoints
(snapshot the robot at each pose with `np.save`):

- `partialhomepickplace.npy` — first reset waypoint
- `partialhome.npy`          — intermediate waypoint
- `pickuppose.npy`           — final grasp-ready pose

The `--task-mode pick-place` flag homes the robot, pauses for the operator to
load the object, then closes the gripper before the episode begins.
`--freeze-gripper` removes the gripper dimension from policy control: the
gripper is closed at reset and opened at episode termination, isolating arm
behavior under the small-demo regime.

> **Safety note.** Demonstrations recorded at 10 Hz must be replayed at 10 Hz;
> using `--control-hz 15` rescales the deltas by 1.5× and trips the joint
> velocity reflex (~2.17 rad/s on joints 1–4). Pair `--joint-delta-clip 0.05`
> with `--control-hz 10` and `--velocity-factor 0.15` for a stable margin.

### 4.1 Demonstration capture (pick-place mode)

```bash
python record_waypoints.py \
    --robot-ip $ROBOT_IP --num-demos 10 --hz 10 \
    --save-dir ../runs/pick_drop/waypoints

python train_real.py --agent arsq \
    --robot-ip $ROBOT_IP \
    --camera-mode orbbec --orbbec-serials "$ORBBEC" \
    --demo-mode waypoint \
    --waypoint-dir ../runs/pick_drop/waypoints \
    --demo-dir     ../runs/pick_drop/demos \
    --task-mode pick-place \
    --home-sequence partialhomepickplace.npy partialhome.npy pickuppose.npy \
    --save-dir     ../runs/pick_drop
```

### 4.2 Offline training (recommended)

Offline mode trains purely from captured demonstrations, eliminating reflex
aborts and removing the need to stand at the robot. The robot still needs to
be reachable at startup for action-spec discovery.

```bash
PICK_COMMON=(
    --robot-ip      $ROBOT_IP
    --camera-mode   orbbec
    --orbbec-serials "$ORBBEC"
    --load-demos-only --offline
    --demo-dir      ../runs/pick_drop/demos
    --task-mode     pick-place
    --home-sequence partialhomepickplace.npy partialhome.npy pickuppose.npy
    --freeze-gripper
    --control-hz 10 --joint-delta-clip 0.05 --velocity-factor 0.15
    --num-train-steps 10000
    --batch-size 32 --demo-batch-size 256
    --save-every 1000 --log-every 100
)

# ARSQ baseline
python train_real.py --agent arsq --no-fk-reg "${PICK_COMMON[@]}" \
    --save-dir ../runs/pick_drop_arsq_base \
    2>&1 | tee ../runs/pick_drop_arsq_base/train.log

# Pi-ARSQ
python train_real.py --agent arsq --use-fk-reg \
    --fk-weight 1.0 --kappa 0.1 --nu 0.01 --num-walks 10 \
    "${PICK_COMMON[@]}" \
    --save-dir ../runs/pick_drop_arsq_phys \
    2>&1 | tee ../runs/pick_drop_arsq_phys/train.log
```

The two offline runs are independent — they may be launched in parallel on
separate GPUs (`CUDA_VISIBLE_DEVICES=0`, `CUDA_VISIBLE_DEVICES=1`).

### 4.3 Evaluation

```bash
PICK_EVAL_COMMON=(
    --robot-ip      $ROBOT_IP
    --camera-mode   orbbec
    --orbbec-serials "$ORBBEC"
    --goal-pose     goal_pose.json
    --task-mode     pick-place
    --home-sequence partialhomepickplace.npy partialhome.npy pickuppose.npy
    --freeze-gripper
    --control-hz 10 --joint-delta-clip 0.05 --velocity-factor 0.15
    --num-episodes 10
)

python eval_real.py --agent arsq --no-fk-reg "${PICK_EVAL_COMMON[@]}" \
    --snapshot     ../runs/pick_drop_arsq_base/snapshot.pt \
    --action-stats ../runs/pick_drop_arsq_base/action_stats.npz \
    --results-dir  ../runs/pick_drop_arsq_base \
    2>&1 | tee ../runs/pick_drop_arsq_base/eval.log

python eval_real.py --agent arsq --use-fk-reg "${PICK_EVAL_COMMON[@]}" \
    --snapshot     ../runs/pick_drop_arsq_phys/snapshot.pt \
    --action-stats ../runs/pick_drop_arsq_phys/action_stats.npz \
    --results-dir  ../runs/pick_drop_arsq_phys \
    2>&1 | tee ../runs/pick_drop_arsq_phys/eval.log

python compare_evals.py \
    --results ../runs/pick_drop_arsq_base/eval_results.json \
              ../runs/pick_drop_arsq_phys/eval_results.json \
    --labels  "ARSQ" "Pi-ARSQ" \
    --save    ../runs/pick_drop_eval_comparison.png
```

## Troubleshooting

- **Reflex aborts during training.** Verify `--control-hz` matches the demo
  recording rate, lower `--joint-delta-clip`, and reduce `--velocity-factor`.
- **CUDA out of memory.** The default RGB encoder requires ~7 GB at batch 32.
  Reduce `--batch-size` and `--demo-batch-size` in tandem; the 50/50 sampling
  ratio is preserved automatically.
- **Stale gripper behavior in pick-place.** Use `--freeze-gripper` to remove
  the gripper dimension from policy control; the environment closes the
  gripper at reset and opens it on episode termination.
- **No camera output.** Run `python check_cameras.py` to enumerate Orbbec
  devices and verify each serial.
