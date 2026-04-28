# Real-Robot Validation — Franka Emika Panda

End-to-end deployment of `Pi-ARSQ`, `ARSQ`, and `CQN-AS` on a real Franka Emika
Panda manipulator. This directory provides the hardware-side counterpart to the
RLBench experiments in `phys_fk_continuous_control-main/rlbench/`, validating
that the geometric regularizer transfers from simulation to a physical robot
(Section V of the paper).

## Layout

```
simtoreal/
├── cameras.py            # Camera abstractions: Orbbec Femto Mega, USB, dummy
├── real_env.py           # RealFrankaEnv — mirrors the RLBench env interface
├── train_real.py         # On-robot / offline training entrypoint
├── eval_real.py          # Evaluation from a trained checkpoint
├── record_waypoints.py   # Freedrive demonstration recording (joint waypoints)
├── teleop.py             # Optional 6-DoF teleoperation utility
├── save_goal_pose.py     # Persist the success-detection goal pose
├── check_cameras.py      # Visual sanity check for connected Orbbec cameras
├── plot_training.py      # Per-run training curve plot
├── plot_comparison.py    # Cross-run training comparison
├── compare_evals.py      # Cross-run evaluation comparison
├── compare_runs.sh       # Helper script for the 3-way real-robot comparison
├── franka.py             # Standalone homing utility
├── partialhome*.npy      # Reset waypoints (joint-space)
├── pickuppose.npy        # Pick-place grasp-ready pose
├── image_45.npy          # Reference home pose
├── goal_pose.json        # Success-detection target pose (XYZ + quaternion)
└── RUNBOOK.md            # Full reproducible procedure for paper experiments
```

## Hardware

- Franka Emika Panda with FCI enabled and a Franka Hand gripper.
- 4× Orbbec Femto Mega RGB-D cameras (the implementation also supports Intel
  RealSense and generic USB sources). The four logical viewpoints used by the
  encoder are `front`, `wrist`, `left_shoulder`, `right_shoulder`. Order is
  positional and must match the RLBench training configuration in
  `cfgs/rlbench_task/default.yaml`.

## Software

```bash
# Robot control
pip install franky-panda

# Cameras (choose what you have)
pip install pyorbbecsdk    # Orbbec Femto Mega
pip install pyrealsense2   # Intel RealSense

# Vision / RL stack (already covered by the root conda_env.yml)
pip install opencv-python torch dm-env gymnasium numpy
```

## Reproducible Procedure

The complete command sequence used for the paper's real-robot experiments —
demonstration capture, three-way comparison (`CQN-AS`, `ARSQ`, `Pi-ARSQ`),
offline training, evaluation, and plotting — is documented in
[`RUNBOOK.md`](RUNBOOK.md). The summary below covers the most common
workflows.

### 1. Record waypoint demonstrations

Put the robot in freedrive mode and capture joint-space trajectories:

```bash
python record_waypoints.py \
    --robot-ip <IP> --num-demos 10 --hz 10 \
    --save-dir ../runs/<task>/waypoints
```

### 2. Replay waypoints to capture camera demos

The environment homes, optionally pauses for object placement (`--task-mode
pick-place`), then replays each waypoint trajectory while recording camera
frames into `--demo-dir`.

```bash
python train_real.py --agent arsq \
    --camera-mode orbbec \
    --orbbec-serials '{"front":"...","wrist":"...","left_shoulder":"...","right_shoulder":"..."}' \
    --demo-mode waypoint \
    --waypoint-dir ../runs/<task>/waypoints \
    --demo-dir     ../runs/<task>/demos \
    --save-dir     ../runs/<task>
```

### 3. Train (offline, demo-only)

Robot is connected only for the action spec; no rollouts occur. This is the
configuration used to produce the real-robot rows of Section V.

```bash
python train_real.py --agent arsq --use-fk-reg \
    --fk-weight 1.0 --kappa 0.1 --nu 0.01 --num-walks 10 \
    --load-demos-only --offline \
    --demo-dir ../runs/<task>/demos \
    --num-train-steps 10000 \
    --batch-size 32 --demo-batch-size 256 \
    --save-dir ../runs/<task>/pi_arsq
```

### 4. Evaluate

```bash
python eval_real.py --agent arsq --use-fk-reg \
    --snapshot     ../runs/<task>/pi_arsq/snapshot.pt \
    --action-stats ../runs/<task>/pi_arsq/action_stats.npz \
    --goal-pose    goal_pose.json \
    --camera-mode orbbec \
    --orbbec-serials '{"front":"...","wrist":"...","left_shoulder":"...","right_shoulder":"..."}' \
    --num-episodes 10
```

## Action Statistics

The policy emits actions normalized to `[-1, 1]`. To convert to joint deltas,
the inverse normalization uses `action_stats.npz` (per-joint min/max) computed
from the demonstrations. These are saved to `<save-dir>/action_stats.npz`
during training and must be passed to `eval_real.py` via `--action-stats`.

## Safety

| Parameter             | Default      | Description                                   |
|-----------------------|--------------|-----------------------------------------------|
| `--joint-delta-clip`  | 0.05 rad     | Maximum per-step joint delta                  |
| `--velocity-factor`   | 0.15         | Fraction of maximum joint velocity            |
| `--control-hz`        | 10 Hz        | Control loop rate (must match `demo-hz`)      |
| `--gripper-force`     | 20 N         | Grasping force                                |

Always begin with conservative values; the Franka libfranka layer enforces
hard reflex limits, but exceeding them aborts the episode.

## Runtime Controls

| Key       | Behavior                                             |
|-----------|------------------------------------------------------|
| `Ctrl+X`  | Abort the current episode and trigger a reset       |
| `Ctrl+C`  | Save snapshot and exit cleanly                      |
