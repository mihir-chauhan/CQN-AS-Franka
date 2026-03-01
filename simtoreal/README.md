# Sim-to-Real: CQN-AS on Franka Panda

End-to-end deployment of CQN-AS on a real Franka Emika Panda robot using the **franky** library for robot control and **Intel RealSense** cameras for visual observations.

## Overview

```
simtoreal/
├── __init__.py
├── cameras.py          # Camera abstraction (RealSense, USB, Dummy)
├── real_env.py         # RealFrankaEnv — mirrors the RLBench env interface
├── franka.py           # Original robot homing/utility script
├── train_real.py       # Training on the real robot with kinesthetic demos
├── eval_real.py        # Evaluation from a trained checkpoint
└── README.md           # This file
```

## Architecture

The code mirrors the RLBench environment interface exactly, so the CQN-AS agent (`rlbench_src/cqn_as.py`) works without modification:

| Component | RLBench (sim) | Real Robot |
|---|---|---|
| Robot control | CoppeliaSim via PyRep | franky (`JointWaypointMotion`) |
| Action format | 7 joint deltas + gripper (normalised to [-1,1]) | Same |
| Cameras | RLBench built-in (front, wrist, left_shoulder, right_shoulder) | Intel RealSense / USB cameras |
| Observations | (V, 3×fs, 84, 84) RGB + (8×fs,) low-dim | Same |
| Demos | Pre-recorded RLBench demos | Kinesthetic teaching |

## Prerequisites

### Hardware
- Franka Emika Panda robot with FCI enabled
- **4× Intel RealSense cameras** (or USB webcams) — one per training viewpoint:

  | Camera key | Mounting position | Description |
  |---|---|---|
  | `front` | ~1m in front of robot, chest height | Faces the robot/workspace head-on |
  | `wrist` | Mounted on the gripper/flange | Moves with the end-effector |
  | `left_shoulder` | ~0.5m above the robot's left shoulder | Angled down at the workspace |
  | `right_shoulder` | ~0.5m above the robot's right shoulder | Angled down at the workspace |

  > **Important**: These 4 cameras match the RLBench training config exactly
  > (`cfgs/rlbench_task/default.yaml`). The CNN encoder processes them
  > positionally — **order and number matter**. Swapping or omitting cameras
  > will degrade policy performance.

### Software
```bash
# franky — Franka Control Interface Python bindings
pip install franky-panda

# Intel RealSense (only if using RealSense cameras)
pip install pyrealsense2

# OpenCV for image processing
pip install opencv-python

# Everything else from the main project
pip install torch numpy dm-env gymnasium
```

## Quick Start

### 1. Training (wrist camera only)

```bash
# Step 1: Record kinesthetic demonstrations
# Put the robot in freedrive/gravity-comp mode, then:
python -m simtoreal.train_real \
    --robot-ip 192.168.131.41 \
    --camera-mode wrist \
    --wrist-serial <YOUR_ORBBEC_SERIAL> \
    --num-demos 10 \
    --demo-dir ./demos \
    --save-dir ./runs/real_train \
    --num-train-steps 50000

# Step 2: Resume training (demos already saved)
python -m simtoreal.train_real \
    --robot-ip 192.168.131.41 \
    --camera-mode wrist \
    --wrist-serial <SERIAL> \
    --load-demos-only \
    --demo-dir ./demos \
    --resume ./runs/real_train/snapshot.pt \
    --save-dir ./runs/real_train \
    --num-train-steps 100000
```

### 2. Evaluation (full camera rig)

```bash
python -m simtoreal.eval_real \
    --robot-ip 192.168.131.41 \
    --snapshot ./runs/real_train/snapshot.pt \
    --action-stats ./runs/real_train/action_stats.npz \
    --camera-mode full \
    --camera-serials '{"front":"SN1","wrist":"SN2","left_shoulder":"SN3","right_shoulder":"SN4"}' \
    --num-episodes 10 \
    --save-video
```

### 3. Evaluation with sim-trained checkpoint

If you trained in RLBench sim and want to test on the real robot:

```bash
# First, you need the action_stats from sim training.
# Add this to your sim training script to save them:
#   np.savez("action_stats.npz", **env._action_stats)

python -m simtoreal.eval_real \
    --robot-ip 192.168.131.41 \
    --snapshot /path/to/sim/snapshot.pt \
    --action-stats /path/to/sim/action_stats.npz \
    --camera-mode full \
    --camera-serials '{"front":"SN1","wrist":"SN2","left_shoulder":"SN3","right_shoulder":"SN4"}' \
    --num-episodes 10
```

### 4. Dry run (no robot, no cameras)

```bash
python -m simtoreal.eval_real \
    --snapshot ./runs/real_train/snapshot.pt \
    --action-stats ./runs/real_train/action_stats.npz \
    --camera-mode dummy \
    --dry-run \
    --num-episodes 2
```

## Camera Modes

| Mode | Cameras active | Use case |
|---|---|---|
| `full` (default) | All 4: front, wrist, left_shoulder, right_shoulder | **Training & Eval** — best results |
| `wrist` | wrist Orbbec only; other 3 zero-filled | Degraded performance; only if you truly cannot mount other cameras |
| `dummy` | All zero-filled | Code testing without any hardware |

> **Recommendation**: Always use `full` mode for both training and evaluation.
> A policy trained with all 4 views will not perform well when some are missing.
> If you can only afford 1 camera, retrain in sim with `camera_keys: [wrist]` first.

### Finding camera serial numbers

**RealSense** (for `full` mode):
```bash
rs-enumerate-devices | grep "Serial Number"
```

**Orbbec** (for `wrist` mode): Use the Orbbec SDK or device manager to list connected Orbbec cameras and their serial numbers.

## Action Stats

**Critical**: The policy outputs normalised `[-1, 1]` actions. To convert them to raw joint deltas, you need the `action_stats` (min/max arrays) that define the normalisation range.

- **Training from scratch**: Action stats are automatically computed from kinesthetic demos and saved to `<save_dir>/action_stats.npz`.
- **Sim-to-real transfer**: You must extract and save the action stats from the sim environment during training.

### Saving action stats from sim training

Add this to your RLBench training script after `load_rlbench_demos()`:

```python
np.savez(
    "action_stats.npz",
    min=self.train_env._env._action_stats["min"],
    max=self.train_env._env._action_stats["max"],
)
```

## Safety

| Parameter | Default | Description |
|---|---|---|
| `--joint-delta-clip` | 0.05 rad (~3°) | Max per-step joint delta |
| `--velocity-factor` | 0.15 | Fraction of max robot velocity |
| `--control-hz` | 10 Hz | Control loop rate |
| `--gripper-force` | 20 N | Grasping force |

**Start with very conservative values** and increase gradually:

```bash
# Ultra-safe first test
python -m simtoreal.eval_real \
    --joint-delta-clip 0.02 \
    --velocity-factor 0.05 \
    --control-hz 5 \
    ...
```

## Kinesthetic Demo Recording

1. Enable freedrive / gravity compensation mode on the Franka (via the desk interface or the physical button).
2. Run the training script — it will prompt you before each demo.
3. Physically guide the robot through the task.
4. Press `Ctrl-C` to stop recording each demo early, or let it reach `--demo-max-steps`.
5. Demos are saved as `.npz` files to `--demo-dir`.

## File Descriptions

### `cameras.py`
- `CameraBase` — abstract interface (captures `(3, H, W)` uint8)
- `RealSenseCamera` — wraps pyrealsense2 with auto-resize to 84×84
- `USBCamera` — OpenCV VideoCapture for cheap webcams
- `DummyCamera` — returns zeros (for missing viewpoints)
- `CameraRig` — manages ordered set of cameras matching training config
- Convenience: `make_wrist_only_rig()`, `make_full_rig()`, `make_dummy_rig()`

### `real_env.py`
- `RealFrankaEnv` — drop-in replacement for the RLBench env
  - `reset()` → homes robot, returns initial `TimeStep`
  - `step(action)` → executes normalised action, returns `TimeStep`
  - `record_demo()` → kinesthetic demo recording
  - `extract_action_stats()` / `rescale_demo_actions()` — same as RLBench
  - All the same `*_spec()` methods for observation/action shapes
- `make()` factory function (mirrors `rlbench_env.make()`)

### `train_real.py`
- Full training loop matching `train_cqn_as_rlbench.py`
- Kinesthetic demo collection + replay buffer insertion
- Periodic eval + checkpoint saving

### `eval_real.py`
- Loads snapshot + action stats
- Runs N episodes with human success labelling
- Optional video recording of wrist camera
- Prints success rate summary
