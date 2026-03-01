"""
deploy_real_panda.py
====================
Minimal sim-to-real deployment script for CQN-AS on a real Franka Panda.

This replaces the RLBench simulation loop with:
  1. Real-time camera observations  (RealSense / ZED / etc.)
  2. Joint reading & commanding via `panda_robot` (PandaArm)

Prerequisites
-------------
- ROS Noetic (or Melodic) with `franka_ros_interface` running.
- `pip install panda-robot` (or built from source in your catkin workspace).
- A trained CQN-AS checkpoint (`snapshot.pt`) + its Hydra config.
- Camera(s) producing images that match the training distribution
  (resolution, viewpoint, background – use domain-randomisation or
  fine-tuning to bridge the gap if needed).

Usage
-----
    python deploy_real_panda.py \
        --snapshot /path/to/exp/snapshot.pt \
        --config-dir /path/to/exp/.hydra \
        --max-steps 200
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import yaml

# ---------------------------------------------------------------------------
# ROS / panda_robot  (only imported when actually deploying)
# ---------------------------------------------------------------------------
try:
    import rospy
    from panda_robot import PandaArm

    HAS_PANDA = True
except ImportError:
    HAS_PANDA = False
    print(
        "[WARN] panda_robot not found – you can still dry-run with --dry-run flag."
    )

# ---------------------------------------------------------------------------
# Project imports (CQN-AS agent, utils)
# ---------------------------------------------------------------------------
import utils
from rlbench_src.cqn_as import CQNASAgent


# ========================== helpers ========================================

def load_agent_from_snapshot(
    snapshot_path: str,
    config_dir: str,
    device: str = "cuda",
) -> tuple[CQNASAgent, dict]:
    """Load the trained agent and action-stats from a training snapshot.

    Returns
    -------
    agent : CQNASAgent
    meta   : dict  – holds action_stats, frame_stack, action_sequence, etc.
    """
    # ---- load Hydra config that was used during training -------------------
    hydra_cfg_path = Path(config_dir) / "config.yaml"
    with open(hydra_cfg_path) as f:
        cfg = yaml.safe_load(f)

    # ---- reconstruct shapes from config -----------------------------------
    frame_stack = cfg.get("frame_stack", 8)
    action_sequence = cfg.get("action_sequence", 4)
    num_cameras = 4  # front, wrist, left_shoulder, right_shoulder by default
    camera_h, camera_w = 84, 84
    rgb_obs_shape = (num_cameras, 3 * frame_stack, camera_h, camera_w)
    low_dim_obs_shape = (8 * frame_stack,)  # 7 joints + gripper_open
    action_dim = 8  # 7 joints + 1 gripper
    action_shape = (action_sequence, action_dim)

    agent_cfg = cfg["agent"]

    # ---- build agent -------------------------------------------------------
    agent = CQNASAgent(
        rgb_obs_shape=rgb_obs_shape,
        low_dim_obs_shape=low_dim_obs_shape,
        action_shape=action_shape,
        device=device,
        lr=agent_cfg.get("lr", 5e-5),
        feature_dim=agent_cfg.get("feature_dim", 64),
        hidden_dim=agent_cfg.get("hidden_dim", 512),
        levels=agent_cfg.get("levels", 3),
        bins=agent_cfg.get("bins", 5),
        atoms=agent_cfg.get("atoms", 51),
        v_min=agent_cfg.get("v_min", -2.0),
        v_max=agent_cfg.get("v_max", 2.0),
        bc_lambda=agent_cfg.get("bc_lambda", 1.0),
        bc_margin=agent_cfg.get("bc_margin", 0.01),
        gru_layers=agent_cfg.get("gru_layers", 1),
        rgb_encoder_layers=agent_cfg.get("rgb_encoder_layers", 0),
        use_parallel_impl=agent_cfg.get("use_parallel_impl", False),
        critic_lambda=agent_cfg.get("critic_lambda", 0.1),
        critic_target_tau=agent_cfg.get("critic_target_tau", 0.02),
        critic_target_interval=agent_cfg.get("critic_target_interval", 1),
        weight_decay=agent_cfg.get("weight_decay", 0.1),
        num_expl_steps=0,
        update_every_steps=1,
        stddev_schedule="0.0",  # no exploration noise at deploy time
    )

    # ---- load weights ------------------------------------------------------
    payload = torch.load(snapshot_path, map_location=device)
    # snapshot saves the full agent object; we just need state-dicts
    saved_agent = payload["agent"]
    agent.encoder.load_state_dict(saved_agent.encoder.state_dict())
    agent.critic.load_state_dict(saved_agent.critic.state_dict())
    agent.critic_target.load_state_dict(saved_agent.critic_target.state_dict())
    agent.train(False)
    print("[INFO] Agent loaded from", snapshot_path)

    # ---- meta (action stats, etc.) -----------------------------------------
    # Action stats are stored inside the *environment*, not the agent.
    # You MUST supply them (saved during training) so the normalised [-1,1]
    # policy output can be converted to raw joint deltas.
    meta = dict(
        frame_stack=frame_stack,
        action_sequence=action_sequence,
        action_dim=action_dim,
    )
    return agent, meta


# ---------------------------------------------------------------------------
# Action conversion – mirrors rlbench_env._convert_action_to_raw
# ---------------------------------------------------------------------------

def convert_action_to_raw(
    action: np.ndarray,
    action_stats: dict[str, np.ndarray],
) -> np.ndarray:
    """Map [-1, 1] normalised policy output → raw joint-delta + gripper."""
    action_min = action_stats["min"]
    action_max = action_stats["max"]
    _action_min = action_min - np.fabs(action_min) * 0.2
    _action_max = action_max + np.fabs(action_max) * 0.2
    new_action = (action + 1) / 2.0                             # → [0, 1]
    new_action = new_action * (_action_max - _action_min) + _action_min
    return new_action.astype(np.float32)


# ---------------------------------------------------------------------------
# Camera stub – replace with your real camera capture code
# ---------------------------------------------------------------------------

def capture_cameras(num_cameras: int = 4, h: int = 84, w: int = 84) -> np.ndarray:
    """Return (num_cameras, 3, H, W) uint8 image from real cameras.

    TODO: Replace this stub with your actual camera pipeline
    (e.g. pyrealsense2, pyzed, or ROS image_transport subscriber).
    """
    # Placeholder – return random noise so the script is syntactically runnable
    return np.random.randint(0, 255, (num_cameras, 3, h, w), dtype=np.uint8)


# ---------------------------------------------------------------------------
# Frame-stacking helper
# ---------------------------------------------------------------------------

class FrameStack:
    """Maintains a rolling window of the last `k` observations."""

    def __init__(self, k: int, low_dim_dim: int, rgb_shape: tuple):
        self.k = k
        self._low_dims: list[np.ndarray] = []
        self._rgbs: list[np.ndarray] = []  # list of (V, 3, H, W) arrays
        self.low_dim_dim = low_dim_dim
        self.rgb_shape = rgb_shape  # (V, 3, H, W)

    def reset(self):
        self._low_dims.clear()
        self._rgbs.clear()

    def push(self, low_dim: np.ndarray, rgb: np.ndarray):
        if len(self._low_dims) == 0:
            for _ in range(self.k):
                self._low_dims.append(low_dim)
                self._rgbs.append(rgb)
        else:
            self._low_dims.append(low_dim)
            self._rgbs.append(rgb)
            if len(self._low_dims) > self.k:
                self._low_dims.pop(0)
                self._rgbs.pop(0)

    @property
    def low_dim_obs(self) -> np.ndarray:
        return np.concatenate(self._low_dims, axis=0)  # (8*k,)

    @property
    def rgb_obs(self) -> np.ndarray:
        # Each entry is (V, 3, H, W); stack along channel dim → (V, 3*k, H, W)
        return np.concatenate(self._rgbs, axis=1)


# ========================== main loop ======================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", type=str, required=True,
                        help="Path to snapshot.pt")
    parser.add_argument("--config-dir", type=str, required=True,
                        help="Path to .hydra/ directory from the training run")
    parser.add_argument("--action-stats", type=str, default=None,
                        help="Path to saved action_stats.npz (min/max arrays). "
                             "If not provided, uses default ±0.2 range.")
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--control-hz", type=float, default=10.0,
                        help="Control loop frequency (Hz)")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dry-run", action="store_true",
                        help="Run without a real robot (prints actions instead)")
    parser.add_argument("--joint-delta-clip", type=float, default=0.05,
                        help="Safety: max absolute joint delta per step (rad)")
    args = parser.parse_args()

    # ---- load agent --------------------------------------------------------
    agent, meta = load_agent_from_snapshot(
        args.snapshot, args.config_dir, args.device
    )
    frame_stack = meta["frame_stack"]
    action_sequence = meta["action_sequence"]
    action_dim = meta["action_dim"]

    # ---- load action stats -------------------------------------------------
    if args.action_stats is not None:
        data = np.load(args.action_stats)
        action_stats = {"min": data["min"], "max": data["max"]}
    else:
        print("[WARN] No --action-stats provided; using default ±0.2 range.")
        action_stats = {
            "min": np.array([-0.2] * 7 + [0.0], dtype=np.float32),
            "max": np.array([0.2] * 7 + [1.0], dtype=np.float32),
        }

    # ---- init robot --------------------------------------------------------
    robot = None
    if not args.dry_run:
        assert HAS_PANDA, "panda_robot is required for real deployment"
        rospy.init_node("cqn_as_deploy", anonymous=True)
        robot = PandaArm()
        robot.move_to_neutral()
        gripper = robot.get_gripper()
        gripper.home_joints()
        print("[INFO] Robot at neutral pose. Starting policy rollout...")

    # ---- frame-stack -------------------------------------------------------
    fs = FrameStack(k=frame_stack, low_dim_dim=8, rgb_shape=(4, 3, 84, 84))

    # ---- initial observation -----------------------------------------------
    def get_obs():
        """Read joint state + camera images from the real robot."""
        if robot is not None:
            joint_pos = np.array(robot.angles(), dtype=np.float32)  # (7,)
            gripper_open = np.array(
                [1.0 if gripper.is_open() else 0.0], dtype=np.float32
            )
        else:
            joint_pos = np.zeros(7, dtype=np.float32)
            gripper_open = np.array([1.0], dtype=np.float32)
        low_dim = np.concatenate([joint_pos, gripper_open])  # (8,)
        rgb = capture_cameras()  # (4, 3, 84, 84)
        return low_dim, rgb

    low_dim, rgb = get_obs()
    fs.reset()
    fs.push(low_dim, rgb)

    dt = 1.0 / args.control_hz
    step = 0
    sub_step = 0  # index within current action sequence
    action_seq = None  # will hold (action_sequence, action_dim)

    print(f"[INFO] Running for up to {args.max_steps} steps at {args.control_hz} Hz")
    print(f"[INFO] Action sequence length = {action_sequence}")
    print(f"[INFO] Joint-delta safety clip = ±{args.joint_delta_clip} rad")

    while step < args.max_steps:
        t0 = time.time()

        # ---- query policy every `action_sequence` steps --------------------
        if action_seq is None or sub_step >= action_sequence:
            with torch.no_grad(), utils.eval_mode(agent):
                raw_action = agent.act(
                    fs.rgb_obs,
                    fs.low_dim_obs,
                    step=999999,       # large step → no exploration
                    eval_mode=True,
                )
            # raw_action shape: (action_sequence * action_dim,)
            action_seq = raw_action.reshape(action_sequence, action_dim)
            sub_step = 0

        # ---- convert normalised action → raw joint deltas ------------------
        norm_action = action_seq[sub_step]  # (8,) in [-1, 1]
        raw = convert_action_to_raw(norm_action, action_stats)

        joint_delta = raw[:7]   # 7 joint deltas (radians)
        gripper_cmd = raw[7]    # 0 or 1  (thresholded below)

        # ---- safety clipping -----------------------------------------------
        joint_delta = np.clip(joint_delta, -args.joint_delta_clip,
                              args.joint_delta_clip)

        # ---- execute on real robot -----------------------------------------
        if robot is not None:
            current_joints = np.array(robot.angles(), dtype=np.float32)
            target_joints = current_joints + joint_delta
            # PandaArm.move_to_joint_position expects a list/array of 7 values
            robot.move_to_joint_position(target_joints.tolist())

            # Gripper
            if gripper_cmd > 0.5:
                gripper.open()
            else:
                gripper.close()
        else:
            print(
                f"  [dry-run] step={step:3d}  Δjoints={np.round(joint_delta, 4)}  "
                f"gripper={'open' if gripper_cmd > 0.5 else 'close'}"
            )

        # ---- read next observation -----------------------------------------
        low_dim, rgb = get_obs()
        fs.push(low_dim, rgb)

        sub_step += 1
        step += 1

        # ---- control-rate sleep --------------------------------------------
        elapsed = time.time() - t0
        if elapsed < dt:
            time.sleep(dt - elapsed)

    print(f"[INFO] Rollout finished ({step} steps).")
    if robot is not None:
        robot.move_to_neutral()
        print("[INFO] Robot returned to neutral.")


if __name__ == "__main__":
    main()
