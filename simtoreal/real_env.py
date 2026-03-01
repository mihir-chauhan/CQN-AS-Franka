"""
real_env.py — Real Franka Panda environment that mirrors the RLBench env interface.

This wraps a franky Robot + Gripper plus a CameraRig to produce the same
TimeStep / ExtendedTimeStep / spec objects that the CQN-AS agent expects.

Key design decisions
--------------------
- Actions are 8-dim: 7 joint deltas (radians) + 1 gripper (0=close, 1=open),
  normalised to [-1, 1] by the same action_stats used in RLBench training.
- Low-dim obs = 7 joint positions + gripper_open flag  (8 dims).
- RGB obs = (V, 3*frame_stack, H, W) from CameraRig.
- step() applies the delta via franky JointWaypointMotion for smooth execution.
"""

from __future__ import annotations

import time
from collections import deque
from pathlib import Path
from typing import Any, Dict, NamedTuple, Optional

import numpy as np

# dm_env provides StepType and specs used throughout CQN-AS
from dm_env import StepType, specs

# franky for robot control
from franky import (
    Affine,
    Gripper,
    JointWaypoint,
    JointWaypointMotion,
    RelativeDynamicsFactor,
    Robot,
)

from simtoreal.cameras import CAMERA_H, CAMERA_KEYS, CAMERA_W, NUM_CAMERAS, CameraRig

# ---------------------------------------------------------------------------
# Re-use the same NamedTuples as RLBench env so all downstream code works
# ---------------------------------------------------------------------------


class TimeStep(NamedTuple):
    step_type: Any
    reward: Any
    discount: Any
    rgb_obs: Any
    low_dim_obs: Any
    demo: Any

    def first(self):
        return self.step_type == StepType.FIRST

    def mid(self):
        return self.step_type == StepType.MID

    def last(self):
        return self.step_type == StepType.LAST

    def __getitem__(self, attr):
        if isinstance(attr, str):
            return getattr(self, attr)
        else:
            return tuple.__getitem__(self, attr)


class ExtendedTimeStep(NamedTuple):
    step_type: Any
    reward: Any
    discount: Any
    rgb_obs: Any
    low_dim_obs: Any
    action: Any
    demo: Any

    def first(self):
        return self.step_type == StepType.FIRST

    def mid(self):
        return self.step_type == StepType.MID

    def last(self):
        return self.step_type == StepType.LAST

    def __getitem__(self, attr):
        if isinstance(attr, str):
            return getattr(self, attr)
        else:
            return tuple.__getitem__(self, attr)


class ExtendedTimeStepWrapper:
    """Same wrapper as in rlbench_env.py — adds action field to TimeStep."""

    def __init__(self, env):
        self._env = env

    def reset(self):
        time_step = self._env.reset()
        return self._augment_time_step(time_step)

    def step(self, action):
        time_step = self._env.step(action)
        return self._augment_time_step(time_step, action)

    def _augment_time_step(self, time_step, action=None):
        if action is None:
            action_spec = self.action_spec()
            action = np.zeros(action_spec.shape, dtype=action_spec.dtype)
        return ExtendedTimeStep(
            rgb_obs=time_step.rgb_obs,
            low_dim_obs=time_step.low_dim_obs,
            step_type=time_step.step_type,
            action=action,
            reward=time_step.reward,
            discount=time_step.discount,
            demo=time_step.demo,
        )

    def low_dim_observation_spec(self):
        return self._env.low_dim_observation_spec()

    def rgb_observation_spec(self):
        return self._env.rgb_observation_spec()

    def low_dim_raw_observation_spec(self):
        return self._env.low_dim_raw_observation_spec()

    def rgb_raw_observation_spec(self):
        return self._env.rgb_raw_observation_spec()

    def action_spec(self):
        return self._env.action_spec()

    def __getattr__(self, name):
        return getattr(self._env, name)


# ============================================================================
# Main environment
# ============================================================================

# Sensible default home pose  (same as IMAGE_Q from franka.py)
DEFAULT_HOME_Q = [
    -np.pi / 4,
    -np.pi / 4,
    0.0,
    -3 * np.pi / 4,
    0.0,
    np.pi / 2,
    np.pi / 4,
]

HALF_VEL = RelativeDynamicsFactor(0.1, 0.01, 0.01)


class RealFrankaEnv:
    """
    Real robot environment matching the RLBench env interface.

    Parameters
    ----------
    robot_ip : str
        IP address of the Franka controller (e.g. "192.168.131.41").
    camera_rig : CameraRig
        Configured camera setup (wrist-only, full, or dummy).
    episode_length : int
        Max steps before auto-truncation.
    frame_stack : int
        Number of frames to stack for obs.
    home_q : list
        Joint configuration for the home / reset pose.
    joint_delta_clip : float
        Safety clamp on per-step joint delta (rad).
    velocity_factor : float
        Fraction of max velocity for motions (0.0–1.0).
    acceleration_factor : float
        Fraction of max acceleration (0.0–1.0).
    jerk_factor : float
        Fraction of max jerk (0.0–1.0).
    action_stats_path : str or None
        Path to action_stats.npz with "min" and "max" arrays.
        If None, uses conservative ±0.05 rad default.
    gripper_speed : float
        Gripper open/close speed (m/s).
    gripper_force : float
        Gripper grasping force (N).
    """

    def __init__(
        self,
        robot_ip: str,
        camera_rig: CameraRig,
        episode_length: int = 200,
        frame_stack: int = 8,
        home_q: list | None = None,
        joint_delta_clip: float = 0.05,
        velocity_factor: float = 0.1,
        acceleration_factor: float = 0.01,
        jerk_factor: float = 0.01,
        action_stats_path: str | None = None,
        gripper_speed: float = 0.1,
        gripper_force: float = 20.0,
    ):
        self._episode_length = episode_length
        self._frame_stack = frame_stack
        self._camera_rig = camera_rig
        self._joint_delta_clip = joint_delta_clip
        self._gripper_speed = gripper_speed
        self._gripper_force = gripper_force
        self._home_q = home_q or DEFAULT_HOME_Q

        # Camera layout — must match training config
        assert len(camera_rig.camera_keys) == NUM_CAMERAS, (
            f"Expected {NUM_CAMERAS} cameras {CAMERA_KEYS}, "
            f"got {len(camera_rig.camera_keys)} {camera_rig.camera_keys}"
        )
        assert camera_rig.camera_keys == CAMERA_KEYS, (
            f"Camera key order must match training: {CAMERA_KEYS}, "
            f"got {camera_rig.camera_keys}"
        )
        self._num_cameras = NUM_CAMERAS
        self._camera_h = CAMERA_H
        self._camera_w = CAMERA_W

        # Connect to robot
        self._robot = Robot(robot_ip)
        self._gripper = Gripper(robot_ip)
        self._robot.relative_dynamics_factor = RelativeDynamicsFactor(
            velocity=velocity_factor, acceleration=acceleration_factor, jerk=jerk_factor,
        )
        self._robot.set_collision_behavior(50, 50)
        self._robot.recover_from_errors()
        print(f"[RealFrankaEnv] Connected to {robot_ip}")

        # Frame-stacking deques
        self._low_dim_obses: deque = deque([], maxlen=frame_stack)
        self._frames: dict[str, deque] = {
            k: deque([], maxlen=frame_stack) for k in camera_rig.camera_keys
        }

        # Step counter
        self._step_counter = 0

        # Action stats (normalisation <-> raw)
        self._action_stats = self._load_action_stats(action_stats_path)

        # Build spaces
        self._build_spaces()

        # Gripper state tracking
        self._gripper_is_open = True

    # ------------------------------------------------------------------
    # Specs (mirror RLBench interface)
    # ------------------------------------------------------------------

    def low_dim_observation_spec(self):
        shape = (8 * self._frame_stack,)
        return specs.Array(shape, np.float32, "low_dim_obs")

    def low_dim_raw_observation_spec(self):
        return specs.Array((8,), np.float32, "low_dim_obs")

    def rgb_observation_spec(self):
        shape = (
            self._num_cameras,
            3 * self._frame_stack,
            self._camera_h,
            self._camera_w,
        )
        return specs.Array(shape, np.uint8, "rgb_obs")

    def rgb_raw_observation_spec(self):
        shape = (self._num_cameras, 3, self._camera_h, self._camera_w)
        return specs.Array(shape, np.uint8, "rgb_obs")

    def action_spec(self):
        return specs.Array((8,), np.float32, "action")

    # ------------------------------------------------------------------
    # Core env methods
    # ------------------------------------------------------------------

    def reset(self) -> TimeStep:
        """Move to home, open gripper, return initial observation."""
        # Clear frame stacks
        self._low_dim_obses.clear()
        for frames in self._frames.values():
            frames.clear()

        # Home robot
        self._robot.recover_from_errors()
        time.sleep(1.0)  # let robot fully settle before planning
        motion = JointWaypointMotion(
            [JointWaypoint(self._home_q)],
            HALF_VEL,
        )
        try:
            self._robot.move(motion)
        except Exception as e:
            print(f"[RealFrankaEnv] Homing failed: {e}")
            print("[RealFrankaEnv] Recovering and retrying...")
            self._robot.recover_from_errors()
            time.sleep(2.0)
            motion = JointWaypointMotion(
                [JointWaypoint(self._home_q)],
                HALF_VEL,
            )
            self._robot.move(motion)
        self._gripper.open(self._gripper_speed)
        self._gripper_is_open = True
        time.sleep(0.5)  # settle

        self._step_counter = 0
        obs = self._get_obs()
        return TimeStep(
            rgb_obs=obs["rgb_obs"],
            low_dim_obs=obs["low_dim_obs"],
            step_type=StepType.FIRST,
            reward=0.0,
            discount=1.0,
            demo=0.0,
        )

    def step(self, action: np.ndarray) -> TimeStep:
        """
        Execute a normalised [-1, 1] action on the real robot.

        action : (8,) float32 — 7 joint deltas + 1 gripper, normalised.
        """
        raw = self._convert_action_to_raw(action)
        joint_delta = raw[:7]
        gripper_cmd = raw[7]

        # Safety clamp
        joint_delta = np.clip(
            joint_delta, -self._joint_delta_clip, self._joint_delta_clip
        )

        # Compute target joint positions
        current_q = np.array(self._robot.current_joint_state.position, dtype=np.float64)
        target_q = current_q + joint_delta

        # Execute joint motion (single waypoint — franky handles current→target)
        try:
            motion = JointWaypointMotion(
                [JointWaypoint(target_q.tolist())],
                HALF_VEL,
            )
            self._robot.move(motion)
        except Exception as e:
            print(f"[RealFrankaEnv] Motion error: {e}")
            self._robot.recover_from_errors()
            time.sleep(0.5)

        # Gripper
        if gripper_cmd > 0.5 and not self._gripper_is_open:
            self._gripper.open(self._gripper_speed)
            self._gripper_is_open = True
        elif gripper_cmd <= 0.5 and self._gripper_is_open:
            self._gripper.grasp(
                0.0,
                self._gripper_speed,
                self._gripper_force,
                epsilon_inner=1.0,
                epsilon_outer=1.0,
            )
            self._gripper_is_open = False

        self._step_counter += 1

        # Observe
        obs = self._get_obs()

        # Truncation
        truncated = self._step_counter >= self._episode_length
        # On the real robot we don't have automatic success detection,
        # so terminated is always False (reward must come from external signal)
        terminated = False

        if terminated or truncated:
            step_type = StepType.LAST
        else:
            step_type = StepType.MID
        discount = float(1 - terminated)

        return TimeStep(
            rgb_obs=obs["rgb_obs"],
            low_dim_obs=obs["low_dim_obs"],
            step_type=step_type,
            reward=0.0,  # override externally if needed
            discount=discount,
            demo=0.0,
        )

    # ------------------------------------------------------------------
    # Kinesthetic demonstration recording
    # ------------------------------------------------------------------

    def record_demo(
        self,
        hz: float = 10.0,
        max_steps: int = 500,
        reward_at_end: float = 1.0,
    ) -> list[ExtendedTimeStep]:
        """
        Record a kinesthetic demonstration.

        Put the robot in gravity-compensation / freedrive mode externally
        (e.g. press the button on the robot), then call this.  It records
        joint states + camera images at the given rate.

        Returns a list of ExtendedTimeSteps (same format as RLBench demos).
        """
        print(
            f"[Demo] Recording at {hz} Hz for up to {max_steps} steps.  "
            "Press Ctrl-C to stop early."
        )
        self._low_dim_obses.clear()
        for frames in self._frames.values():
            frames.clear()

        dt = 1.0 / hz
        joint_positions_list: list[np.ndarray] = []
        gripper_open_list: list[float] = []
        obs_list: list[dict] = []

        try:
            for i in range(max_steps):
                t0 = time.time()
                obs = self._get_obs()
                q = np.array(
                    self._robot.current_joint_state.position, dtype=np.float32
                )
                g = 1.0 if self._gripper_is_open else 0.0
                joint_positions_list.append(q)
                gripper_open_list.append(g)
                obs_list.append(obs)
                elapsed = time.time() - t0
                if elapsed < dt:
                    time.sleep(dt - elapsed)
        except KeyboardInterrupt:
            print(f"[Demo] Stopped early at step {len(obs_list)}")

        # Convert to ExtendedTimeSteps with delta-joint actions
        timesteps: list[ExtendedTimeStep] = []
        for i in range(len(obs_list)):
            if i == 0:
                action = np.zeros(8, dtype=np.float32)
                step_type = StepType.FIRST
                reward, discount = 0.0, 1.0
            else:
                delta = joint_positions_list[i] - joint_positions_list[i - 1]
                gripper = gripper_open_list[i]
                action = np.concatenate([delta, [gripper]]).astype(np.float32)
                if i == len(obs_list) - 1:
                    step_type = StepType.LAST
                    reward, discount = reward_at_end, 0.0
                else:
                    step_type = StepType.MID
                    reward, discount = 0.0, 1.0

            timesteps.append(
                ExtendedTimeStep(
                    rgb_obs=obs_list[i]["rgb_obs"],
                    low_dim_obs=obs_list[i]["low_dim_obs"],
                    step_type=step_type,
                    action=action,
                    reward=reward,
                    discount=discount,
                    demo=1.0,
                )
            )
        print(f"[Demo] Recorded {len(timesteps)} steps")
        return timesteps

    # ------------------------------------------------------------------
    # Action stats (same logic as rlbench_env.py)
    # ------------------------------------------------------------------

    def extract_action_stats(
        self, demos: list[list[ExtendedTimeStep]]
    ) -> dict[str, np.ndarray]:
        actions = []
        for demo in demos:
            for ts in demo:
                actions.append(ts.action)
        actions = np.stack(actions)
        action_max = np.hstack([np.max(actions, 0)[:-1], 1])
        action_min = np.hstack([np.min(actions, 0)[:-1], 0])
        return {"max": action_max, "min": action_min}

    def set_action_stats(self, stats: dict[str, np.ndarray]):
        self._action_stats = stats

    def get_action_stats(self) -> dict[str, np.ndarray]:
        return self._action_stats

    def save_action_stats(self, path: str):
        np.savez(path, **self._action_stats)
        print(f"[RealFrankaEnv] Action stats saved to {path}")

    def rescale_demo_actions(
        self, demo: list[ExtendedTimeStep]
    ) -> list[ExtendedTimeStep]:
        return [ts._replace(action=self._convert_action_from_raw(ts.action))
                for ts in demo]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_spaces(self):
        """Pre-compute action / observation space objects."""
        # Action space (normalised)
        self._action_shape = (8,)

    def _load_action_stats(self, path: str | None) -> dict[str, np.ndarray]:
        if path is not None:
            data = np.load(path)
            stats = {"min": data["min"].astype(np.float32),
                     "max": data["max"].astype(np.float32)}
            print(f"[RealFrankaEnv] Loaded action stats from {path}")
            return stats
        # Conservative defaults
        print("[RealFrankaEnv] Using default action stats (±0.05 rad)")
        return {
            "min": np.array([-0.05] * 7 + [0.0], dtype=np.float32),
            "max": np.array([0.05] * 7 + [1.0], dtype=np.float32),
        }

    def _convert_action_to_raw(self, action: np.ndarray) -> np.ndarray:
        """[-1, 1] → raw joint deltas + gripper  (same as rlbench_env)."""
        action = np.clip(action, -1.0, 1.0)
        a_min = self._action_stats["min"]
        a_max = self._action_stats["max"]
        _a_min = a_min - np.fabs(a_min) * 0.2
        _a_max = a_max + np.fabs(a_max) * 0.2
        raw = (action + 1) / 2.0  # → [0, 1]
        raw = raw * (_a_max - _a_min) + _a_min
        return raw.astype(np.float32)

    def _convert_action_from_raw(self, action: np.ndarray) -> np.ndarray:
        """Raw joint deltas + gripper → [-1, 1]  (same as rlbench_env)."""
        a_min = self._action_stats["min"]
        a_max = self._action_stats["max"]
        _a_min = a_min - np.fabs(a_min) * 0.2
        _a_max = a_max + np.fabs(a_max) * 0.2
        norm = (action - _a_min) / (_a_max - _a_min + 1e-8)  # → [0, 1]
        norm = norm * 2 - 1  # → [-1, 1]
        return norm.astype(np.float32)

    def _get_obs(self) -> dict[str, np.ndarray]:
        """Read joints + cameras, apply frame stacking."""
        # Low-dim
        q = np.array(self._robot.current_joint_state.position, dtype=np.float32)
        g = np.array([1.0 if self._gripper_is_open else 0.0], dtype=np.float32)
        low_dim = np.concatenate([q, g])  # (8,)

        if len(self._low_dim_obses) == 0:
            for _ in range(self._frame_stack):
                self._low_dim_obses.append(low_dim)
        else:
            self._low_dim_obses.append(low_dim)

        # RGB — one capture per camera
        all_frames = self._camera_rig.capture_all()  # (V, 3, H, W)
        for idx, key in enumerate(self._camera_rig.camera_keys):
            pixels = all_frames[idx]  # (3, H, W)
            if len(self._frames[key]) == 0:
                for _ in range(self._frame_stack):
                    self._frames[key].append(pixels)
            else:
                self._frames[key].append(pixels)

        return {
            "low_dim_obs": np.concatenate(list(self._low_dim_obses), axis=0),
            "rgb_obs": np.stack(
                [
                    np.concatenate(list(self._frames[k]), axis=0)
                    for k in self._camera_rig.camera_keys
                ],
                axis=0,
            ),
        }

    def close(self):
        self._camera_rig.close()
        print("[RealFrankaEnv] Closed.")


# ============================================================================
# Factory (mirrors rlbench_env.make)
# ============================================================================

def make(
    robot_ip: str,
    camera_rig: CameraRig,
    episode_length: int = 200,
    frame_stack: int = 8,
    home_q: list | None = None,
    joint_delta_clip: float = 0.05,
    velocity_factor: float = 0.1,
    acceleration_factor: float = 0.01,
    jerk_factor: float = 0.01,
    action_stats_path: str | None = None,
    gripper_speed: float = 0.1,
    gripper_force: float = 20.0,
) -> ExtendedTimeStepWrapper:
    env = RealFrankaEnv(
        robot_ip=robot_ip,
        camera_rig=camera_rig,
        episode_length=episode_length,
        frame_stack=frame_stack,
        home_q=home_q,
        joint_delta_clip=joint_delta_clip,
        velocity_factor=velocity_factor,
        acceleration_factor=acceleration_factor,
        jerk_factor=jerk_factor,
        action_stats_path=action_stats_path,
        gripper_speed=gripper_speed,
        gripper_force=gripper_force,
    )
    return ExtendedTimeStepWrapper(env)
