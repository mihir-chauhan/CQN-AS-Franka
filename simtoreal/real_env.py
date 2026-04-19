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
    CartesianMotion,
    Gripper,
    JointWaypoint,
    JointWaypointMotion,
    ReferenceType,
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

    def reset(self, *args, **kwargs):
        time_step = self._env.reset(*args, **kwargs)
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

# Load home pose from image_45.npy (the tested home position)
_HOME_NPY = Path(__file__).resolve().parent / "image_45.npy"
DEFAULT_HOME_Q = np.load(str(_HOME_NPY)).tolist()

# Load partial-home (safe intermediate waypoint before full home)
_PARTIAL_HOME_NPY = Path(__file__).resolve().parent / "partialhome.npy"
if _PARTIAL_HOME_NPY.exists():
    DEFAULT_PARTIAL_HOME_Q: list | None = np.load(str(_PARTIAL_HOME_NPY)).tolist()
else:
    DEFAULT_PARTIAL_HOME_Q = None

# Slow dynamics for homing only (safe return to start)
HOME_VEL = RelativeDynamicsFactor(0.15, 0.05, 0.05)

# ---------------------------------------------------------------------------
# Safety limits
# ---------------------------------------------------------------------------

# Panda joint position limits (radians) — from Franka documentation
PANDA_JOINT_MIN = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
PANDA_JOINT_MAX = np.array([ 2.8973,  1.7628,  2.8973, -0.0698,  2.8973,  3.7525,  2.8973])

# Cartesian workspace box (metres, in robot base frame).
# Only the z lower bound is enforced (table surface).
# *** TUNE THESE FOR YOUR SETUP using teleop EE readouts ***
WORKSPACE_MIN = np.array([-10.0, -10.0,  0.266])  # only z_min matters (table height)
WORKSPACE_MAX = np.array([ 10.0,  10.0,  10.0 ])  # no upper bound enforced


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
        Joint configuration for the final / home reset pose. Used only if
        ``home_sequence`` is not provided (kept for backward compatibility).
    home_sequence : list[list[float]] or None
        Ordered list of joint configurations executed one after another
        during reset. If None, defaults to
        ``[DEFAULT_PARTIAL_HOME_Q, home_q or DEFAULT_HOME_Q]`` (skipping
        the partial-home waypoint if its .npy is missing).
    gripper_at_reset : str
        "open" (default) or "closed" — final gripper state at the end of
        reset, after any human-in-the-loop pause.
    pause_for_human : bool
        If True, after running the home sequence, block on ``input()`` so
        an operator can place an object in the gripper / reset the scene
        before the episode starts. The final gripper state is set *after*
        the pause.
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
        velocity_factor: float = 0.4,
        acceleration_factor: float = 0.15,
        jerk_factor: float = 0.1,
        action_stats_path: str | None = None,
        gripper_speed: float = 0.1,
        gripper_force: float = 20.0,
        home_sequence: list | None = None,
        gripper_at_reset: str = "open",
        pause_for_human: bool = False,
        freeze_gripper: bool = False,
    ):
        self._episode_length = episode_length
        self._frame_stack = frame_stack
        self._camera_rig = camera_rig
        self._joint_delta_clip = joint_delta_clip
        self._gripper_speed = gripper_speed
        self._gripper_force = gripper_force
        self._home_q = home_q or DEFAULT_HOME_Q

        if home_sequence is None:
            seq = []
            if DEFAULT_PARTIAL_HOME_Q is not None:
                seq.append(list(DEFAULT_PARTIAL_HOME_Q))
            seq.append(list(self._home_q))
            self._home_sequence = seq
        else:
            self._home_sequence = [list(q) for q in home_sequence]
            if not self._home_sequence:
                raise ValueError("home_sequence must be non-empty")

        if gripper_at_reset not in ("open", "closed"):
            raise ValueError(
                f"gripper_at_reset must be 'open' or 'closed', got {gripper_at_reset!r}"
            )
        self._gripper_at_reset = gripper_at_reset
        self._pause_for_human = bool(pause_for_human)
        self._freeze_gripper = bool(freeze_gripper)

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
        self._dynamics = RelativeDynamicsFactor(
            velocity=velocity_factor, acceleration=acceleration_factor, jerk=jerk_factor,
        )
        self._robot.relative_dynamics_factor = self._dynamics
        self._robot.set_collision_behavior(50, 50)
        self._robot.recover_from_errors()
        print(f"[RealFrankaEnv] Connected to {robot_ip} "
              f"(vel={velocity_factor}, accel={acceleration_factor}, jerk={jerk_factor})")

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

    def reset(self, skip_home: bool = False) -> TimeStep:
        """Move through the configured home sequence and return obs.

        Flow:
          1. Open gripper (always, so we never drag an object home).
          2. Walk through ``self._home_sequence`` in order.
          3. If ``pause_for_human`` is set, block on input() so the
             operator can load an object / reset the scene.
          4. Set final gripper state per ``gripper_at_reset``.

        If ``skip_home`` is True, steps 1 and 2 are skipped — use for
        the initial reset when the robot is already in the desired
        starting pose. The pause and final-gripper-state steps still run.
        """
        # Clear frame stacks
        self._low_dim_obses.clear()
        for frames in self._frames.values():
            frames.clear()

        if skip_home:
            print("[RealFrankaEnv] skip_home=True — skipping home sequence.")

        # Open gripper BEFORE homing so the robot doesn't drag an object
        if not skip_home:
            self._robot.recover_from_errors()
            if not self._gripper_is_open:
                self._gripper.open(self._gripper_speed)
                self._gripper_is_open = True
                time.sleep(0.3)
            time.sleep(0.5)  # let robot settle

        # ---------- Run the configured home sequence ----------
        last_idx = len(self._home_sequence) - 1
        _home_iter = enumerate(self._home_sequence) if not skip_home else iter(())
        for i, q in _home_iter:
            try:
                motion = JointWaypointMotion([JointWaypoint(q)], HOME_VEL)
                self._robot.move(motion)
                self._wait_for_home(q)
            except Exception as e:
                print(f"[RealFrankaEnv] Home step {i} failed: {e}")
                print("[RealFrankaEnv] Recovering and retrying...")
                try:
                    self._robot.recover_from_errors()
                except Exception as rec_e:
                    print(f"[RealFrankaEnv] recover_from_errors failed: {rec_e}")
                time.sleep(1.0)
                # On final step, retry once; on intermediate steps skip.
                # Swallow any retry failure — reflex aborts typically fire
                # after the robot is already near the target, so continuing
                # is safer than crashing training mid-episode.
                if i == last_idx:
                    try:
                        motion = JointWaypointMotion([JointWaypoint(q)], HOME_VEL)
                        self._robot.move(motion)
                        self._wait_for_home(q)
                    except Exception as e2:
                        print(f"[RealFrankaEnv] Home step {i} retry also failed "
                              f"(continuing anyway): {e2}")
                        try:
                            self._robot.recover_from_errors()
                        except Exception:
                            pass

        # ---------- Optional human-in-the-loop pause ----------
        if self._pause_for_human:
            try:
                input(
                    "\n  >>> [pick-place] Place object in gripper and press "
                    "Enter to start the episode... "
                )
            except EOFError:
                # Non-interactive stdin — proceed without pause.
                print("[RealFrankaEnv] pause_for_human: stdin closed, continuing.")

        # ---------- Set final gripper state ----------
        if self._gripper_at_reset == "closed":
            self._gripper.grasp(
                0.0, self._gripper_speed, self._gripper_force,
                epsilon_inner=1.0, epsilon_outer=1.0,
            )
            self._gripper_is_open = False
        else:
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

    def _wait_for_home(self, target_q_list: list, timeout: float = 8.0):
        """Poll until joints reach target (used for homing stages)."""
        target = np.array(target_q_list, dtype=np.float64)
        t0 = time.time()
        while time.time() - t0 < timeout:
            current = np.array(
                self._robot.current_joint_state.position, dtype=np.float64,
            )
            if np.max(np.abs(current - target)) < 0.01:
                return
            time.sleep(0.02)
        err = np.max(np.abs(
            np.array(self._robot.current_joint_state.position, dtype=np.float64)
            - target
        ))
        print(f"  [WARN] Home settle timeout: max joint error = {err:.4f} rad")

    def step(self, action: np.ndarray) -> TimeStep:
        """
        Execute a normalised [-1, 1] action on the real robot.

        action : (8,) float32 — 7 joint deltas + 1 gripper, normalised.
        """
        raw = self._convert_action_to_raw(action)
        joint_delta = raw[:7]
        gripper_cmd = raw[7]

        # Safety clamp on delta (should be >= demo action range to avoid
        # throttling the agent below demo speed)
        joint_delta = np.clip(
            joint_delta, -self._joint_delta_clip, self._joint_delta_clip
        )

        # Compute target joint positions
        current_q = np.array(self._robot.current_joint_state.position, dtype=np.float64)
        target_q = current_q + joint_delta

        # ── Safety: clamp to joint limits ──
        target_q = self._clamp_joint_limits(target_q)

        # ── Safety: check workspace bounds BEFORE moving ──
        ee_pos = self._get_ee_position()
        if ee_pos is not None and self._is_outside_workspace(ee_pos):
            print(f"[SAFETY] Workspace violation! EE={ee_pos.round(3)} "
                  f"Bounds=[{WORKSPACE_MIN}, {WORKSPACE_MAX}]")
            print("[SAFETY] Ending episode and homing.")
            self._step_counter = self._episode_length  # force truncation
            obs = self._get_obs()
            return TimeStep(
                rgb_obs=obs["rgb_obs"],
                low_dim_obs=obs["low_dim_obs"],
                step_type=StepType.LAST,
                reward=0.0,
                discount=0.0,
                demo=0.0,
            )

        # Execute joint motion (single waypoint — franky handles current→target)
        try:
            motion = JointWaypointMotion(
                [JointWaypoint(target_q.tolist())],
                self._dynamics,
            )
            self._robot.move(motion)
        except Exception as e:
            print(f"[RealFrankaEnv] Motion error: {e}")
            self._robot.recover_from_errors()
            time.sleep(0.5)

        # Gripper (agent-controlled unless frozen)
        if not self._freeze_gripper:
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

        # ── Safety: check workspace bounds AFTER moving ──
        ee_pos_after = self._get_ee_position()
        workspace_violation = (ee_pos_after is not None and
                               self._is_outside_workspace(ee_pos_after))
        if workspace_violation:
            print(f"[SAFETY] Post-move workspace violation! EE={ee_pos_after.round(3)}")

        # Observe
        obs = self._get_obs()

        # Truncation (normal OR safety)
        truncated = (self._step_counter >= self._episode_length) or workspace_violation
        terminated = False

        if terminated or truncated:
            step_type = StepType.LAST
        else:
            step_type = StepType.MID
        discount = float(1 - terminated)

        # If gripper is frozen, auto-release at the end of the episode so
        # the operator can retrieve the object before the next reset.
        if self._freeze_gripper and step_type == StepType.LAST and not self._gripper_is_open:
            try:
                self._gripper.open(self._gripper_speed)
                self._gripper_is_open = True
            except Exception as e:
                print(f"[RealFrankaEnv] End-of-episode gripper open failed: {e}")

        return TimeStep(
            rgb_obs=obs["rgb_obs"],
            low_dim_obs=obs["low_dim_obs"],
            step_type=step_type,
            reward=0.0,
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

    def record_teleop_demo(
        self,
        hz: float = 10.0,
        max_steps: int = 500,
        reward_at_end: float = 1.0,
    ) -> list[ExtendedTimeStep]:
        """
        Record a demonstration via keyboard teleop.

        Uses the same key bindings as simtoreal.teleop but records
        observations + joint deltas. Your hands stay on the keyboard
        (away from cameras), so the visual observations match what the
        robot sees during autonomous execution.

        Keys (same as teleop.py):
          W/S = +X/-X   A/D = -Y/+Y   R/F = +Z/-Z  (TRANSLATE)
          T = toggle to ORIENT mode (W/S=pitch, A/D=yaw, Q/E=roll)
          SPACE = toggle gripper   1/2/3 = speed   ESC = end demo
        """
        import fcntl
        import os
        import sys
        import termios
        import tty
        from scipy.spatial.transform import Rotation as Rot

        STEP_PRESETS = {"1": 0.010, "2": 0.025, "3": 0.050}
        ROT_PRESETS  = {"1": 0.03,  "2": 0.06,  "3": 0.10}
        TELEOP_DYN = RelativeDynamicsFactor(0.5, 0.25, 0.25)

        # Save & restore robot dynamics after demo
        saved_dynamics = self._dynamics

        print(
            f"[Teleop Demo] Recording at {hz} Hz, max {max_steps} steps.\n"
            "  W/S/A/D/R/F = translate   T = toggle orient mode\n"
            "  SPACE = gripper   1/2/3 = speed   ESC = finish demo"
        )

        self._low_dim_obses.clear()
        for frames in self._frames.values():
            frames.clear()

        dt = 1.0 / hz
        step_m = STEP_PRESETS["2"]
        rot_rad = ROT_PRESETS["2"]
        translate_mode = True

        joint_positions_list: list[np.ndarray] = []
        gripper_open_list: list[float] = []
        obs_list: list[dict] = []

        # Set teleop dynamics
        self._robot.relative_dynamics_factor = TELEOP_DYN

        fd = sys.stdin.fileno()
        old_term = termios.tcgetattr(fd)
        old_flags = fcntl.fcntl(fd, fcntl.F_GETFL)

        try:
            tty.setcbreak(fd)
            fcntl.fcntl(fd, fcntl.F_SETFL, old_flags | os.O_NONBLOCK)

            for step_i in range(max_steps):
                t0 = time.time()

                # --- Read keys ---
                keys: set[str] = set()
                try:
                    while True:
                        ch = sys.stdin.read(1)
                        if not ch:
                            break
                        keys.add(ch.lower())
                except (IOError, BlockingIOError):
                    pass

                # ESC = end demo
                if "\x1b" in keys:
                    print(f"\n[Teleop Demo] Stopped at step {step_i}")
                    break

                # Toggle mode
                if "t" in keys:
                    translate_mode = not translate_mode
                    mode = "TRANSLATE" if translate_mode else "ORIENT"
                    print(f"  Mode: {mode}")

                # Speed
                for k in ("1", "2", "3"):
                    if k in keys:
                        step_m = STEP_PRESETS[k]
                        rot_rad = ROT_PRESETS[k]

                # Gripper
                if " " in keys:
                    if self._gripper_is_open:
                        self._gripper.grasp(
                            0.0, self._gripper_speed, self._gripper_force,
                            epsilon_inner=1.0, epsilon_outer=1.0,
                        )
                        self._gripper_is_open = False
                    else:
                        self._gripper.open(self._gripper_speed)
                        self._gripper_is_open = True

                # --- Move robot ---
                if translate_mode:
                    dx, dy, dz = 0.0, 0.0, 0.0
                    if "w" in keys: dx += step_m
                    if "s" in keys: dx -= step_m
                    if "a" in keys: dy -= step_m
                    if "d" in keys: dy += step_m
                    if "r" in keys: dz += step_m
                    if "f" in keys: dz -= step_m
                    if dx or dy or dz:
                        try:
                            self._robot.move(CartesianMotion(
                                Affine([dx, dy, dz]),
                                ReferenceType.Relative,
                                TELEOP_DYN,
                            ))
                        except Exception:
                            self._robot.recover_from_errors()
                else:
                    droll, dpitch, dyaw = 0.0, 0.0, 0.0
                    if "w" in keys: dpitch += rot_rad
                    if "s" in keys: dpitch -= rot_rad
                    if "a" in keys: dyaw += rot_rad
                    if "d" in keys: dyaw -= rot_rad
                    if "q" in keys: droll -= rot_rad
                    if "e" in keys: droll += rot_rad
                    if droll or dpitch or dyaw:
                        quat = Rot.from_euler(
                            "xyz", [droll, dpitch, dyaw]
                        ).as_quat()
                        try:
                            self._robot.move(CartesianMotion(
                                Affine([0, 0, 0], quat),
                                ReferenceType.Relative,
                                TELEOP_DYN,
                            ))
                        except Exception:
                            self._robot.recover_from_errors()

                # --- Record observation ---
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
            print(f"\n[Teleop Demo] Interrupted at step {len(obs_list)}")
        finally:
            # Restore terminal and dynamics
            fcntl.fcntl(fd, fcntl.F_SETFL, old_flags)
            termios.tcsetattr(fd, termios.TCSADRAIN, old_term)
            self._robot.relative_dynamics_factor = saved_dynamics

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
        print(f"[Teleop Demo] Recorded {len(timesteps)} steps")
        return timesteps

    # ------------------------------------------------------------------
    # Waypoint-based demonstration replay
    # ------------------------------------------------------------------

    def replay_waypoint_demo(
        self,
        waypoint_path: str,
        hz: float = 10.0,
        reward_at_end: float = 1.0,
    ) -> list[ExtendedTimeStep]:
        """
        Replay a saved trajectory file — fully sequential, no batching.

        For each waypoint:
          1. Move to the joint position
          2. Poll until robot actually reaches it (not just command sent)
          3. Apply gripper change if needed
          4. Record one observation

        Then reverse-retrace every waypoint one at a time.
        """
        import json

        REPLAY_DYN = RelativeDynamicsFactor(0.3, 0.1, 0.1)

        # Tolerance for "robot has arrived" check (radians)
        POSITION_TOL = 0.01  # ~0.6 degrees per joint
        SETTLE_TIMEOUT = 5.0  # max seconds to wait for convergence

        def _wait_for_position(target_q_list: list[float]):
            """Block until robot joints are within tolerance of target."""
            target = np.array(target_q_list, dtype=np.float64)
            t0 = time.time()
            while time.time() - t0 < SETTLE_TIMEOUT:
                current = np.array(
                    self._robot.current_joint_state.position,
                    dtype=np.float64,
                )
                err = np.max(np.abs(current - target))
                if err < POSITION_TOL:
                    return
                time.sleep(0.02)  # 50 Hz poll
            # Timed out — print warning but continue
            current = np.array(
                self._robot.current_joint_state.position, dtype=np.float64
            )
            err = np.max(np.abs(current - target))
            print(f"  [WARN] Settle timeout: max joint error = {err:.4f} rad")

        with open(waypoint_path, "r") as f:
            data = json.load(f)
        waypoints = data["waypoints"]
        rec_hz = data.get("hz", hz)
        print(f"[Waypoint Demo] Replaying {len(waypoints)} samples "
              f"(recorded at {rec_hz} Hz) from {waypoint_path}")

        # Clear frame stacks
        self._low_dim_obses.clear()
        for frames in self._frames.values():
            frames.clear()

        joint_positions_list: list[np.ndarray] = []
        gripper_open_list: list[float] = []
        obs_list: list[dict] = []

        # Home the robot first
        self.reset()

        for wp_idx, wp in enumerate(waypoints):
            target_q = wp["joints"]
            gripper_open = wp["gripper_open"]

            # 1. Issue move command
            move_ok = False
            try:
                motion = JointWaypointMotion(
                    [JointWaypoint(target_q)],
                    REPLAY_DYN,
                )
                self._robot.move(motion)
                move_ok = True
            except Exception as e:
                print(f"[Waypoint Demo] Motion error at sample {wp_idx}: {e}")
                self._robot.recover_from_errors()
                time.sleep(0.5)

            # 2. Wait until robot has PHYSICALLY arrived at the target
            #    Skip if move failed — robot didn't go anywhere
            if move_ok:
                _wait_for_position(target_q)

            # 3. NOW apply gripper — robot is confirmed at the position
            if gripper_open and not self._gripper_is_open:
                self._gripper.open(self._gripper_speed)
                self._gripper_is_open = True
                time.sleep(0.5)  # let gripper fully open
            elif not gripper_open and self._gripper_is_open:
                self._gripper.grasp(
                    0.0, self._gripper_speed, self._gripper_force,
                    epsilon_inner=1.0, epsilon_outer=1.0,
                )
                self._gripper_is_open = False
                time.sleep(0.5)  # let gripper fully close

            # 4. Record observation at this exact settled state
            obs = self._get_obs()
            q = np.array(
                self._robot.current_joint_state.position, dtype=np.float32
            )
            g = 1.0 if self._gripper_is_open else 0.0
            joint_positions_list.append(q)
            gripper_open_list.append(g)
            obs_list.append(obs)

            if (wp_idx + 1) % 50 == 0:
                print(f"  ... {wp_idx + 1}/{len(waypoints)} samples replayed")

        print(f"[Waypoint Demo] Forward pass complete: {len(obs_list)} obs")

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
        print(f"[Waypoint Demo] Recorded {len(timesteps)} steps")

        # --- Reverse retrace: sequential, one waypoint at a time ---
        # Open gripper first so we don't drag the object back
        if not self._gripper_is_open:
            self._gripper.open(self._gripper_speed)
            self._gripper_is_open = True
            time.sleep(0.5)

        rev = list(reversed(waypoints))
        print(f"[Waypoint Demo] Retracing {len(rev)} waypoints in reverse "
              "(sequential, one at a time)...")
        for ri, wp in enumerate(rev):
            move_ok = False
            try:
                motion = JointWaypointMotion(
                    [JointWaypoint(wp["joints"])],
                    REPLAY_DYN,
                )
                self._robot.move(motion)
                move_ok = True
            except Exception as e:
                print(f"[Waypoint Demo] Reverse error at step {ri}: {e}")
                self._robot.recover_from_errors()
                time.sleep(0.5)

            # Wait until robot actually reaches the reverse waypoint
            # before issuing the next one — prevents jumps
            if move_ok:
                _wait_for_position(wp["joints"])

            if (ri + 1) % 50 == 0:
                print(f"  ... {ri + 1}/{len(rev)} reverse steps")
        print("[Waypoint Demo] Retrace complete.")

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
        # Auto-set joint_delta_clip from demo range so the agent can
        # reproduce demo speeds.  Use the expanded range (with 20% margin)
        # to match _convert_action_to_raw.
        a_max = np.max(np.abs(np.concatenate([
            stats["max"][:7], stats["min"][:7]
        ])))
        expanded = a_max + np.fabs(a_max) * 0.2
        new_clip = max(float(expanded), self._joint_delta_clip)
        if new_clip > self._joint_delta_clip:
            print(f"[RealFrankaEnv] Auto-adjusting joint_delta_clip: "
                  f"{self._joint_delta_clip:.4f} → {new_clip:.4f} "
                  f"(from demo action range)")
            self._joint_delta_clip = new_clip

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

    # ── Safety helpers ─────────────────────────────────────────────────
    def _get_ee_position(self) -> np.ndarray | None:
        """Return the current end-effector (flange) position [x, y, z]."""
        try:
            pose = self._robot.current_cartesian_state.pose
            # franky Affine → 4×4 homogeneous matrix
            T = np.array(pose.matrix())
            return T[:3, 3].copy()
        except Exception as e:
            print(f"[SAFETY] Could not read EE pose: {e}")
            return None

    def get_ee_pose(self) -> tuple[np.ndarray, np.ndarray] | None:
        """Return (position [x,y,z], quaternion [x,y,z,w]) of end-effector."""
        try:
            ee = self._robot.current_cartesian_state.pose.end_effector_pose
            pos = np.array(ee.translation, dtype=np.float64)
            quat = np.array(ee.quaternion, dtype=np.float64)  # xyzw
            return pos, quat
        except Exception as e:
            print(f"[get_ee_pose] Could not read EE pose: {e}")
            return None

    @staticmethod
    def _is_outside_workspace(ee_pos: np.ndarray) -> bool:
        """True if *any* coordinate is outside the safe box."""
        return bool(
            np.any(ee_pos < WORKSPACE_MIN) or np.any(ee_pos > WORKSPACE_MAX)
        )

    @staticmethod
    def _clamp_joint_limits(target_q: np.ndarray) -> np.ndarray:
        """Clamp target joints to Panda limits (with small margin)."""
        margin = 0.02  # ~1 deg buffer from hard stops
        lo = PANDA_JOINT_MIN + margin
        hi = PANDA_JOINT_MAX - margin
        return np.clip(target_q, lo, hi)

    # ── Action conversions ───────────────────────────────────────────
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

    def close_gripper(self):
        """Close / grasp the gripper."""
        try:
            self._gripper.grasp(
                0.0, self._gripper_speed, self._gripper_force,
                epsilon_inner=1.0, epsilon_outer=1.0,
            )
            self._gripper_is_open = False
        except Exception as e:
            print(f"[RealFrankaEnv] Gripper close failed: {e}")

    def open_gripper(self):
        """Open the gripper."""
        try:
            self._gripper.open(self._gripper_speed)
            self._gripper_is_open = True
        except Exception as e:
            print(f"[RealFrankaEnv] Gripper open failed: {e}")

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
    velocity_factor: float = 0.4,
    acceleration_factor: float = 0.15,
    jerk_factor: float = 0.1,
    action_stats_path: str | None = None,
    gripper_speed: float = 0.1,
    gripper_force: float = 20.0,
    home_sequence: list | None = None,
    gripper_at_reset: str = "open",
    pause_for_human: bool = False,
    freeze_gripper: bool = False,
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
        home_sequence=home_sequence,
        gripper_at_reset=gripper_at_reset,
        pause_for_human=pause_for_human,
        freeze_gripper=freeze_gripper,
    )
    return ExtendedTimeStepWrapper(env)
