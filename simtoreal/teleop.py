"""
teleop.py — Continuous keyboard teleoperation of the Franka Panda.

Hold keys to move continuously. The robot moves at a fixed control rate
(default 10 Hz) with small Cartesian increments while a key is held.

Two modes:
  T  — toggle between TRANSLATE mode (default) and ORIENT mode.

TRANSLATE mode (move the end-effector position):
  W / S     — forward / backward   (+X / -X)
  A / D     — left / right         (-Y / +Y)
  R / F     — up / down            (+Z / -Z)

ORIENT mode (rotate the end-effector in place):
  W / S     — pitch down / up     (rotation about EE Y-axis)
  A / D     — yaw left / right    (rotation about EE Z-axis)
  Q / E     — roll CCW / CW       (rotation about EE X-axis)

Always available:
  SPACE     — toggle gripper open / close
  1 / 2 / 3 — speed preset (slow / medium / fast)
  H         — home the robot
  ESC / X   — quit
  T         — toggle TRANSLATE ↔ ORIENT mode

Usage:
    python -m simtoreal.teleop
    python -m simtoreal.teleop --robot-ip 192.168.131.41 --hz 15
"""

from __future__ import annotations

import argparse
import fcntl
import os
import select
import sys
import termios
import time
import tty
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R
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

# ── Constants ────────────────────────────────────────────────────────────────

TELEOP_VEL = RelativeDynamicsFactor(0.5, 0.25, 0.25)  # faster for teleop only
HOME_VEL = RelativeDynamicsFactor(0.2, 0.1, 0.1)      # gentler for homing

# Load home pose from image_45.npy
_HOME_NPY = Path(__file__).resolve().parent / "image_45.npy"
HOME_Q = np.load(str(_HOME_NPY)).tolist()

# Step size per control tick (metres). At 10 Hz, speed ≈ step × 10.
SPEED_PRESETS = {
    "1": 0.010,   # 10 mm/tick  → ~10 cm/s
    "2": 0.025,   # 25 mm/tick  → ~25 cm/s  (default)
    "3": 0.050,   # 50 mm/tick  → ~50 cm/s
}

# Orientation step per control tick (radians).
# At 10 Hz, rotation speed ≈ step × 10.
ORIENT_PRESETS = {
    "1": 0.03,    #  ~1.7°/tick → ~17°/s
    "2": 0.06,    #  ~3.4°/tick → ~34°/s  (default)
    "3": 0.10,    #  ~5.7°/tick → ~57°/s
}

GRIPPER_SPEED = 0.1   # m/s
GRIPPER_FORCE = 20.0  # N


# ── Non-blocking keyboard ───────────────────────────────────────────────────

class RawKeyboard:
    """
    Context manager that puts the terminal in raw mode and provides
    non-blocking key reads.  Drains all buffered keys each call so
    we get the *latest* held key, not a stale queue.
    """

    def __enter__(self):
        self._fd = sys.stdin.fileno()
        self._old = termios.tcgetattr(self._fd)
        tty.setcbreak(self._fd)  # cbreak = char-at-a-time, signals still work
        # Make stdin non-blocking
        self._old_flags = fcntl.fcntl(self._fd, fcntl.F_GETFL)
        fcntl.fcntl(self._fd, fcntl.F_SETFL, self._old_flags | os.O_NONBLOCK)
        return self

    def __exit__(self, *args):
        fcntl.fcntl(self._fd, fcntl.F_SETFL, self._old_flags)
        termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old)

    def get_keys(self) -> set[str]:
        """Return all keys currently buffered (approximates 'held keys')."""
        keys: set[str] = set()
        try:
            while True:
                ch = sys.stdin.read(1)
                if not ch:
                    break
                keys.add(ch.lower())
        except (IOError, BlockingIOError):
            pass
        return keys


# ── Main ─────────────────────────────────────────────────────────────────────

def _euler_to_quat(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Small-angle RPY (EE-local frame) → quaternion [x, y, z, w]."""
    return R.from_euler("xyz", [roll, pitch, yaw]).as_quat()  # scipy: xyzw


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Continuous keyboard teleop for Franka")
    parser.add_argument("--robot-ip", type=str, default="192.168.131.41")
    parser.add_argument("--hz", type=float, default=10.0,
                        help="Control loop rate (Hz)")
    args = parser.parse_args()

    # Connect
    print(f"Connecting to robot at {args.robot_ip}...")
    robot = Robot(args.robot_ip)
    gripper = Gripper(args.robot_ip)
    robot.relative_dynamics_factor = TELEOP_VEL
    robot.set_collision_behavior(50, 50)
    robot.recover_from_errors()

    # Home
    print("Homing robot...")
    motion = JointWaypointMotion([JointWaypoint(HOME_Q)], HOME_VEL)
    robot.move(motion)
    gripper.open(GRIPPER_SPEED)
    gripper_open = True
    time.sleep(0.5)

    step_m = SPEED_PRESETS["2"]   # default translation step
    rot_rad = ORIENT_PRESETS["2"]  # default orientation step
    dt = 1.0 / args.hz
    translate_mode = True  # True = TRANSLATE, False = ORIENT

    def _print_banner():
        mode_str = "🔹 TRANSLATE" if translate_mode else "🔸 ORIENT"
        print()
        print("=" * 60)
        print("  FRANKA CONTINUOUS TELEOP")
        print("=" * 60)
        print()
        print(f"  Current mode: {mode_str}")
        print()
        if translate_mode:
            print("  W/S  = forward / back     (+X / -X)")
            print("  A/D  = left / right        (-Y / +Y)")
            print("  R/F  = up / down           (+Z / -Z)")
        else:
            print("  W/S  = pitch down / up     (about local Y)")
            print("  A/D  = yaw left / right    (about local Z)")
            print("  Q/E  = roll CCW / CW       (about local X)")
        print()
        print("  T     = toggle TRANSLATE ↔ ORIENT mode")
        print("  SPACE = toggle gripper")
        print("  1/2/3 = speed preset")
        print("  H     = home          ESC or X = quit")
        print(f"  Rate : {args.hz:.0f} Hz")
        print(f"  Trans: {step_m * 1000:.0f} mm/tick "
              f"({step_m * args.hz * 100:.0f} cm/s)")
        print(f"  Rot  : {rot_rad * 180 / np.pi:.1f}°/tick "
              f"({rot_rad * args.hz * 180 / np.pi:.0f}°/s)")
        print(f"  Gripper: {'OPEN' if gripper_open else 'CLOSED'}")
        print()
        print("  Ready! Hold movement keys...")
        print()

    _print_banner()

    last_print_time = 0.0
    gripper_toggled = False  # debounce
    mode_toggled = False     # debounce for T

    with RawKeyboard() as kb:
        try:
            while True:
                t0 = time.time()
                keys = kb.get_keys()

                # ── Quit ──
                if "\x1b" in keys or "x" in keys:
                    print("\n  Quitting...")
                    break

                # ── Home ──
                if "h" in keys:
                    print("  🏠 Homing...")
                    robot.recover_from_errors()
                    time.sleep(0.5)
                    try:
                        robot.move(JointWaypointMotion([JointWaypoint(HOME_Q)], HOME_VEL))
                    except Exception as ex:
                        print(f"  Homing failed: {ex}")
                        robot.recover_from_errors()
                        time.sleep(1.0)
                        robot.move(JointWaypointMotion([JointWaypoint(HOME_Q)], HOME_VEL))
                    print("  Home reached.")
                    continue

                # ── Mode toggle (with debounce) ──
                if "t" in keys:
                    if not mode_toggled:
                        mode_toggled = True
                        translate_mode = not translate_mode
                        _print_banner()
                else:
                    mode_toggled = False

                # ── Speed presets (affect both translation & rotation) ──
                for k in ("1", "2", "3"):
                    if k in keys:
                        step_m = SPEED_PRESETS[k]
                        rot_rad = ORIENT_PRESETS[k]
                        print(f"  Speed preset {k}: "
                              f"trans={step_m * 1000:.0f} mm/tick  "
                              f"rot={rot_rad * 180 / np.pi:.1f}°/tick")

                # ── Gripper toggle (with debounce) ──
                if " " in keys:
                    if not gripper_toggled:
                        gripper_toggled = True
                        if gripper_open:
                            print("  ✊ Gripper CLOSING...")
                            gripper.grasp(
                                0.0, GRIPPER_SPEED, GRIPPER_FORCE,
                                epsilon_inner=1.0, epsilon_outer=1.0,
                            )
                            gripper_open = False
                        else:
                            print("  🖐  Gripper OPENING...")
                            gripper.open(GRIPPER_SPEED)
                            gripper_open = True
                else:
                    gripper_toggled = False

                # ── Accumulate movement from held keys ──
                if translate_mode:
                    # ── TRANSLATE mode: W/S/A/D/R/F move position ──
                    dx, dy, dz = 0.0, 0.0, 0.0
                    if "w" in keys:
                        dx += step_m
                    if "s" in keys:
                        dx -= step_m
                    if "a" in keys:
                        dy -= step_m
                    if "d" in keys:
                        dy += step_m
                    if "r" in keys:
                        dz += step_m
                    if "f" in keys:
                        dz -= step_m

                    if dx != 0.0 or dy != 0.0 or dz != 0.0:
                        try:
                            cart_motion = CartesianMotion(
                                Affine([dx, dy, dz]),
                                ReferenceType.Relative,
                                TELEOP_VEL,
                            )
                            robot.move(cart_motion)

                            now = time.time()
                            if now - last_print_time > 0.5:
                                ee = robot.current_cartesian_state.pose.end_effector_pose
                                pos = ee.translation
                                quat = ee.quaternion  # xyzw
                                rpy = R.from_quat(quat).as_euler("xyz", degrees=True)
                                print(f"  pos: x={pos[0]:.3f} y={pos[1]:.3f} z={pos[2]:.3f}"
                                      f"  rpy: R={rpy[0]:.1f} P={rpy[1]:.1f} Y={rpy[2]:.1f}")
                                last_print_time = now
                        except Exception:
                            robot.recover_from_errors()
                            time.sleep(0.1)

                else:
                    # ── ORIENT mode: W/S=pitch, A/D=yaw, Q/E=roll ──
                    droll, dpitch, dyaw = 0.0, 0.0, 0.0
                    if "w" in keys:
                        dpitch += rot_rad   # pitch down (nose down)
                    if "s" in keys:
                        dpitch -= rot_rad   # pitch up
                    if "a" in keys:
                        dyaw += rot_rad     # yaw left
                    if "d" in keys:
                        dyaw -= rot_rad     # yaw right
                    if "q" in keys:
                        droll -= rot_rad    # roll CCW
                    if "e" in keys:
                        droll += rot_rad    # roll CW

                    if droll != 0.0 or dpitch != 0.0 or dyaw != 0.0:
                        try:
                            quat = _euler_to_quat(droll, dpitch, dyaw)
                            cart_motion = CartesianMotion(
                                Affine([0.0, 0.0, 0.0], quat),
                                ReferenceType.Relative,
                                TELEOP_VEL,
                            )
                            robot.move(cart_motion)

                            now = time.time()
                            if now - last_print_time > 0.5:
                                ee = robot.current_cartesian_state.pose.end_effector_pose
                                pos = ee.translation
                                eq = ee.quaternion
                                rpy = R.from_quat(eq).as_euler("xyz", degrees=True)
                                print(f"  pos: x={pos[0]:.3f} y={pos[1]:.3f} z={pos[2]:.3f}"
                                      f"  rpy: R={rpy[0]:.1f} P={rpy[1]:.1f} Y={rpy[2]:.1f}")
                                last_print_time = now
                        except Exception:
                            robot.recover_from_errors()
                            time.sleep(0.1)

                # ── Control rate ──
                elapsed = time.time() - t0
                if elapsed < dt:
                    time.sleep(dt - elapsed)

        except KeyboardInterrupt:
            print("\n  Interrupted.")
        finally:
            print("  Done.")


if __name__ == "__main__":
    main()
