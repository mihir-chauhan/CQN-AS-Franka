"""
teleop.py — Continuous keyboard teleoperation of the Franka Panda.

Hold keys to move continuously. The robot moves at a fixed control rate
(default 10 Hz) with small Cartesian increments while a key is held.

Controls (robot perspective, facing the table):
────────────────────────────────────────────────
  Movement (hold to move continuously):
    W / S     — forward / backward   (+X / -X)
    A / D     — left / right         (-Y / +Y)
    R / F     — up / down            (+Z / -Z)

  Wrist rotation (hold):
    Q / E     — rotate wrist CCW / CW

  Gripper:
    SPACE     — toggle gripper open / close

  Speed:
    1         — slow   (2 mm/step  → ~2 cm/s at 10 Hz)
    2         — medium (5 mm/step  → ~5 cm/s at 10 Hz) [default]
    3         — fast   (10 mm/step → ~10 cm/s at 10 Hz)

  Other:
    H         — home the robot
    ESC / X   — quit

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

HALF_VEL = RelativeDynamicsFactor(0.1, 0.01, 0.01)

# Load home pose from image_45.npy
_HOME_NPY = Path(__file__).resolve().parent / "image_45.npy"
HOME_Q = np.load(str(_HOME_NPY)).tolist()

# Step size per control tick (metres). At 10 Hz, speed ≈ step × 10.
SPEED_PRESETS = {
    "1": 0.002,   # 2 mm/tick  → ~2 cm/s
    "2": 0.005,   # 5 mm/tick  → ~5 cm/s  (default)
    "3": 0.010,   # 10 mm/tick → ~10 cm/s
}

ROTATION_STEP = 0.01  # radians/tick (~0.6°/tick → ~6°/s at 10 Hz)

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
    robot.relative_dynamics_factor = HALF_VEL
    robot.set_collision_behavior(50, 50)
    robot.recover_from_errors()

    # Home
    print("Homing robot...")
    motion = JointWaypointMotion([JointWaypoint(HOME_Q)], HALF_VEL)
    robot.move(motion)
    gripper.open(GRIPPER_SPEED)
    gripper_open = True
    time.sleep(0.5)

    step_m = 0.005  # default 5 mm/tick
    dt = 1.0 / args.hz

    print()
    print("=" * 60)
    print("  FRANKA CONTINUOUS TELEOP")
    print("=" * 60)
    print()
    print("  HOLD keys to move continuously:")
    print()
    print("  W/S  = forward/back    A/D  = left/right")
    print("  R/F  = up/down         Q/E  = rotate wrist")
    print("  SPACE = toggle gripper")
    print("  1/2/3 = speed (2mm / 5mm / 10mm per tick)")
    print(f"  Control rate: {args.hz:.0f} Hz")
    print("  H = home               ESC or X = quit")
    print()
    print(f"  Speed: {step_m * 1000:.0f} mm/tick ({step_m * args.hz * 100:.0f} cm/s)")
    print(f"  Gripper: OPEN")
    print()
    print("  Ready! Hold a movement key...")
    print()

    last_print_time = 0.0
    gripper_toggled = False  # debounce

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
                        robot.move(JointWaypointMotion([JointWaypoint(HOME_Q)], HALF_VEL))
                    except Exception as ex:
                        print(f"  Homing failed: {ex}")
                        robot.recover_from_errors()
                        time.sleep(1.0)
                        robot.move(JointWaypointMotion([JointWaypoint(HOME_Q)], HALF_VEL))
                    print("  Home reached.")
                    continue

                # ── Speed presets ──
                for k, v in SPEED_PRESETS.items():
                    if k in keys:
                        step_m = v
                        print(f"  Speed: {step_m * 1000:.0f} mm/tick "
                              f"({step_m * args.hz * 100:.0f} cm/s)")

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
                dx, dy, dz, dq7 = 0.0, 0.0, 0.0, 0.0

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
                if "q" in keys:
                    dq7 -= ROTATION_STEP
                if "e" in keys:
                    dq7 += ROTATION_STEP

                # ── Execute Cartesian motion ──
                if dx != 0.0 or dy != 0.0 or dz != 0.0:
                    try:
                        cart_motion = CartesianMotion(
                            Affine([dx, dy, dz]),
                            ReferenceType.Relative,
                            HALF_VEL,
                        )
                        robot.move(cart_motion)

                        # Print position at ~2 Hz to avoid spam
                        now = time.time()
                        if now - last_print_time > 0.5:
                            pos = robot.current_cartesian_state.pose.end_effector_pose.translation
                            print(f"  pos: x={pos[0]:.3f}  y={pos[1]:.3f}  z={pos[2]:.3f}")
                            last_print_time = now
                    except Exception as ex:
                        robot.recover_from_errors()
                        time.sleep(0.1)

                # ── Execute wrist rotation ──
                if dq7 != 0.0:
                    try:
                        current_q = list(robot.current_joint_state.position)
                        target_q = current_q.copy()
                        target_q[6] += dq7
                        motion = JointWaypointMotion(
                            [JointWaypoint(target_q)], HALF_VEL
                        )
                        robot.move(motion)
                    except Exception as ex:
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
