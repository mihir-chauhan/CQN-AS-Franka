"""
teleop.py — Keyboard teleoperation of the Franka Panda in Cartesian space.

Controls (from the robot's perspective, facing the table):
──────────────────────────────────────────────────────────
  Movement:
    W / S     — forward / backward   (+X / -X)
    A / D     — left / right          (-Y / +Y)
    R / F     — up / down             (+Z / -Z)

  Wrist rotation (last joint):
    Q / E     — rotate wrist CCW / CW

  Gripper:
    SPACE     — toggle gripper open / close

  Speed:
    1         — slow   (1 cm steps)
    2         — medium (3 cm steps)  [default]
    3         — fast   (5 cm steps)

  Other:
    H         — home the robot
    ESC / X   — quit

Usage:
    python -m simtoreal.teleop
    python -m simtoreal.teleop --robot-ip 192.168.131.41
"""

from __future__ import annotations

import argparse
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

SPEED_PRESETS = {
    "1": 0.01,   # 1 cm
    "2": 0.03,   # 3 cm  (default)
    "3": 0.05,   # 5 cm
}

ROTATION_STEP = 0.05  # radians (~3°) per key press

GRIPPER_SPEED = 0.1   # m/s
GRIPPER_FORCE = 20.0  # N


# ── Keyboard helpers ─────────────────────────────────────────────────────────

def get_key() -> str:
    """Read a single keypress from stdin (raw mode, no echo)."""
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return ch


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Keyboard teleop for Franka Panda")
    parser.add_argument("--robot-ip", type=str, default="192.168.131.41")
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

    step_m = 0.03  # default 3 cm steps

    print()
    print("=" * 60)
    print("  FRANKA KEYBOARD TELEOP")
    print("=" * 60)
    print()
    print("  W/S  = forward/back    A/D  = left/right")
    print("  R/F  = up/down         Q/E  = rotate wrist")
    print("  SPACE = toggle gripper")
    print("  1/2/3 = step size (1cm / 3cm / 5cm)")
    print("  H = home               ESC or X = quit")
    print()
    print(f"  Step size: {step_m * 100:.0f} cm")
    print(f"  Gripper: {'OPEN' if gripper_open else 'CLOSED'}")
    print()
    print("  Ready! Press a key...")
    print()

    try:
        while True:
            key = get_key().lower()

            dx, dy, dz = 0.0, 0.0, 0.0
            dq7 = 0.0  # wrist rotation delta (joint 7)
            action = None

            # ── Movement ──
            if key == "w":
                dx = step_m
                action = "forward"
            elif key == "s":
                dx = -step_m
                action = "backward"
            elif key == "a":
                dy = -step_m
                action = "left"
            elif key == "d":
                dy = step_m
                action = "right"
            elif key == "r":
                dz = step_m
                action = "up"
            elif key == "f":
                dz = -step_m
                action = "down"

            # ── Wrist rotation ──
            elif key == "q":
                dq7 = -ROTATION_STEP
                action = "rotate CCW"
            elif key == "e":
                dq7 = ROTATION_STEP
                action = "rotate CW"

            # ── Gripper toggle ──
            elif key == " ":
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
                continue

            # ── Speed presets ──
            elif key in SPEED_PRESETS:
                step_m = SPEED_PRESETS[key]
                print(f"  Step size: {step_m * 100:.0f} cm")
                continue

            # ── Home ──
            elif key == "h":
                print("  🏠 Homing...")
                robot.recover_from_errors()
                time.sleep(0.5)
                motion = JointWaypointMotion([JointWaypoint(HOME_Q)], HALF_VEL)
                try:
                    robot.move(motion)
                except Exception as ex:
                    print(f"  Homing failed: {ex}")
                    robot.recover_from_errors()
                    time.sleep(1.0)
                    robot.move(JointWaypointMotion([JointWaypoint(HOME_Q)], HALF_VEL))
                print("  Home reached.")
                continue

            # ── Quit ──
            elif key == "\x1b" or key == "x":  # ESC or x
                print("\n  Quitting...")
                break

            else:
                # Unknown key — ignore
                continue

            # ── Execute Cartesian motion ──
            if dx != 0.0 or dy != 0.0 or dz != 0.0:
                try:
                    cart_motion = CartesianMotion(
                        Affine([dx, dy, dz]),
                        ReferenceType.Relative,
                        HALF_VEL,
                    )
                    robot.move(cart_motion)
                    pos = robot.current_cartesian_state.pose.end_effector_pose.translation
                    print(f"  → {action:10s}  |  pos: x={pos[0]:.3f}  y={pos[1]:.3f}  z={pos[2]:.3f}")
                except Exception as ex:
                    print(f"  ⚠ Motion error ({action}): {ex}")
                    robot.recover_from_errors()
                    time.sleep(0.3)

            # ── Execute wrist rotation (joint-space for last joint only) ──
            elif dq7 != 0.0:
                try:
                    current_q = list(robot.current_joint_state.position)
                    target_q = current_q.copy()
                    target_q[6] += dq7
                    motion = JointWaypointMotion(
                        [JointWaypoint(target_q)], HALF_VEL
                    )
                    robot.move(motion)
                    print(f"  → {action:10s}  |  joint7: {target_q[6]:.3f} rad")
                except Exception as ex:
                    print(f"  ⚠ Rotation error: {ex}")
                    robot.recover_from_errors()
                    time.sleep(0.3)

    except KeyboardInterrupt:
        print("\n  Interrupted.")
    finally:
        print("  Done.")


if __name__ == "__main__":
    main()
