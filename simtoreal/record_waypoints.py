"""
record_waypoints.py — Record joint trajectories via freedrive.

Move the robot by hand in freedrive / gravity-comp mode.  Joint positions
and gripper state are recorded continuously at a fixed rate (default 10 Hz).
Press G to toggle the gripper, ENTER to finish a demo.

The saved .json files are later replayed autonomously (no hand in frame)
while the training script records camera observations + joint deltas.

Usage:
    python -m simtoreal.record_waypoints \
        --robot-ip 192.168.131.41 \
        --num-demos 10 \
        --hz 10 \
        --save-dir ../runs/shelf_4cam/waypoints
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import sys
import termios
import time
import tty
from pathlib import Path

import numpy as np
from franky import Gripper, Robot


def record_one(robot: Robot, gripper: Gripper, demo_idx: int, hz: float) -> dict | None:
    """Continuously record joint positions during freedrive for one demo."""
    print(f"\n{'='*60}")
    print(f"  Demo {demo_idx}")
    print(f"{'='*60}")
    print(f"  Move the robot by hand (freedrive mode).  Recording at {hz} Hz.")
    print("  Commands:")
    print("    G      — toggle gripper open / close")
    print("    SPACE  — finish this demo (gripper auto-opens after)")
    print("    Q      — abort this demo")
    input("  Press ENTER to start recording...")

    dt = 1.0 / hz
    waypoints: list[dict] = []
    gripper_open = True
    gripper.open(0.1)

    fd = sys.stdin.fileno()
    old_term = termios.tcgetattr(fd)
    old_flags = fcntl.fcntl(fd, fcntl.F_GETFL)

    aborted = False
    try:
        tty.setcbreak(fd)
        fcntl.fcntl(fd, fcntl.F_SETFL, old_flags | os.O_NONBLOCK)

        while True:
            t0 = time.time()

            # Record current state
            q = list(robot.current_joint_state.position)
            waypoints.append({
                "joints": [float(v) for v in q],
                "gripper_open": gripper_open,
            })

            # Check for key presses (non-blocking)
            keys: set[str] = set()
            try:
                while True:
                    ch = sys.stdin.read(1)
                    if not ch:
                        break
                    keys.add(ch.lower())
            except (IOError, BlockingIOError):
                pass

            if " " in keys:
                break

            if "q" in keys:
                print(f"\n  ✗ Demo aborted at {len(waypoints)} samples.")
                aborted = True
                break

            if "g" in keys:
                if gripper_open:
                    gripper.grasp(0.0, 0.1, 20.0, epsilon_inner=1.0, epsilon_outer=1.0)
                    gripper_open = False
                    print(f"\r  [{len(waypoints)} samples] ✊ Gripper CLOSED  ", end="", flush=True)
                else:
                    gripper.open(0.1)
                    gripper_open = True
                    print(f"\r  [{len(waypoints)} samples] 🖐  Gripper OPEN   ", end="", flush=True)
            else:
                print(f"\r  [{len(waypoints)} samples] Recording...      ", end="", flush=True)

            elapsed = time.time() - t0
            if elapsed < dt:
                time.sleep(dt - elapsed)

    except KeyboardInterrupt:
        print(f"\n  Stopped early at {len(waypoints)} samples.")
    finally:
        fcntl.fcntl(fd, fcntl.F_SETFL, old_flags)
        termios.tcsetattr(fd, termios.TCSADRAIN, old_term)

    # Always open gripper after recording ends (NOT part of the data).
    # This lets the user close the gripper as the final demo action
    # without needing to re-open it manually.
    if not gripper_open:
        print("  Opening gripper (post-collection, not recorded)...")
        gripper.open(0.1)

    if aborted:
        return None

    if len(waypoints) < 2:
        print("\n  Need at least 2 samples. Skipping.")
        return None

    duration = len(waypoints) / hz
    print(f"\n  ✓ Demo complete: {len(waypoints)} samples ({duration:.1f}s at {hz} Hz)")
    return {
        "hz": hz,
        "num_waypoints": len(waypoints),
        "waypoints": waypoints,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Record freedrive joint trajectories for later autonomous replay"
    )
    parser.add_argument("--robot-ip", type=str, default="192.168.131.41")
    parser.add_argument("--num-demos", type=int, default=10)
    parser.add_argument("--hz", type=float, default=10.0,
                        help="Recording rate in Hz (default: 10)")
    parser.add_argument("--save-dir", type=str, required=True,
                        help="Directory to save trajectory .json files")
    args = parser.parse_args()

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"Connecting to robot at {args.robot_ip}...")
    robot = Robot(args.robot_ip)
    gripper = Gripper(args.robot_ip)
    robot.set_collision_behavior(50, 50)
    robot.recover_from_errors()
    gripper.open(0.1)
    print("Connected. Put the robot in freedrive / gravity-comp mode.\n")

    saved = 0
    for i in range(args.num_demos):
        result = record_one(robot, gripper, i + 1, args.hz)
        if result is None:
            print("  Skipping this demo.")
            continue

        path = save_dir / f"waypoints_{saved:04d}.json"
        with open(path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"  Saved to {path}")
        saved += 1

    print(f"\nDone! {saved} trajectory files saved in {save_dir}/")
    print("Next: run training with --demo-mode waypoint --waypoint-dir " + str(save_dir))


if __name__ == "__main__":
    main()
