"""
Save the current end-effector pose (position + quaternion) to a JSON file.

Usage:
    Move the robot to the desired goal pose, then run:
        python -m simtoreal.save_goal_pose --robot-ip 192.168.131.41 -o goal_pose.json
"""

import argparse
import json
import numpy as np
from franky import Robot


def main():
    p = argparse.ArgumentParser(description="Save current EE pose to JSON")
    p.add_argument("--robot-ip", default="192.168.131.41")
    p.add_argument("-o", "--output", default="goal_pose.json",
                   help="Output JSON file path")
    args = p.parse_args()

    robot = Robot(args.robot_ip)
    robot.recover_from_errors()

    ee = robot.current_cartesian_state.pose.end_effector_pose
    pos = list(np.array(ee.translation, dtype=float))
    quat = list(np.array(ee.quaternion, dtype=float))  # [x, y, z, w]

    data = {
        "position": pos,
        "quaternion_xyzw": quat,
    }

    with open(args.output, "w") as f:
        json.dump(data, f, indent=2)

    print(f"Saved goal pose to {args.output}")
    print(f"  Position : [{pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}]")
    print(f"  Quaternion: [{quat[0]:.4f}, {quat[1]:.4f}, {quat[2]:.4f}, {quat[3]:.4f}]")


if __name__ == "__main__":
    main()
