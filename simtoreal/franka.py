import time
from franky import *
import numpy as np
from scipy.spatial.transform import Rotation as R

# HOME_Q = [0., -np.pi/4, 0., -3*np.pi/4, 0., np.pi/2, np.pi/4] # real home
HOME_Q = [
    -np.pi / 4,
    -np.pi / 4,
    0.0,
    -3 * np.pi / 4,
    0.0,
    np.pi / 2,
    np.pi / 4,
] # home turned towards table
IMAGE_Q = np.load("positions/image_45.npy")
CAM_OFFT = np.array([0.065, 0.035, 0.044])
SPEED = 0.1
FORCE = 20

HALF_VEL = RelativeDynamicsFactor(0.1, 0.01, 0.01)


def home_robot(robot, gripper, open_gripper: bool = True):
    # robot.move(JointMotion(IMAGE_Q))
    if open_gripper:
        gripper.open(SPEED)
    robot.move(
        JointWaypointMotion(
            [JointWaypoint(robot.current_joint_state.position), JointWaypoint(IMAGE_Q)],
            HALF_VEL,
        )
    )


def move_down_to_grasp(
    robot, down_m: float, stop_force_N: int = 10, verbose: bool = False
):
    m_down_Xcm = CartesianMotion(
    Affine([0.0, 0.0, down_m]), ReferenceType.Relative, HALF_VEL
    )
    m_up_1cm = CartesianMotion(
    Affine([0.0, 0.0, -0.01]), ReferenceType.Relative, HALF_VEL
    )
    # m_stop = CartesianStopMotion(HALF_VEL)
    try:
        robot.set_collision_behavior(stop_force_N, stop_force_N)
        print('going down 30 cm')
        robot.move(m_down_Xcm)
        print("moved down successfully without collision?")
    except ControlException as e:
        if verbose:
            print("reseting after table collision...")
        set_default_collision_behavior(robot)
        robot.recover_from_errors()
        robot.move(m_up_1cm)


def downwards_ee_orn(angle: float):
    wrapped_angle = (angle + np.pi / 2) % np.pi - np.pi / 2
    return R.from_euler("xyz", [np.pi, 0, wrapped_angle - np.pi / 4])


def abs_to_rel_orn(robot, abs_orn: R) -> R:
    ee_pose_cur = robot.current_cartesian_state.pose.end_effector_pose
    ee_rot_cur = R.from_quat(ee_pose_cur.quaternion)
    return ee_rot_cur.inv() * abs_orn
    

def set_default_collision_behavior(robot):
    robot.set_collision_behavior(50, 50)


def main(robot, gripper):
    # TESTED EXAMPLE:

    # move to some position/orn:
    offset_m = [0.1, 0.05, 0.3] # 10cm forward, 5cm right, same height
    ee_quat_rel = abs_to_rel_orn(
        robot, downwards_ee_orn(np.pi / 4)
    ).as_quat() # 45 degrees, anticlockwise, starting from +x axis, like in math
    m_approx = CartesianWaypointMotion(
        [
            CartesianWaypoint(
                Affine(CAM_OFFT), ReferenceType.Relative, HALF_VEL
            ), # move to camera offset
            CartesianWaypoint(
                Affine(offset_m, ee_quat_rel), ReferenceType.Relative, HALF_VEL
            ),
        ]
    )
    robot.move(m_approx)

    # move down to table (max of 50 cm) until resistance more than 10N
    move_down_to_grasp(robot, down_m=0.3, stop_force_N=10, verbose=True)

    # grasp
    gripper.grasp(
        0.0, SPEED, FORCE, epsilon_inner=1.0, epsilon_outer=1.0
    ) # 0.005 is default for both

    # move to home
    home_robot(robot, gripper, open_gripper=False)
    # move_down_to_grasp(robot, down_m=0.3, stop_force_N=10)


if __name__ == "__main__":
    robot = Robot("192.168.131.41")
    gripper = Gripper("192.168.131.41")

    robot.relative_dynamics_factor = RelativeDynamicsFactor(
        velocity=0.1, acceleration=0.01, jerk=0.01
    )
    set_default_collision_behavior(robot)
    robot.recover_from_errors()
    home_robot(robot, gripper)
    # main(robot, gripper)

# from home, when facing table (robot perspective)
# +x = forward
# +y = right
# +z = up
# last joint + moves clockwise
