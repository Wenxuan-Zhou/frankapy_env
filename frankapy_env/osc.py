import json
import math
import numpy as np
import os
import rospy
from autolab_core import RigidTransform
from franka_interface_msgs.msg import SensorDataGroup
from frankapy import FrankaConstants as FC
from frankapy import SensorDataMessageType
from frankapy.proto import PosePositionSensorMessage, ShouldTerminateSensorMessage, CartesianImpedanceSensorMessage
from frankapy.proto_utils import sensor_proto2ros_msg, make_sensor_group_msg


class OSC(object):
    """
    6D OSC Controller
    """

    def __init__(self, robot,
                 controller_type="OSC",
                 control_freq=None,
                 horizon=None,
                 use_frankapy_impedance=False,
                 control_axis=None,
                 max_translation=0.035,
                 max_rotation=0.25,
                 impedances=(300., 300., 300., 30., 30., 30.),
                 position_limits=((0.4, -0.2, 0), (0.8, 0.2, 0.25)),
                 orientation_limits=60.,
                 ):

        self.robot = robot

        # Setup controller
        self.controller_type = controller_type.upper()
        self.control_axis = np.ones(6, dtype=np.float64) if control_axis is None else control_axis
        self.control_dim = np.int(np.sum(self.control_axis))
        self.max_translation = max_translation
        self.max_rotation = max_rotation
        self.impedances = np.array(impedances, dtype=np.float64)
        self.position_limits = np.array(position_limits, dtype=np.float64)
        self.orientation_limits = np.array(orientation_limits, dtype=np.float64)
        self.use_frankapy_impedance = use_frankapy_impedance  # Use the implementation of OSC from frankapy if true (otherwise use the implementation of OSC from libfranka)

        # Prepare for dynamic trajectories
        if control_freq is None:
            print('Control frequency not available. Not using dynamic commands.')
        else:
            print('Control frequency:', control_freq)
        self.control_freq = control_freq
        self.dt = 1. / control_freq if control_freq is not None else None
        self.episode_init_time = None
        self.ros_pub = None
        self.ros_rate = None
        self.message_count = None
        self._publisher_running = False
        self.horizon = horizon

        self.current_desired_pose = None
        self.init_pose = None
        self.initial_ee_ori_mat = None
        self.initial_ee_pos = None

    """
    Operational Space Controller related
    """

    def reset(self):
        self.current_desired_pose = self.robot.get_pose().copy()
        self.init_pose = self.robot.get_pose().copy()
        self.initial_ee_ori_mat = np.round(self.init_pose.rotation)
        self.initial_ee_pos = self.init_pose.translation

    def run_controller(self, action):
        assert len(action) == self.control_dim, "Required action dimension: {}. Commanded action dimension: {}.".format(
            self.control_dim, len(action))

        # Align actions to corresponding the enabled axis.
        # For example, 3D action space in XZPlane will be assigned to x-translation, z-translation and y-rotation.
        delta = np.zeros_like(self.control_axis)
        dim_count = 0
        for i in range(len(self.control_axis)):
            if self.control_axis[i] == 1:
                delta[i] = action[dim_count]
                dim_count += 1
        assert dim_count == len(action)

        pose = self._get_desired_pose(delta)
        # If the desired pose is reaching orientation limit or joint limit,
        # it will use the previous goal
        if self.orientation_feasible(pose.rotation) and self.joint_limit_feasible(pose):
            self.current_desired_pose = pose.copy()

        if self._publisher_running:
            self._step_robot_dynamic(self.current_desired_pose)
        else:
            self._step_robot(self.current_desired_pose)

        if self.robot.is_skill_done():
            rospy.loginfo('OSC: Not using dynamics skill or the dynamic skill is stopped.')

    def _get_desired_pose(self, action):
        base_pose = self.robot.get_pose().copy()
        pos = base_pose.translation
        orn = base_pose.rotation

        # Apply translation
        pos += action[:3] * self.max_translation
        # Use initial ee_pos if the axis is not enabled
        pos = np.where(self.control_axis[:3] == 1, pos, self.initial_ee_pos)

        # Limit desired pose within the bounding box
        if np.any(pos > self.position_limits[1]) or np.any(pos < self.position_limits[0]):
            rospy.loginfo('OSC: Reaching translation limit.')
            pos = np.minimum(self.position_limits[1], pos)
            pos = np.maximum(self.position_limits[0], pos)

        # BE CAREFUL ABOUT THE CENTER OF ROTATION!
        # Current frame of rotation: center - fingertip. axis - global
        if np.all(self.control_axis[3:] == np.array([0, 1, 0])):
            # Rotation along Y axis only
            z_axis = orn[:, 2]  # global frame
            initial_z_axis = self.initial_ee_ori_mat[:, 2]  # global frame
            # TODO: remove angle_diff
            rotation_angle = angle_diff(z_axis, initial_z_axis) / 180 * np.pi
            rotation_angle *= np.sign(np.cross(initial_z_axis, z_axis)[1])
            # rotation axis given by positive y-direction in the global frame
            rotation_angle += action[4] * self.max_rotation

            rotation = RigidTransform.rotation_from_axis_angle(rotation_angle * np.array([0, 1, 0]))
            orn = rotation.dot(self.initial_ee_ori_mat)
        elif np.all(self.control_axis[3:] == np.array([1, 1, 1])):
            # divide by sqrt(3) because the norm of the max action is larger than dim=1.
            rotation = RigidTransform.rotation_from_axis_angle(action[3:] * self.max_rotation / np.sqrt(3))
            orn = rotation.dot(orn)
        elif np.all(self.control_axis[3:] == np.array([0, 0, 0])):
            orn = self.initial_ee_ori_mat
        else:
            raise NotImplementedError

        new_pose = RigidTransform(
            translation=pos,
            rotation=orn,
            from_frame='franka_tool', to_frame='world'
        )
        return new_pose

    def orientation_feasible(self, orn):
        quat = RigidTransform(rotation=orn.dot(self.initial_ee_ori_mat)).quaternion
        angle = np.arccos(min(abs(quat[0]), 1)) / np.pi * 180 * 2
        assert self.orientation_limits.shape == (), 'Only taking a scalar limit for now.'
        if angle > self.orientation_limits:
            if VERBOSE:
                print('Desired_pose reaching orientation limit')
            return False

        return True

    def joint_limit_feasible(self, desired_pose):
        # joint limit from frankapy
        JOINT_LIMITS_MIN = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973]) + 0.05
        JOINT_LIMITS_MAX = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973]) - 0.05

        current_pose = self.robot.get_pose()
        delta_pos = desired_pose.translation - current_pose.translation
        delta_ori = orientation_error(desired_pose.rotation, current_pose.rotation)

        jacobian = self.robot.get_jacobian(self.robot.get_joints())
        J_inv = np.linalg.pinv(jacobian)
        new_joint_pos = self.robot.get_joints() + np.dot(J_inv, np.concatenate([delta_pos, delta_ori]))

        if np.any(new_joint_pos < JOINT_LIMITS_MIN) or np.any(new_joint_pos > JOINT_LIMITS_MAX):
            rospy.loginfo('OSC: Desired_pose reaching joint limit')
            return False

        return True

    """
    ROS related
    """

    def _step_robot(self, pose):
        self.robot.goto_pose(pose, use_impedance=self.use_frankapy_impedance, cartesian_impedances=list(self.impedances),
                             duration=1)

    def start_episode(self, horizon=None, dynamic_command=False):
        if not dynamic_command:
            return

        steps = self.horizon if horizon is None else horizon
        if self.dt is None or steps is None:
            rospy.loginfo("Running OSC dynamic_command but self.dt or steps is None.")

        """Start ROS Publisher"""

        rospy.loginfo('OSC: Initializing Sensor Publisher')
        self.ros_pub = rospy.Publisher(FC.DEFAULT_SENSOR_PUBLISHER_TOPIC, SensorDataGroup, queue_size=1000)
        self.ros_rate = rospy.Rate(self.control_freq)
        self.message_count = 0

        rospy.loginfo('OSC: Publishing pose trajectory...')
        # To ensure skill doesn't end before completing trajectory, make the buffer time much longer than needed
        p0 = self.robot.get_pose().copy()
        self.robot.goto_pose(p0, duration=self.dt * steps, dynamic=True, buffer_time=10,
                             cartesian_impedances=list(self.impedances), use_impedance=self.use_frankapy_impedance)
        self.episode_init_time = rospy.Time.now().to_time()
        self._publisher_running = True

    def _step_robot_dynamic(self, pose):
        assert self._publisher_running, "Need to call env.start_episode() before calling env.step()."
        timestamp = rospy.Time.now().to_time() - self.episode_init_time
        traj_gen_proto_msg = PosePositionSensorMessage(
            id=self.message_count, timestamp=timestamp,
            position=pose.translation, quaternion=pose.quaternion
        )
        fb_ctrlr_proto = CartesianImpedanceSensorMessage(
            id=self.message_count, timestamp=timestamp,
            translational_stiffnesses=list(self.impedances[:3]),
            rotational_stiffnesses=list(self.impedances[3:]),
        )
        ros_msg = make_sensor_group_msg(
            trajectory_generator_sensor_msg=sensor_proto2ros_msg(
                traj_gen_proto_msg, SensorDataMessageType.POSE_POSITION),
            feedback_controller_sensor_msg=sensor_proto2ros_msg(
                fb_ctrlr_proto, SensorDataMessageType.CARTESIAN_IMPEDANCE)
        )

        rospy.loginfo('OSC: Publishing: ID {}'.format(traj_gen_proto_msg.id))
        self.message_count += 1
        self.ros_pub.publish(ros_msg)
        self.ros_rate.sleep()

    def end_episode(self):
        # Stop the skill
        if self._publisher_running:
            # Alternatively can call self.robot.stop_skill()
            term_proto_msg = ShouldTerminateSensorMessage(timestamp=rospy.Time.now().to_time() - self.episode_init_time,
                                                          should_terminate=True)
            ros_msg = make_sensor_group_msg(
                termination_handler_sensor_msg=sensor_proto2ros_msg(
                    term_proto_msg, SensorDataMessageType.SHOULD_TERMINATE)
            )
            self.ros_pub.publish(ros_msg)
            self._publisher_running = False
            self.message_count = None
            rospy.loginfo('OSC: Episode finished.')
            self.robot.stop_skill()


class OSCXZPlane(OSC):
    def __init__(self, *args, **kwargs):
        kwargs['control_axis'] = np.array([1., 0., 1., 0., 1., 0.])
        kwargs['controller_type'] = "OSC_XZPLANE"
        super().__init__(*args, **kwargs)


"""
Helper functions
"""


def orientation_error(desired, current):
    """
    This function calculates a 3-dimensional orientation error vector for use in the
    impedance controller. It does this by computing the delta rotation between the
    inputs and converting that rotation to exponential coordinates (axis-angle
    representation, where the 3d vector is axis * angle).
    See https://en.wikipedia.org/wiki/Axis%E2%80%93angle_representation for more information.
    Optimized function to determine orientation error from matrices

    Args:
        desired (np.array): 2d array representing target orientation matrix
        current (np.array): 2d array representing current orientation matrix

    Returns:
        np.array: 2d array representing orientation error as a matrix
    """
    rc1 = current[0:3, 0]
    rc2 = current[0:3, 1]
    rc3 = current[0:3, 2]
    rd1 = desired[0:3, 0]
    rd2 = desired[0:3, 1]
    rd3 = desired[0:3, 2]

    error = 0.5 * (np.cross(rc1, rd1) + np.cross(rc2, rd2) + np.cross(rc3, rd3))

    return error


def angle_diff(vec1, vec2, degree=True):
    vec1 = vec1 / np.linalg.norm(vec1)
    vec2 = vec2 / np.linalg.norm(vec2)
    prod = np.dot(vec1, vec2)
    prod = np.clip(prod, -1, 1)
    angle = np.arccos(prod)
    if degree:
        angle = angle / np.pi * 180
    return angle
