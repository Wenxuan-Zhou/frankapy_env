import gym
import ipdb
import numpy as np
import rospy
import signal
import time
from abc import ABC
from autolab_core import RigidTransform
from collections import OrderedDict
from frankapy import FrankaArm

from frankapy_env.osc import OSC, OSCXZPlane


class FrankaEnv(gym.Env, ABC):
    """
    Basic functionalities of the franka robot wrapped as a gym environment
    """

    def __init__(self,
                 offline=False,
                 init_joints=(0, 0.15, 0, -2.44, 0, 2.62, -7.84e-01),
                 controller="OSC",
                 control_freq=None,
                 horizon=None,
                 ):

        # Setup robot
        self.robot = FrankaArm(offline=offline)
        self.init_joints = np.array(init_joints)

        # Setup controller
        self.horizon = horizon
        if controller.upper() == "OSC":
            self.controller = OSC(self.robot, control_freq=control_freq, horizon=horizon)
        elif controller.upper() == "OSC_XZPLANE":
            self.controller = OSCXZPlane(self.robot, control_freq=control_freq, horizon=horizon)
        else:
            raise NotImplementedError

        self.action_space = gym.spaces.Box(low=-1., high=1., shape=(self.controller.control_dim,))
        signal.signal(signal.SIGINT, self.signal_handler)

    def signal_handler(self, signal_received, frame):
        print('SIGINT or CTRL-C detected.')
        self.close()
        exit(0)

    def reset(self):
        rospy.loginfo('RESET...')
        self.controller.end_episode()
        # Reset robot
        self.robot.goto_joints(self.init_joints)
        self.robot.open_gripper()
        while self.robot.get_gripper_width() < 0.07:
            print("Cannot open gripper.")
            ipdb.set_trace()
        rospy.loginfo('Moved to ready pose.')
        time.sleep(1)
        # Reset controller
        self.controller.reset()
        return self._get_observations()

    def step(self, action: np.ndarray):
        # Execute the action
        self.robot.open_gripper(block=False)
        action = np.clip(np.array(action), a_min=-1, a_max=1)
        self.controller.run_controller(action)

        reward = 0
        done = False
        obs = self._get_observations()
        info = obs  # Include full_obs as info
        return obs, reward, done, info

    def get_info(self):
        return self._get_observations()

    def _get_observations(self):
        # All the poses should be [translation(xyz), quaternion(wxyz)]
        obs = OrderedDict()

        # Robot related
        robot_state = self.robot.get_robot_state()
        robot_state.pop('pose')  # remove because this is in RigidTransform format
        obs.update(robot_state)
        ee_pose = self.robot.get_pose()
        obs['ee_pose'] = ee_pose.vec
        obs['desired_ee_pose'] = self.controller.current_desired_pose.vec
        obs['jacobian'] = self.robot.get_jacobian(self.robot.get_joints())
        obs['ee_vel'] = self.robot.get_ee_velocity()
        obs['ee_vel_from_franka_interface'] = self.robot.get_ee_velocity(from_franka_interface=True)

        base_to_finger = RigidTransform(rotation=RigidTransform.rotation_from_quaternion([0.707107, 0, 0, 0.707107]),
                                        translation=np.array([0., 0., 0.097]),
                                        from_frame='franka_tool',
                                        to_frame='franka_tool_base')
        ee_base_pose = ee_pose * base_to_finger.inverse()
        obs['ee_base_pose'] = ee_base_pose.vec

        return obs

    def start_episode(self, horizon=None, dynamic_command=True):
        # Countdown
        rospy.loginfo("EPISODE COUNTDOWN (3s):")
        for i in range(3, 0, -1):
            rospy.loginfo(i)
            time.sleep(1)
        rospy.loginfo("START!")

        self.controller.start_episode(horizon=horizon, dynamic_command=dynamic_command)

    def end_episode(self):
        self.controller.end_episode()
        rospy.loginfo('END OF EPISODE.')

    def close(self):
        self.controller.end_episode()
        self.robot.stop_skill()
        self.robot.open_gripper()
        if self.robot.get_gripper_width() < 0.07:
            print("Cannot open gripper.")
            ipdb.set_trace()

        # Move up gently
        pose = self.robot.get_pose()
        pose.translation[0] -= 0.1
        pose.translation[2] += 0.1
        impedance = [100, 100, 100, 10, 10, 10]
        self.robot.goto_pose(pose, cartesian_impedances=impedance)

        self.reset()
        rospy.loginfo('Env closed.')
