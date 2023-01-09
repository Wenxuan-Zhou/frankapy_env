import numpy as np
import open3d as o3d
import os
import rospy
import time
from autolab_core import RigidTransform
from collections import OrderedDict

from frankapy_env.franka_env import FrankaEnv
from frankapy_env.pointcloud import PointCloudModule, box, ground


class OccludedGraspingEnv(FrankaEnv):
    """
    Defines task-specific functions of the Occluded Grasping task.
    """

    def __init__(self, offline=False, init_joints=(0, 0.15, 0, -2.44, 0, 2.62, -7.84e-01), controller="OSC",
                 control_freq=None, horizon=None, object_name="Box-0"):

        super().__init__(offline, init_joints, controller, control_freq, horizon)

        self.object_name = object_name  # Used to load object model for ICP and decide gripper width
        self.grip_width = 0.08

        # Setup point cloud
        self.pc = None
        self.setup_pc_module()

    def setup_pc_module(self):
        """
        Setup point cloud observation and pose estimation
        """
        self.pc = PointCloudModule(icp_threshold=0.01, init_node=False, z_min=0.065)
        self.pc.add_visualizations(ground())
        if self.object_name == "Box-0":
            self.pc.template = box()  # Box-0
        elif self.object_name == "Box-1":
            self.pc.template = box(box_size=(0.154, 0.292, 0.058))  # Box-1
            self.grip_width = 0.038
        elif self.object_name == "Box-2":
            self.pc.template = box(box_size=(0.153, 0.222, 0.074))  # Box-2
            self.grip_width = 0.055
        elif self.object_name == "Box-3":
            self.pc.template = box(box_size=(0.165, 0.245, 0.052))  # Box-3
        else:
            object_pcd_file = os.path.join(os.path.dirname(__file__), 'scanned_objects', self.object_name + ".pcd")
            self.pc.template = o3d.io.read_point_cloud(object_pcd_file)
            if self.object_name == "container" or self.object_name == "container_reverse":
                self.grip_width = 0.045
            elif self.object_name == "largebottle":
                self.grip_width = 0.01

        self.pc.template.colors = o3d.utility.Vector3dVector(
            np.array([(1, 0, 0) for _ in range(len(self.pc.template.points))]))
        object_pose = np.eye(4)
        object_pose[:3, 3] = -np.array([0.55, 0, 0.09])
        self.pc.icp_result = object_pose
        return

    def step(self, action: np.ndarray):
        # Execute the action
        self.robot.open_gripper(block=False)
        action = np.clip(np.array(action), a_min=-1, a_max=1)
        self.controller.run_controller(action)

        # Update the point cloud
        if self.pc is not None:
            self.pc.update()
            self.pc.run_icp()

        reward = 0
        done = False
        obs = self._get_observations()
        info = obs  # Include full_obs as info
        return obs, reward, done, info

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

        # Object related
        if self.pc is not None:
            obs['pc'] = self.pc.pc
            obs['pc_icp'] = self.pc.get_transformed_template_pc()  # ICP result
            object_pose_mat = self.pc.template_pose
        else:
            obs['pc'] = None
            obs['pc_icp'] = None
            object_pose_mat = np.eye(4)
        object_pose = RigidTransform(rotation=object_pose_mat[:3, :3],
                                     translation=object_pose_mat[:3, 3],
                                     from_frame='object', to_frame='world')
        obs['object_pose'] = object_pose.vec  # + np.array([0.02, 0, 0, 0, 0, 0, 0])
        obs['ee_relative_to_object'] = (object_pose.inverse() * ee_pose).vec  # fingertip
        obs['ee_base_relative_to_object'] = (object_pose.inverse() * ee_base_pose).vec

        # Goal related
        obs['desired_goal'] = np.array([-0.152, 0., 0., 0.70710678, 0., 0.70710678, 0.])

        desired_goal = RigidTransform(rotation=obs['desired_goal'][3:],
                                      translation=obs['desired_goal'][:3],
                                      from_frame='franka_tool_base',
                                      to_frame='object')  # Defined at gripper base
        obs['desired_ee_relative_to_object'] = (desired_goal * base_to_finger).vec
        return obs

    def start_episode(self, horizon=None, dynamic_command=True):
        # Verify state estimation first
        if self.pc is not None:
            self.pc.update()
            self.pc.run_icp()
            self.pc.draw_icp()
        super().start_episode(horizon, dynamic_command)

    def end_episode(self):
        self.controller.end_episode()
        self.grasp_and_lift()  # Try to lift the object as a measure of success
        rospy.loginfo('END OF EPISODE.')

    def grasp_and_lift(self):
        rospy.loginfo("Try to grasp and lift the object.")
        self.robot.goto_gripper(self.grip_width)
        T_ee_world = self.robot.get_pose()
        T_ee_world.translation += [-0.05, 0, 0.10]
        self.robot.goto_pose(T_ee_world)
        time.sleep(2)
