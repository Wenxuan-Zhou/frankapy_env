import numpy as np
from autolab_core import transformations
from gym.core import ObservationWrapper


class ObsWrapper(ObservationWrapper):

    def observation(self, obs):
        controller_type = self.controller.controller_type
        if '3D' in controller_type or 'XZPLANE' in controller_type:
            reduced_pose = clean_xzplane_pose
        else:
            reduced_pose = clean_6d_pose

        di = dict()
        di['gripper_pos'] = obs['ee_base_pose'][:3]
        di['gripper_quat'] = obs['ee_base_pose'][3:]
        di['cube_pos'] = obs['object_pose'][:3]
        di['cube_quat'] = obs['object_pose'][3:]
        di['achieved_goal'] = obs['ee_base_relative_to_object']
        di['desired_goal'] = obs['desired_goal']

        # Simplified obs
        gripper_pose = reduced_pose(di['gripper_pos'], di['gripper_quat'], offset=False)
        cube_pose = reduced_pose(di['cube_pos'], di['cube_quat'], offset=True)
        achieved_goal_pos = reduced_pose(di['achieved_goal'][:3], di['achieved_goal'][3:], offset=True)
        di['observation'] = np.concatenate([gripper_pose, cube_pose, achieved_goal_pos, di['desired_goal']])
        return di['observation']


def clean_xzplane_pose(pos, quat, offset=False):
    if offset:
        # An ugly implementation that avoids discontinuity of euler output.
        rotate_back = transformations.quaternion_from_euler(0, -np.pi / 2, 0)
        quat = transformations.quaternion_multiply(quat, rotate_back)
    euler = transformations.euler_from_quaternion(quat)
    return np.array([pos[0], pos[2], -euler[1]])


def clean_6d_pose(pos, quat, offset=False):
    return np.concatenate([pos, quat])
