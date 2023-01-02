'''
References:
    https://github.com/felixchenfy/open3d_ros_pointcloud_conversion
    https://stackoverflow.com/questions/39772424/how-to-effeciently-convert-ros-pointcloud2-to-pcl-point-cloud-and-visualize-it-i
'''

import copy
import datetime
import ipdb
import numpy as np
import open3d as o3d
import os
import ros_numpy
import rospy
import time
import yaml
from sensor_msgs.msg import PointCloud2

PC_RAW = None
PC_TIMESTAMP = None
UPDATE_PC = True
CALIBRATION_FILE = os.path.join(os.path.dirname(__file__), 'easy_handeye', 'easy_handeye_eye_on_base.yaml')
FINETUNE_CALIBRATION_FILE = os.path.join(os.path.dirname(__file__), 'easy_handeye', 'finetune_calibration.npy')


class PointCloudModule(object):
    def __init__(self,
                 rosnode_name='pointcloud_module',
                 init_node=True,
                 transformation=None,
                 x_min=0.4,
                 x_max=0.75,
                 y_min=-0.2,
                 y_max=0.2,
                 z_min=0.01,
                 z_max=0.25,
                 down_sample=5,
                 icp_initial_solution=None,
                 icp_threshold=0.02,
                 ):
        self.pc = None
        self.pc_timestamp = None
        self.transformation = np.eye(4) if transformation is None else transformation
        self.load_transformation(FINETUNE_CALIBRATION_FILE)
        self.visualizations = []

        # Processing parameters
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        self.z_min = z_min
        self.z_max = z_max
        self.down_sample = down_sample

        # ICP-related
        self.template = None
        self.icp_result = np.eye(4) if icp_initial_solution is None else icp_initial_solution
        self.icp_threshold = icp_threshold

        if init_node:
            rospy.init_node(rosnode_name, anonymous=True)

        def callback(ros_cloud):
            global PC_RAW, PC_TIMESTAMP, UPDATE_PC
            if ros_cloud is not None and UPDATE_PC:
                PC_RAW = ros_cloud
                PC_TIMESTAMP = time.time()
            return

        rospy.Subscriber('/points2', PointCloud2, callback)

    def load_transformation(self, filename):
        if '.yaml' in filename:
            calibration = yaml.load(open(filename, 'rb'), Loader=yaml.FullLoader)
            calibration = calibration['transformation']
            trans = np.array([calibration['x'], calibration['y'], calibration['z']])
            quat = np.array([calibration['qw'], calibration['qx'], calibration['qy'], calibration['qz']])
            R = o3d.geometry.get_rotation_matrix_from_quaternion(quat)
            T = np.eye(4)
            T[:3, :3] = R
            T[:3, 3] = trans
            self.transformation = T
        elif '.npy' in filename:
            with open(filename, 'rb') as f:
                self.transformation = np.load(f)
        return

    def tune_transformation(self):
        print("Adjust camera calibration to align the object orientation.")
        # Tune camera calibration to align the current object orientation to zero rotation.
        self.load_transformation(CALIBRATION_FILE)
        print("Loaded original calibration file.")
        self.update()
        self.run_icp()
        self.draw_icp()

        # Move pc object center
        template_pose = self.template_pose
        T1 = np.eye(4)
        T1[:3, 3] = -template_pose[:3, 3]
        # Rotate pc to align with [1,0,0,0]
        R = np.eye(4)
        R[:3, :3] = template_pose[:3, :3].transpose()
        T2 = np.eye(4)
        T2[:3, 3] = template_pose[:3, 3]
        T2[:3, 3] = np.array([0.65, 0, 0.09])  # Assume the object is at x=0.65, z=0.09
        # T2[:3, 3] = np.array([0.65, template_pose[1, 3], 0.09])  # Assume the object is at x=0.65, z=0.09
        print('Object Rotation before:\n', template_pose[:3, :3], '\nObject Rotation After:\n', np.eye(3))
        print('Object Translation before:\n', template_pose[:3, 3], '\nObject Translation After\n:', T2[:3, 3])
        self.transformation = T2.dot(R.dot(T1.dot(self.transformation))).copy()

        # Update template pose and run icp again
        P = np.eye(4)
        P[:3, 3] = self.template_pose[:3, 3]
        self.icp_result = inv(P)
        self.update()
        self.run_icp()
        self.draw_icp()
        with open(FINETUNE_CALIBRATION_FILE, 'wb') as f:
            np.save(f, self.transformation)
        # print('Transformation:')
        # print(self.transformation)
        print(f'Saved at {FINETUNE_CALIBRATION_FILE}.')
        return

    def update(self):
        global UPDATE_PC, PC_TIMESTAMP
        # Stop updating the point cloud and make a local copy
        UPDATE_PC = False
        pc_raw_local = PC_RAW
        pc_timestamp_local = PC_TIMESTAMP
        UPDATE_PC = True

        if pc_raw_local is None:
            rospy.loginfo("-- No point cloud.")
            return

        pc_o3d = convert_to_o3d(pc_raw_local)           # Convert ros_cloud to open3d point cloud
        pc_o3d = pc_o3d.uniform_down_sample(self.down_sample)
        pc_o3d = pc_o3d.transform(self.transformation)  # Transform into robot coordinate
        pc_o3d = crop(pc_o3d, x_min=self.x_min, x_max=self.x_max, y_min=self.y_min, y_max=self.y_max,
                      z_min=self.z_min, z_max=self.z_max)
        self.pc = pc_o3d
        self.pc_timestamp = pc_timestamp_local
        return

    def load(self, filename):
        self.pc = o3d.io.read_point_cloud(filename)
        self.pc_timestamp = None
        return

    def draw(self):
        o3d.visualization.draw_geometries([self.pc] + self.visualizations)
        return

    def save(self, filename=None):
        if filename is None:
            filename = str(datetime.datetime.now()) + '.pcd'
        o3d.io.write_point_cloud(filename, self.pc)
        rospy.loginfo("-- Write result point cloud to: " + filename)

    def add_visualizations(self, pointcloud=None, filename=None):
        if filename is not None:
            pointcloud = o3d.io.read_point_cloud(filename)
        self.visualizations.append(pointcloud)

    def run_icp(self):
        t = time.time()
        reg_p2p = o3d.pipelines.registration.registration_icp(
            self.pc, self.template, self.icp_threshold, self.icp_result,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            o3d.pipelines.registration.ICPConvergenceCriteria(relative_fitness=1e-06, relative_rmse=1e-06, max_iteration=100))
        rospy.loginfo("ICP: Delay: {:.2f}\t Fitness {:.2f} \t MSE {:.2e}"
                      .format(time.time()-t, reg_p2p.fitness, reg_p2p.inlier_rmse))
        self.icp_result = reg_p2p.transformation
        return

    def draw_icp(self):
        template_copy = copy.deepcopy(self.template)
        template_copy = template_copy.transform(self.template_pose)
        o3d.visualization.draw_geometries([self.pc, template_copy] + self.visualizations)
        return

    def get_transformed_template_pc(self):
        template_copy = copy.deepcopy(self.template)
        template_copy = template_copy.transform(self.template_pose)
        return template_copy

    def save_icp(self, filename=None):
        t = time.time()
        if filename is None:
            filename = 'icp ' + str(datetime.datetime.now()) + '.pcd'
        template_copy = copy.deepcopy(self.template)
        template_copy = template_copy.transform(self.template_pose)
        self.pc.colors = o3d.utility.Vector3dVector(np.array([[0, 0, 1] for _ in range(len(self.pc.points))]))
        o3d.io.write_point_cloud(filename, self.pc + template_copy)
        rospy.loginfo("-- Write result point cloud to: " + filename)
        rospy.loginfo("File saving delay:" + str(time.time()-t))

    @property
    def template_pose(self):
        return inv(self.icp_result)


def inv(rgt):
    rgt_inv = np.eye(4)
    rgt_inv[:3, :3] = rgt[:3, :3].transpose()
    rgt_inv[:3, 3] = -rgt_inv[:3, :3] @ rgt[:3, 3]
    return rgt_inv


def convert_to_o3d(data):
    pc = ros_numpy.numpify(data)
    points = np.zeros((pc.shape[0], 3), dtype=np.float64)
    points[:, 0] = pc['x']
    points[:, 1] = pc['y']
    points[:, 2] = pc['z']
    points = np.stack([pc['x'], pc['y'], pc['z']], axis=1).astype(np.float64)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    return pcd


def crop(pc, x_min=None, x_max=None, y_min=None, y_max=None, z_min=None, z_max=None):
    pc_o3d_points = np.asarray(pc.points)
    x, y, z = pc_o3d_points[:, 0], pc_o3d_points[:, 1], pc_o3d_points[:, 2]
    pfilter = np.ones_like(x, dtype=bool)
    if x_min is not None:
        pfilter *= (x > x_min)
    if x_max is not None:
        pfilter *= (x < x_max)
    if y_min is not None:
        pfilter *= (y > y_min)
    if y_max is not None:
        pfilter *= (y < y_max)
    if z_min is not None:
        pfilter *= (z > z_min)
    if z_max is not None:
        pfilter *= (z < z_max)
    pc.points = o3d.utility.Vector3dVector(pc_o3d_points[pfilter, :])
    return pc


def ground():
    # Ground
    x = np.linspace(0, 1, 31)
    y = np.linspace(-0.5, 0.5, 31)
    mesh_x, mesh_y = np.meshgrid(x, y)
    xyz = np.zeros((np.size(mesh_x), 3))
    xyz[:, 0] = np.reshape(mesh_x, -1)
    xyz[:, 1] = np.reshape(mesh_y, -1)
    ground_pcd = o3d.geometry.PointCloud()
    ground_pcd.points = o3d.utility.Vector3dVector(xyz)
    return ground_pcd


def box(box_size=(0.15, 0.20, 0.05), color=(1, 0, 0)):
    box_size = np.array(box_size)
    # Object
    x = np.linspace(0, box_size[0], int(box_size[0] * 100 * 2))
    y = np.linspace(0, box_size[1], int(box_size[1] * 100 * 2))
    z = np.linspace(0, box_size[2], int(box_size[2] * 100 * 2))
    points = []
    # Top, bottom
    mesh_x, mesh_y = np.meshgrid(x, y)
    xyz = np.zeros((np.size(mesh_x), 3))
    xyz[:, 0] = np.reshape(mesh_x, -1)
    xyz[:, 1] = np.reshape(mesh_y, -1)
    xyz[:, 2] = 0
    points.append(xyz.copy())
    xyz[:, 2] = box_size[2]
    points.append(xyz.copy())
    # Left, right
    mesh_x, mesh_z = np.meshgrid(x, z)
    xyz = np.zeros((np.size(mesh_x), 3))
    xyz[:, 0] = np.reshape(mesh_x, -1)
    xyz[:, 1] = 0
    xyz[:, 2] = np.reshape(mesh_z, -1)
    points.append(xyz.copy())
    xyz[:, 1] = box_size[1]
    points.append(xyz.copy())
    # Front, back
    mesh_y, mesh_z = np.meshgrid(y, z)
    xyz = np.zeros((np.size(mesh_y), 3))
    xyz[:, 0] = 0
    xyz[:, 1] = np.reshape(mesh_y, -1)
    xyz[:, 2] = np.reshape(mesh_z, -1)
    points.append(xyz.copy())
    xyz[:, 0] = box_size[0]
    points.append(xyz.copy())

    points = np.concatenate(points)
    points += -box_size / 2
    object_pcd = o3d.geometry.PointCloud()
    object_pcd.points = o3d.utility.Vector3dVector(points)
    object_pcd.colors = o3d.utility.Vector3dVector(np.array([color for i in range(len(points))]))
    return object_pcd
