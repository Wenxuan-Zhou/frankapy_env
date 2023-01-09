# frankapy_env

A Gym wrapper for the Franka Emika Panda robot based on FrankaPy. 
This repository includes the code for the real robot experiments in [Learning to Grasp the Ungraspable with Emergent Extrinsic Dexterity](https://sites.google.com/view/grasp-ungraspable). 
You may find the simulation environment used for Sim2Real transfer in [this repository](https://github.com/Wenxuan-Zhou/ungraspable).

The gym environment of the franka arm is defined in [frankapy_env/franka_env.py](frankapy_env/franka_env.py). 
ROS point cloud processing code and object pose estimation based on Iterative closest point (ICP) can be found in [frankapy_env/pointcloud.py](frankapy_env/pointcloud.py).
If you are interested in the details of the Occluded Grasping task in our paper, please checkout [frankapy_env/occluded_grasping_env.py](frankapy_env/occluded_grasping_env.py).

Please feel free to contact us if you have any questions on the code or anything else related to our paper!

## Installation
1. Robot Setup: Install frankapy and franka-interface by following the instructions [here](https://github.com/iamlab-cmu/frankapy).
2. Camera Setup (Azure Kinect):
   1. Install Azure SDK:
      1. [Useful installation notes](https://gist.github.com/madelinegannon/c212dbf24fc42c1f36776342754d81bc)
      2. [Official instructions](https://docs.microsoft.com/en-us/azure/Kinect-dk/sensor-sdk-download)
      3. Before going to the next step, verify the camera reading in "k4aviewer".
   2. Install [Azure Kinect ROS Driver](https://github.com/microsoft/Azure_Kinect_ROS_Driver). 
   3. Run camera calibration using [easy_hand_eye](https://github.com/IFL-CAMP/easy_handeye).
      1. Replace the yaml file in this repository: [easy_handeye_eye_on_base.yaml](frankapy_env%2Feasy_handeye%2Feasy_handeye_eye_on_base.yaml)
3. Install this repository.
    ```
    pip install -r requirements.txt
    ```
4. (Optional) If you want to try the example code of policy rollout, you also need to install [rlkit](https://github.com/rail-berkeley/rlkit) and [pytorch](https://pytorch.org/).
   1. You only need to ```pip install git+https://github.com/rail-berkeley/rlkit.git``` for rlkit without installing additional dependencies if you just want to try out the rollout code in this repository.

## Usage
To try out the FrankaPy Gym Environment:
```
python scripts/test_env.py
```
To rollout with a policy:
```
python scripts/rollout_policy.py models/example.pkl
```

## Citation
If you find this repository useful, please cite our paper:
```
@inproceedings{zhou2022ungraspable,
  title={Learning to Grasp the Ungraspable with Emergent Extrinsic Dexterity},
  author={Zhou, Wenxuan and Held, David},
  booktitle={Conference on Robot Learning (CoRL)},
  year={2022}
}
```
