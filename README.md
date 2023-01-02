# frankapy_env

A Gym wrapper for the Franka Emika Panda robot based on FrankaPy. This repository includes the code for the real robot experiments in [Learning to Grasp the Ungraspable with Emergent Extrinsic Dexterity](https://sites.google.com/view/grasp-ungraspable?pli=1). 
Feel free to let me know if you have any questions: wenxuanz@andrew.cmu.edu.

## Updates
[Jan 01, 2023] Initial code release.

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
