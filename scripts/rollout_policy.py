import argparse
import torch
from rlkit.torch.pytorch_util import set_gpu_mode
from rlkit.torch.sac.policies import MakeDeterministic

from frankapy_env.occluded_grasping_env import OccludedGraspingEnv
from frankapy_env.rollout import rollout
from frankapy_env.wrappers import ObsWrapper

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('model_path', type=str, help='path to a pkl file')
    args = parser.parse_args()

    # Create Env
    env = ObsWrapper(OccludedGraspingEnv(horizon=30, control_freq=1,
                                         controller="OSC_XZPLANE", object_name="largebottle"))

    # Load Policy
    map_location = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    data = torch.load(args.model_path, map_location=map_location)
    policy = data['evaluation/policy']
    assert isinstance(policy, MakeDeterministic)
    if torch.cuda.is_available():
        set_gpu_mode(True)
        policy.cuda()
    print(f"Policy loaded from {args.model_path}")

    # Rollout
    path = rollout(env, policy, max_path_length=env.horizon, dynamic_command=True)
    env.close()
