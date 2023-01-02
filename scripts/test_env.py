import numpy as np

from frankapy_env import FrankaEnv
from frankapy_env.rollout import rollout, FakeAgent

if __name__ == "__main__":
    print("\n"*2+">"*20)
    print("Directly controller the robot by calling step")
    print("<"*20)
    env = FrankaEnv(controller="OSC_XZPLANE")
    env.reset()
    env.step(np.array([1., 0., 0.]))
    env.close()

    print("\n"*2+">"*20)
    print("Rollout an action sequence.")
    print("(every step is blocking)")
    print("<"*20)
    actions = np.array([[0., 1., 0.]] * 5)
    policy = FakeAgent(actions=actions)
    env = FrankaEnv(controller="OSC_XZPLANE")
    rollout(env, policy, max_path_length=len(actions), dynamic_command=False)
    env.close()

    print("\n"*2+">"*20)
    print("Rollout an action sequence.")
    print("(non-blocking, keep sending commands to the robot)")
    print("<"*20)
    actions = np.array([[0., 1., 0]] * 5)
    policy = FakeAgent(actions=actions)
    env = FrankaEnv(controller="OSC_XZPLANE", control_freq=1, horizon=len(actions))
    rollout(env, policy, max_path_length=len(actions), dynamic_command=True)
    env.close()

    print("\n"*2+">"*20)
    print("Rollout an action sequence with a 6D controller.")
    print("(non-blocking, keep sending commands to the robot)")
    print("<"*20)
    actions = np.array([[0., 0., 0., 0., 1., 0.]] * 5)
    policy = FakeAgent(actions=actions)
    env = FrankaEnv(controller="OSC", control_freq=2, horizon=len(actions))
    rollout(env, policy, max_path_length=len(actions), dynamic_command=True)
    env.close()
