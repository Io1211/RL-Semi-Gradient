import numpy as np
from mushroom_rl.environments import Gym

from CF.GridworldWrapper import FrozenLakeWrapper

mdp = Gym('FrozenLake-v1', is_slippery=False, gamma=.9, render_mode="human")

# Define the given policy
policy_table = np.array([[1.3214e-42, 9.5657e-01, 4.3432e-02, 5.9935e-39],
                         [1.0440e-42, 1.8538e-03, 9.9815e-01, 2.1503e-40],
                         [3.1109e-43, 9.9737e-01, 2.6321e-03, 3.7833e-39],
                         [7.7506e-42, 4.6441e-01, 5.3559e-01, 1.1730e-38],
                         [8.8282e-44, 9.9975e-01, 2.5237e-04, 2.2306e-39],
                         [6.5483e-42, 4.7915e-01, 5.2085e-01, 1.0817e-38],
                         [3.3771e-43, 9.9744e-01, 2.5638e-03, 3.8909e-39],
                         [6.3689e-42, 4.8767e-01, 5.1233e-01, 1.0457e-38],
                         [3.9657e-43, 2.4115e-04, 9.9976e-01, 3.7664e-41],
                         [1.5807e-42, 9.6868e-01, 3.1316e-02, 7.5303e-39],
                         [2.2701e-43, 9.9873e-01, 1.2739e-03, 3.1043e-39],
                         [6.4572e-42, 4.8045e-01, 5.1955e-01, 1.0695e-38],
                         [6.4684e-42, 4.8428e-01, 5.1572e-01, 9.9478e-39],
                         [1.1743e-42, 2.1920e-03, 9.9781e-01, 2.3918e-40],
                         [9.7250e-43, 1.7303e-03, 9.9827e-01, 1.9411e-40],
                         [3.1389e-42, 5.5084e-01, 4.4916e-01, 5.7170e-39]])


def run_policy(policy_table):
    wrapper = FrozenLakeWrapper(mdp, depiction=0)
    wrapper.mdp.reset()
    done = False
    total_reward = 0
    state = wrapper.get_state()

    while not done:
        action = np.argmax(policy_table[state])
        print(f"State: {state}, Action: {action}")
        state, reward, done, _ = mdp.step([action])
        total_reward += reward

    return total_reward
