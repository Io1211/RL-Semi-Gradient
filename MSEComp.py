import copy

import numpy as np
import torch
from matplotlib import pyplot as plt
from mushroom_rl.environments import Gym
from torch import optim

from NN.CoordinateNN import CoordinatePolicy
from CF.GridworldWrapper import FrozenLakeWrapper, GridWorldWrapper
from NN.SimpleNN import ParameterizedPolicy
from CF.cf_solver import Solver


class PolicyTrainer:
    """
    A class to train a policy model using policy gradient ascent.
    :param model: The policy model to train
    :param wrapper: The wrapped environment
    :param sample_bias: The sampled positive bias of the surrogate loss function
    """

    def __init__(self, model: torch.nn.Module, wrapper: GridWorldWrapper, sample_bias: np.array, lr=0.01):
        self.n_states = wrapper.mdp.env.observation_space.n
        self.n_actions = wrapper.mdp.env.action_space.n
        self.mdp = wrapper.mdp
        self.sample_bias = sample_bias
        self.solver = Solver(self.mdp)
        self.policy_model_1 = copy.deepcopy(model)
        self.policy_model_2 = copy.deepcopy(model)
        self.optimizer_unb = optim.Adam(self.policy_model_1.parameters(), lr=lr)
        self.optimizer_b = optim.Adam(self.policy_model_2.parameters(), lr=lr)
        self.returns_b = []
        self.returns_unb = []
        self.gradient_mse_layer1 = []
        self.gradient_mse_layer2 = []
        self.gradient_mse_layer3 = []
        self.gradient_mse_layer4 = []
        self.gradient_mse_layer1_b = []
        self.gradient_mse_layer2_b = []
        self.gradient_mse_layer3_b = []
        self.gradient_mse_layer4_b = []

    def train(self, iterations=1000, log_interval=100):
        """
        Train the policy model using policy gradient ascent.
        """

        # the bias vector should be constant for all iterations
        for iteration in range(iterations):
            policy_table = self.policy_model_1.get_policy_table()
            metric_un = self.solver.get_score_v(policy_table)

            # use unbiased surrogate loss
            J_update_un = self.solver.get_score_q_beta(policy_table, bias_term=self.sample_bias * 0)
            loss_un = -J_update_un

            self.optimizer_unb.zero_grad()
            loss_un.backward()

            # -todo: save the gradient of each layer
            with torch.no_grad():
                for name, param in self.policy_model_1.named_parameters():
                    if 'fc1' in name and param.grad is not None:
                        if param.grad.shape == torch.Size([16, 2]):
                            g_w_1 = param.grad
                        elif param.grad.shape == torch.Size([16]):
                            g_b_1 = param.grad
                    if 'fc2' in name and param.grad is not None:
                        if param.grad.shape == torch.Size([32, 16]):
                            g_w_2 = param.grad
                        elif param.grad.shape == torch.Size([32]):
                            g_b_2 = param.grad
                    if 'fc3' in name and param.grad is not None:
                        if param.grad.shape == torch.Size([32, 32]):
                            g_w_3 = param.grad
                        elif param.grad.shape == torch.Size([32]):
                            g_b_3 = param.grad
                    if 'fc4' in name and param.grad is not None:
                        if param.grad.shape == torch.Size([self.n_actions, 32]):
                            g_w_4 = param.grad
                        elif param.grad.shape == torch.Size([self.n_actions]):
                            g_b_4 = param.grad

            self.optimizer_unb.step()

            self.returns_unb.append(metric_un.item())

            # use the biased surrogate loss
            policy_table = self.policy_model_2.get_policy_table()
            metric_b = self.solver.get_score_v(policy_table)

            J_update_b = self.solver.get_score_q_beta(policy_table, bias_term=self.sample_bias)
            loss_b = -J_update_b

            self.optimizer_b.zero_grad()
            loss_b.backward()

            with torch.no_grad():
                for name, param in self.policy_model_2.named_parameters():
                    if 'fc1' in name and param.grad is not None:
                        if param.grad.shape == torch.Size([16, 2]):
                            g_w_1_b = param.grad
                        elif param.grad.shape == torch.Size([16]):
                            g_b_1_b = param.grad
                    if 'fc2' in name and param.grad is not None:
                        if param.grad.shape == torch.Size([32, 16]):
                            g_w_2_b = param.grad
                        elif param.grad.shape == torch.Size([32]):
                            g_b_2_b = param.grad
                    if 'fc3' in name and param.grad is not None:
                        if param.grad.shape == torch.Size([32, 32]):
                            g_w_3_b = param.grad
                        elif param.grad.shape == torch.Size([32]):
                            g_b_3_b = param.grad
                    if 'fc4' in name and param.grad is not None:
                        if param.grad.shape == torch.Size([self.n_actions, 32]):
                            g_w_4_b = param.grad
                        elif param.grad.shape == torch.Size([self.n_actions]):
                            g_b_4_b = param.grad
            # todo compute the mean square error between the gradients of each layer

            self.gradient_mse_layer1.append(torch.mean((g_w_1 - g_w_1_b) ** 2))
            self.gradient_mse_layer1_b.append(torch.mean((g_b_1 - g_b_1_b) ** 2))
            self.gradient_mse_layer2.append(torch.mean((g_w_2 - g_w_2_b) ** 2))
            self.gradient_mse_layer2_b.append(torch.mean((g_b_2 - g_b_2_b) ** 2))
            self.gradient_mse_layer3.append(torch.mean((g_w_3 - g_w_3_b) ** 2))
            self.gradient_mse_layer3_b.append(torch.mean((g_b_3 - g_b_3_b) ** 2))
            self.gradient_mse_layer4.append(torch.mean((g_w_4 - g_w_4_b) ** 2))
            self.gradient_mse_layer4_b.append(torch.mean((g_b_4 - g_b_4_b) ** 2))

            # todo cumulate all the mse for each layer

            self.optimizer_b.step()

            self.returns_b.append(metric_b.item())

            if iteration % log_interval == 0:
                print(f"Iteration {iteration}: J = {metric_b.item()}")

        # todo return the cumulated mse for each layer
        """
        plt.plot(self.gradient_mse_layer1, label='fc1')
        plt.plot(self.gradient_mse_layer1_b, label='fc2_b')
        plt.plot(self.gradient_mse_layer2, label='fc2')
        plt.plot(self.gradient_mse_layer2_b, label='fc2_b')
        plt.plot(self.gradient_mse_layer3, label='fc3')
        plt.plot(self.gradient_mse_layer3_b, label='fc3_b')
        plt.plot(self.gradient_mse_layer4, label='fc4')
        plt.plot(self.gradient_mse_layer4_b, label='fc4_b')
        plt.xlabel('Iterations')
        plt.ylabel('MSE')
        plt.title('MSE between biased and unbiased gradients')
        plt.legend()
        plt.show()
        """
        return self.returns_b, sum(self.gradient_mse_layer1), sum(self.gradient_mse_layer2), sum(
            self.gradient_mse_layer3), sum(self.gradient_mse_layer4), sum(self.gradient_mse_layer1_b), sum(
            self.gradient_mse_layer2_b), sum(self.gradient_mse_layer3_b), sum(self.gradient_mse_layer4_b)


def compute_policy(model, wrapper, sample_bias, iterations=1000):
    """
    Returns the policy and the expected return using the given model to compute the policy.
    :param model: The model to use for computing the policy
    :param wrapper: The wrapped environment
    :param sample_bias: The sampled positive bias of the surrogate loss function
    """
    wrapper.mdp.reset()
    trainer = PolicyTrainer(model, wrapper, sample_bias, lr=0.01)
    returns = trainer.train(iterations)
    policy = model.get_policy_table()
    print("Final policy table: " + str(policy))
    return returns


def compare_learning(model, wrapper, instance, iterations=1000, log_interval=100):
    """
    Compare the learning of the model using objective function with different biases.
    """
    mse_layer1 = []
    mse_layer2 = []
    mse_layer3 = []
    mse_layer4 = []
    mse_layer1_b = []
    mse_layer2_b = []
    mse_layer3_b = []
    mse_layer4_b = []
    for i in range(instance):
        sample_bias = np.abs(np.random.rand(wrapper.mdp.env.observation_space.n))
        returns, mse1, mse2, mse3, mse4, mse1_b, mse2_b, mse3_b, mse4_b = compute_policy(model, wrapper, sample_bias,
                                                                                         iterations)
        mse_layer1.append(mse1)
        mse_layer2.append(mse2)
        mse_layer3.append(mse3)
        mse_layer4.append(mse4)
        mse_layer1_b.append(mse1_b)
        mse_layer2_b.append(mse2_b)
        mse_layer3_b.append(mse3_b)
        mse_layer4_b.append(mse4_b)
        model.reset()
    # Create a figure with two subplots for the line plots.
    instances_arr = np.arange(1, instance + 1)
    fig, axs = plt.subplots(1, 2, figsize=(14, 6), sharex=True)

    # Plot aggregated MSE for weights.
    axs[0].plot(instances_arr, mse_layer1, marker='o', linestyle='-', label='fc1 weights')
    axs[0].plot(instances_arr, mse_layer2, marker='o', linestyle='-', label='fc2 weights')
    axs[0].plot(instances_arr, mse_layer3, marker='o', linestyle='-', label='fc3 weights')
    axs[0].plot(instances_arr, mse_layer4, marker='o', linestyle='-', label='fc4 weights')
    axs[0].set_title("Aggregated MSE for Weights")
    axs[0].set_xlabel("Instance")
    axs[0].set_ylabel("Aggregated MSE")
    axs[0].grid(True)
    axs[0].legend()

    # Plot aggregated MSE for biases.
    axs[1].plot(instances_arr, mse_layer1_b, marker='s', linestyle='--', label='fc1 bias')
    axs[1].plot(instances_arr, mse_layer2_b, marker='s', linestyle='--', label='fc2 bias')
    axs[1].plot(instances_arr, mse_layer3_b, marker='s', linestyle='--', label='fc3 bias')
    axs[1].plot(instances_arr, mse_layer4_b, marker='s', linestyle='--', label='fc4 bias')
    axs[1].set_title("Aggregated MSE for Biases")
    axs[1].set_xlabel("Instance")
    axs[1].set_ylabel("Aggregated MSE")
    axs[1].grid(True)
    axs[1].legend()

    plt.tight_layout()
    plt.show()

    # Compute cumulative MSE for each layer.
    cumulated_mse = {
        'fc1 weights': sum(mse_layer1),
        'fc2 weights': sum(mse_layer2),
        'fc3 weights': sum(mse_layer3),
        'fc4 weights': sum(mse_layer4),
        'fc1 bias': sum(mse_layer1_b),
        'fc2 bias': sum(mse_layer2_b),
        'fc3 bias': sum(mse_layer3_b),
        'fc4 bias': sum(mse_layer4_b)
    }
    labels = list(cumulated_mse.keys())
    values = list(cumulated_mse.values())

    # Plot a bar chart for cumulative MSE.
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(labels, values, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
                                         '#9467bd', '#8c564b', '#e377c2', '#7f7f7f'])
    ax.set_title("Cumulative MSE for Each Layer Parameter")
    ax.set_ylabel("Cumulative MSE")
    ax.set_xticklabels(labels, rotation=45, ha='right')

    # Annotate each bar with its value.
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, yval + 0.05, f'{yval:.2f}',
                ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.show()


mdp = Gym('FrozenLake-v1', is_slippery=False, gamma=.9, render_mode="rgb_array")
wrapped = FrozenLakeWrapper(mdp, depiction=0)
wrapped.mdp.reset()
model_table = ParameterizedPolicy(wrapped)
model_co = CoordinatePolicy(wrapped)
compare_learning(model_co, wrapped, 7)
