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
        self.policy_model = model
        self.optimizer = optim.Adam(self.policy_model.parameters(), lr=lr)
        self.returns = []

    def train(self, iterations=1000, log_interval=100):
        """
        Train the policy model using policy gradient ascent.
        """
        # the bias vector should be constant for all iterations
        for iteration in range(iterations):
            policy_table = self.policy_model.get_policy_table()
            metric = self.solver.get_score_v(policy_table)

            J_update = self.solver.get_score_q_beta(policy_table, bias_term=self.sample_bias)
            loss = -J_update

            self.optimizer.zero_grad()
            loss.backward()

            # TODO: perturb the gradient of a layer

            self.optimizer.step()

            self.returns.append(metric.item())

            if iteration % log_interval == 0:
                print(f"Iteration {iteration}: J = {metric.item()}")

        return self.returns

    def train_biased_layer(self, iterations=1000, log_interval=100, biased_layer=0, in_layer=0, out_layer=0):
        """
        Train the policy model using policy gradient ascent.
        """

        bias_w = np.abs(np.random.rand(out_layer, in_layer))
        bias_b = np.abs(np.random.rand(out_layer))
        bias_w = torch.tensor(bias_w, dtype=torch.double)
        bias_b = torch.tensor(bias_b, dtype=torch.double)
        # the bias vector should be constant for all iterations
        for iteration in range(iterations):
            policy_table = self.policy_model.get_policy_table()
            metric = self.solver.get_score_v(policy_table)

            J_update = self.solver.get_score_q_beta(policy_table, bias_term=self.sample_bias*0)
            loss = -J_update

            self.optimizer.zero_grad()
            loss.backward()

            with torch.no_grad():
                for name, param in self.policy_model.named_parameters():
                    if 'fc1' in name and param.grad is not None and biased_layer == 1:
                        print("fc1")
                        if param.grad.shape == torch.Size([16, 2]):
                            param.grad += bias_w
                        elif param.grad.shape == torch.Size([16]):
                            param.grad += bias_b
                    if 'fc2' in name and param.grad is not None and biased_layer == 2:
                        print("fc2")
                        if param.grad.shape == torch.Size([32, 16]):
                            param.grad += bias_w
                        elif param.grad.shape == torch.Size([32]):
                            param.grad += bias_b
                    if 'fc3' in name and param.grad is not None and biased_layer == 3:
                        print("fc3")
                        if param.grad.shape == torch.Size([32, 32]):
                            param.grad += bias_w
                        elif param.grad.shape == torch.Size([32]):
                            param.grad += bias_b
                    if 'fc4' in name and param.grad is not None and biased_layer == 4:
                        if param.grad.shape == torch.Size([self.n_actions, 32]):
                            param.grad += bias_w
                        elif param.grad.shape == torch.Size([self.n_actions]):
                            param.grad += bias_b


            # TODO: perturb the gradient of a layer

            self.optimizer.step()

            self.returns.append(metric.item())

            if iteration % log_interval == 0:
                print(f"Iteration {iteration}: J = {metric.item()}")

        return self.returns


def compute_policy(model, wrapper, sample_bias, iterations=1000):
    """
    Returns the policy and the expected return using the given model to compute the policy.
    :param model: The model to use for computing the policy
    :param wrapper: The wrapped environment
    :param sample_bias: The sampled positive bias of the surrogate loss function
    """
    wrapper.mdp.reset()
    trainer = PolicyTrainer(model, wrapper, sample_bias, lr=0.01)
    sample_bias = np.abs(np.random.rand(wrapper.mdp.env.observation_space.n, 2))
    returns = trainer.train_biased_layer(iterations, biased_layer=2, in_layer=16, out_layer=32)
    policy = model.get_policy_table()
    print("Final policy table: " + str(policy))
    return returns


def compare_learning(model, wrapper, instance, iterations=1000, log_interval=100):
    """
    Compare the learning of the model using objective function with different biases.
    """
    for i in range(instance):
        sample_bias = np.abs(np.random.rand(wrapper.mdp.env.observation_space.n))
        returns = compute_policy(model, wrapper, sample_bias, iterations)
        plt.plot(returns)
        model.reset()
    unbiased_return = compute_policy(model, wrapper, np.zeros(wrapper.mdp.env.observation_space.n), iterations)
    plt.plot(unbiased_return, label='Unbiased')
    plt.xlabel('Iterations')
    plt.ylabel('Return')
    plt.title('Return over Training')
    plt.legend()
    plt.show()

def compare_layer_biase(model, wrapper, instance, iterations=1000, log_interval=100):
    """
    Compare the learning of the model using objective function with different biases.
    """
    for i in range(instance):
        sample_bias = np.abs(np.random.rand(wrapper.mdp.env.observation_space.n))
        returns = compute_policy(model, wrapper, sample_bias, iterations)
        plt.plot(returns)
        model.reset()
    unbiased_return = compute_policy(model, wrapper, np.zeros(wrapper.mdp.env.observation_space.n), iterations)
    plt.plot(unbiased_return, label='Unbiased')
    plt.xlabel('Iterations')
    plt.ylabel('Return')
    plt.title('Return over Training')
    plt.legend()
    plt.show()

mdp = Gym('FrozenLake-v1', is_slippery=False, gamma=.9, render_mode="rgb_array")
wrapped = FrozenLakeWrapper(mdp, depiction=0)
wrapped.mdp.reset()
model_table = ParameterizedPolicy(wrapped)
model_co = CoordinatePolicy(wrapped)
compare_learning(model_co, wrapped, 10)


