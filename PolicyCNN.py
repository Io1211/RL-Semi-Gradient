import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from matplotlib import pyplot as plt
from mushroom_rl.environments import Gym

from CF.GridworldWrapper import FrozenLakeWrapper
from CF.cf_solver import Solver


class CNNPolicy(nn.Module):
    def __init__(self, n_actions, input_shape=(256, 256, 3)):
        """
        Initialize the CNN model.
        :param n_actions: Number of actions.
        :param input_shape: Shape of the input image in channels-first format (H, W, C)
        """
        super(CNNPolicy, self).__init__()

        self.input_shape = input_shape
        self.conv1 = nn.Conv2d(in_channels=input_shape[2], out_channels=16, kernel_size=5, stride=2)
        # self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=2)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, stride=2)

        self._conv_output_size = self._get_conv_output(input_shape)

        self.fc1 = nn.Linear(self._conv_output_size, 128)
        self.fc2 = nn.Linear(128, n_actions)

    def _get_conv_output(self, shape):
        """
        Methode to get the output size of the convolutional layers. Rearranged to match PyTorch format.
        :param shape: Tuple (H, W, C)
        :return: number of output features
        """
        dummy_input = torch.zeros(1, shape[2], shape[0], shape[1])
        x = F.relu(self.conv1(dummy_input))
        # x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        size = int(np.prod(x.size()[1:]))
        return size

    def preprocess(self, img):
        """
        Preprocess the image before feeding it to the model by resizing and normalizing it.
        returns: tensor of the preprocessed image (1, C, H, W)
        """
        img = cv2.resize(img, (self.input_shape[1], self.input_shape[0]))
        img = img.astype(np.float32) / 255.0
        tensor = torch.from_numpy(img).permute(2, 0, 1)
        return tensor

    def forward(self, img):
        if isinstance(img, np.ndarray):
            img = self.preprocess(img)
        if img.dim() == 3:
            img = img.unsqueeze(0)
        img = F.relu(self.conv1(img))
        img = F.relu(self.conv2(img))
        img = F.relu(self.conv3(img))

        img = img.reshape(img.size(0), -1)
        img = F.relu(self.fc1(img))
        logits = self.fc2(img)
        policy = F.softmax(logits, dim=1)
        return policy


class PolicyTrainerCNN:

    def __init__(self, n_states, n_actions, input_shape, wrapper, mdp, sample_bias, lr=0.001):
        self.n_actions = n_actions
        self.n_states = n_states
        self.wrapper = wrapper
        self.solver = Solver(mdp)
        self.sample_bias = sample_bias
        self.input_shape = input_shape
        self.policy_model = CNNPolicy(n_actions, input_shape)
        self.optimizer = torch.optim.Adam(self.policy_model.parameters(), lr=lr)
        self.returns = []

        # Set up device (GPU if available)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_model = CNNPolicy(n_actions, input_shape).to(self.device)
        self.optimizer = torch.optim.Adam(self.policy_model.parameters(), lr=lr)
        self.returns = []

        # Precompute and cache all state images (assuming states do not change)
        self.all_state_imgs = self._precompute_all_state_imgs()

    def _precompute_all_state_imgs(self):
        imgs = []
        for s in range(self.n_states):
            img = self.wrapper.get_img_for_state(s)
            tensor_img = self.policy_model.preprocess(img)
            imgs.append(tensor_img)
        return torch.stack(imgs, dim=0).to(self.device)

    def get_policy_table(self):
        """
        Get the policy table for all states.
        """
        policy_list = []
        for s in range(self.n_states):
            img = self.wrapper.get_img_for_state(s)
            policy = self.policy_model(img)
            policy_list.append(policy)
        policy_table = torch.cat(policy_list, dim=0)
        return policy_table

    def compute_entropy(self):
        """
        Compute the average entropy of the policy. Adds a small value to the log to avoid numerical instability (log(0))
        """
        return torch.sum(self.get_policy_table() * torch.log(self.get_policy_table() + 1e-10), dim=1).mean()

    def train(self, iterations=100, log_interval=10):
        for iteration in range(iterations):
            policy_table = self.get_policy_table()

            metric = self.solver.get_score_v(policy_table)
            self.returns.append(metric.item())

            entropy = torch.sum(policy_table * torch.log(policy_table + 1e-10), dim=1).mean()
            J = self.solver.get_score_q_beta(policy_table, self.sample_bias)
            loss = -(J - 0.01 * entropy)

            self.optimizer.zero_grad()
            loss.backward()

            self.optimizer.step()

            if iteration % log_interval == 0:
                print(f"Iteration {iteration}: J = {metric.item()}")

        torch.save(self.policy_model.state_dict(), '../CF/cnn_policy_model.pth')
        print(f"Final policy: {policy_table}")
        return self.returns


def compute_policy_img(wrapper, sample_bias: np.array, iterations=1000, log_interval=100):
    wrapper.mdp.reset()
    n_states = wrapper.mdp.env.observation_space.n
    n_actions = wrapper.mdp.env.action_space.n
    trainer = PolicyTrainer_CNN(n_states, n_actions, wrapper.get_img().shape, wrapper, wrapper.mdp, sample_bias, lr=0.00001)
    returns = trainer.train(iterations, log_interval)

    policy_model = CNNPolicy(n_actions, wrapper.get_img().shape)
    policy_model.load_state_dict(torch.load('../CF/cnn_policy_model.pth'))
    return returns


mdp = Gym('FrozenLake-v1', is_slippery=False, gamma=.9, render_mode="rgb_array")
fl = FrozenLakeWrapper(mdp, depiction=0)
for i in range(3):
    sample_bias = np.abs(np.random.rand(mdp.env.observation_space.n))
    returns = compute_policy_img(fl, sample_bias=sample_bias, iterations=1000, log_interval=100)
    plt.plot(returns)

unbiased_return = compute_policy_img(fl, sample_bias=np.zeros(mdp.env.observation_space.n), iterations=1000)
plt.plot(unbiased_return, label='Unbiased')

plt.xlabel('Iterations')
plt.ylabel('Return')
plt.title('Return over Training')
plt.legend()
plt.show()

