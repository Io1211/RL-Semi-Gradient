import torch

class ParameterizedPolicy(torch.nn.Module):
    """
    A simple parameterized policy model, which is a linear layer followed by a softmax.
    """

    def __init__(self, wrapper):
        super().__init__()
        self.n_states = wrapper.mdp.env.observation_space.n
        self.n_actions = wrapper.mdp.env.action_space.n
        self.theta = torch.nn.Parameter(torch.zeros((self.n_states, self.n_actions)))

    def forward(self):
        return torch.nn.functional.softmax(self.theta, dim=1)

    def reset(self):
        self.theta = torch.nn.Parameter(torch.zeros(self.theta.shape))

    def get_policy_table(self):
        return self.forward()
