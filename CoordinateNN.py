import torch


class CoordinatePolicy(torch.nn.Module):
    """
    A simple nn model that takes coordinates as input and outputs action probabilities for each state.
    """

    def __init__(self, wrapper):
        super().__init__()
        self.n_states = wrapper.mdp.env.observation_space.n
        self.n_actions = wrapper.mdp.env.action_space.n
        self.wrapper = wrapper
        self.fc1 = torch.nn.Linear(2, 16)
        #self.fc2 = torch.nn.Linear(16, n_actions)
        self.fc2 = torch.nn.Linear(16, 32)
        self.fc3 = torch.nn.Linear(32, 32)
        self.fc4 = torch.nn.Linear(32, self.n_actions)

        self.coords = []
        for state in range(self.n_states):
            cor = self.wrapper.get_coordinates_for_state(state)
            self.coords.append(cor)
        self.coords = torch.tensor(self.coords, dtype=torch.float32)

    def forward(self, coords):
        """
        Forward pass of the network. Expects a tensor of all coordinates and returns action probabilities.
        :param coords: Tensor of shape (n_states, 2)
        :return: Tensor of shape (n_states, n_actions)
        """
        #logits = self.fc3(torch.relu(self.fc2(torch.relu(self.fc1(coords)))))
        #logits = self.fc2(torch.relu(self.fc1(coords)))
        logits = self.fc4(self.fc3(torch.relu(self.fc2(torch.relu(self.fc1(coords))))))
        return torch.nn.functional.softmax(logits, dim=1)

    def get_policy_table(self):
        """
        Get the policy table for all states. Computes the coordinates for each state and forwards them as a
        batch(n_state, 2).
        returns: tensor of shape (n_states, n_actions)
        """
        policy = self.forward(self.coords)
        return policy

    def reset(self):
        """
        Reset the network parameters.
        """
        self.fc1 = torch.nn.Linear(2, 16)
        self.fc2 = torch.nn.Linear(16, 32)
        self.fc3 = torch.nn.Linear(32, 32)
        self.fc4 = torch.nn.Linear(32, self.n_actions)

