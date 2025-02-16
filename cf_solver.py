import gym
import numpy as np
import torch
from matplotlib import pyplot as plt


class Solver:
    """
    This class implements the closed form solution for the following quantities:
    - Value function (V)
    - Action-value function (Q)
    - State distribution (mu_pi)
    - State Action distribution (mu_pi_a)
    - Advantage function (A)
    - J function/Score function (Expected return of the policy)
    - Policy gradient
    """
    def __init__(self, mdp):

        self.n_states: int = mdp.info.observation_space.size[0]
        self.n_actions: int = mdp.info.action_space.size[0]
        self.gamma: float = mdp.info.gamma

        self.start_state = 0
        goal_state: int = mdp.info.observation_space.size[0] - 1

        # initialize initial_state distribution
        self.mu_0: torch.Tensor = torch.zeros(self.n_states)
        self.mu_0[self.start_state] = 1.0
        # self.mu_0 = torch.ones(self.n_states) / self.n_states

        # Transition matrix in the form s x s x a
        self.p_a: torch.Tensor = self.transitions_to_matrix(mdp.env.unwrapped.P)

        # Reward matrix in the form s x a
        self.r_a: torch.Tensor = torch.zeros((self.n_states, self.n_actions))
        self.r_a[goal_state, :] = 1.0

    def transitions_to_matrix(self, transitions: dict) -> torch.Tensor:
        """
        Method to convert the transition probabilities to a matrix
        """
        P = torch.zeros((self.n_states, self.n_actions, self.n_states))
        for state in range(self.n_states):
            for action in range(self.n_actions):
                for prob, next_state, _, done in transitions[state][action]:
                    P[state, action, next_state] += prob
        return P

    def get_p_r(self, policy_table: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        """
        This method computes the transition probability matrix and reward vector for a given policy pi
        Args:
            policy_table: This is the matrix of the policy pi

        Returns: The transition probability matrix and reward vector in the form s x s x a and s x a
        """
        p = torch.einsum('ijk,ij->ik', self.p_a, policy_table)
        r = torch.einsum('ij,ij->i', self.r_a, policy_table)
        return p, r

    def get_mu_pi_a(self, p_pi: torch.Tensor, policy_table: torch.Tensor) -> torch.Tensor:
        """
        This method computes the state-action distribution for a given policy pi.
        Args:
            p_pi: This is the nxn array of the state transitions based on the policy pi
            policy_table: Policy table of shape (s, a)

        Returns: The state-action distribution in the form 1 x sa

        """
        mu = self.get_mu_pi(p_pi)
        mu_pi = mu.repeat(self.n_actions).reshape(self.n_actions, -1).T * policy_table
        mu_pi = torch.ravel(mu_pi)
        return mu_pi

    def get_mu_pi_check(self, p_pi):
        sum_state_dis = torch.zeros_like(self.mu_0)
        states_t = self.mu_0
        disc_state_t = self.mu_0
        i = 1
        while True:
            sum_state_dis += disc_state_t
            states_t = states_t.matmul(p_pi)
            disc_state_t = self.gamma ** i * states_t
            i += 1
            if torch.norm(disc_state_t) < 1e-100:
                break
            if i >= 1000:
                break
        # Normalize the state distribution that it sums up to 1
        state_dis_t = sum_state_dis * (1 - self.gamma)
        return state_dis_t

    def get_mu_pi(self, p_pi: torch.Tensor) -> torch.Tensor:
        """
        This method computes the state distribution for a given policy pi
        Args:
            p_pi: The state transition matrix based on the policy pi

        Returns: The state distribution in the form s x 1

        """
        mu = (1 - self.gamma) * torch.linalg.solve(torch.eye(p_pi.shape[0]) - self.gamma * p_pi.T, self.mu_0)
        return mu

    def get_v(self, p_pi: torch.Tensor, r_pi: torch.Tensor) -> torch.Tensor:
        """
        Method to compute the closed form solution for the value function.
        Derived from the Bellman Equation the formular (I - gamma * P_pi)^-1 * R_pi is used.
        where:
        - I is the identity matrix
        - gamma is the discount factor
        - P_pi is the transition matrix under the policy
        - R_pi is the reward vector under the policy
        Args:
            p_pi: The state transition matrix based on the policy pi
            r_pi: The reward vector based on the policy pi

        Returns: The value function in the form s x 1
        """
        v = torch.linalg.inv(torch.eye(p_pi.shape[0]) - self.gamma * p_pi) @ r_pi
        return v

    def get_q(self, p_pi: torch.Tensor, r_pi: torch.Tensor) -> torch.Tensor:
        """
        This method computes the q-function for a given policy pi
        Formular q = R + gamma * P_pi * v_pi is used.
        Args:
            P_pi: This is the array of the state transitions based on the policy pi
            r_pi: This is the array of the rewards based on the policy pi

        Returns: q-function in the form s x a

        """
        v = self.get_v(p_pi, r_pi)
        # Form s0a0, s0a1, s0a2, s0a3, s1a0, s1a1, ...
        trans_val = self.p_a.matmul(v)
        # Transform to a s x a matrix
        trans_val_reshaped = trans_val.reshape(self.n_states, self.n_actions)
        # q_s_a = self.r_a + self.gamma * trans_val_reshaped
        q = torch.ravel(self.r_a + self.gamma * trans_val_reshaped)
        # v = self.get_v(P_pi, r_pi)
        # q = torch.ravel(self.r_a + self.gamma * self.p_a @ v)
        q = q.reshape(self.n_states, self.n_actions)
        return q

    # Methode to compute the advantage function
    def get_A(self, p_pi: torch.Tensor, r_pi: torch.Tensor) -> torch.Tensor:
        """
        This method computes the advantage function for a given policy pi
        The formular q - v is used.
        """
        q = self.get_q(p_pi, r_pi).reshape(self.n_states, self.n_actions)
        v = self.get_v(p_pi, r_pi).reshape(self.n_states, 1)
        return q - v

    # Computes the closed form solution for the J function which represents the expected return of the policy
    def get_score_v(self, policy_table: torch.Tensor) -> torch.Tensor:
        """
        This method computes the J function for a given policy pi using the v-function
        """
        p, r = self.get_p_r(policy_table)
        v = self.get_v(p, r)
        return torch.inner(self.mu_0, v)

    def get_score_q(self, policy_table: torch.Tensor):
        """
        This method computes the J function for a given policy pi using the q-function
        """
        p, r = self.get_p_r(policy_table)
        mu_pi = self.get_mu_pi(p).detach()
        q = self.get_q(p, r).detach()
        expected_q_per_state = torch.sum(policy_table * q, dim=1)
        score = torch.sum(mu_pi * expected_q_per_state)
        return score

    def get_score_q_beta(self, policy_table: torch.Tensor, bias_term: np.ndarray):
        """
        This method computes the J function for a given policy pi using the q-function
        :param policy_table: The policy table
        :param bias_term: A _positive_ bias that we add to each state shape = (n_states)
        """
        p, r = self.get_p_r(policy_table)
        mu_pi = self.get_mu_pi(p).detach()

        mu_pi_biased = mu_pi + bias_term

        mu_pi_biased = mu_pi_biased / torch.sum(mu_pi_biased)
        q = self.get_q(p, r).detach()
        expected_q_per_state = torch.sum(policy_table * q, dim=1)
        score = torch.sum(mu_pi_biased * expected_q_per_state)
        return score

    def get_score_q_alpha(self, policy_table, bias_indices=None, scaling_factor=1.5):
        """
        This method computes the J function for a given policy pi using the q-function
        """
        if bias_indices is None:
            bias_indices = [8]
        p, r = self.get_p_r(policy_table)
        mu_pi = self.get_mu_pi(p).detach()

        scaling_vector = torch.ones_like(mu_pi)
        scaling_vector[bias_indices] = scaling_factor
        mu_pi_biased = mu_pi * scaling_vector

        # Re-normalize to ensure it's a valid distribution:
        mu_pi_biased = mu_pi_biased / torch.sum(mu_pi_biased)
        q = self.get_q(p, r).detach()
        expected_q_per_state = torch.sum(policy_table * q, dim=1)
        score = torch.sum(mu_pi_biased * expected_q_per_state)
        return score

    def get_score_q_gamma(self, policy_table, bias_indices=None, scaling_factor=1.5):
        """
        This method computes the J function for a given policy pi using the q-function
        """
        if bias_indices is None:
            bias_indices = [8]
        p, r = self.get_p_r(policy_table)
        mu_pi = self.get_mu_pi(p).detach()

        scaling_vector = torch.ones_like(mu_pi)
        scaling_vector[bias_indices] = scaling_factor
        mu_pi_biased = mu_pi * scaling_vector

        scaling_vector = torch.ones_like(mu_pi)
        scaling_vector[[1, 4, 5]] = 0.5
        mu_pi_biased = mu_pi_biased * scaling_vector

        # Re-normalize to ensure it's a valid distribution:
        mu_pi_biased = mu_pi_biased / torch.sum(mu_pi_biased)
        q = self.get_q(p, r).detach()
        expected_q_per_state = torch.sum(policy_table * q, dim=1)
        score = torch.sum(mu_pi_biased * expected_q_per_state)
        return score

    def policy_gradient(self, policy_table: torch.Tensor, grad_log_table: torch.Tensor) -> torch.Tensor:
        """
        This method computes the policy gradient for a given policy pi
        """
        p, r = self.get_p_r(policy_table)
        mu_pi_a = self.get_mu_pi_a(p, policy_table)
        q = self.get_q(p, r)
        return grad_log_table.T @ (mu_pi_a * q)


"""
from mushroom_rl.environments import Gym

n_states = 16
n_actions = 4

mdp = Gym('FrozenLake-v1', is_slippery=False, gamma=.9, render_mode="rgb_array")

solver = Solver(mdp)
grad_log_table = torch.rand((solver.n_states * solver.n_actions, 32))
phi_table = torch.rand((solver.n_states * solver.n_actions, solver.n_states))
policy_table = torch.zeros((n_states, n_actions))

# Fill the policy table
for state in range(n_states):
    # Choose a primary action with high probability
    primary_action = state % n_actions
    policy_table[state, primary_action] = 0.9
    policy_table[state, :] += 0.1 / n_actions

score = solver.get_score_q(policy_table)
print(f"Compare score: {score}")

score_beta = solver.get_score_q_beta(policy_table)
print(f"Compare score beta: {score_beta}")
"""