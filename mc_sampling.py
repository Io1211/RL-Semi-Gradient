import torch
from mushroom_rl.environments import Gym
from CF.cf_solver import Solver


class MCSampling:
    """
    This class implements Monte Carlo sampling to check the closed form solution quantities.
    """

    def __init__(self, solver, n_episodes=1000, max_steps=1000):
        self.solver = solver
        self.n_episodes = n_episodes
        self.max_steps = max_steps

    # Sample a trajectory using the policy table
    def sample_trajectory(self, policy_table):
        trajectory = []
        state = self.solver.start_state
        P, r = self.solver.get_P_r(policy_table)
        r_a = self.solver.r_a
        for _ in range(self.max_steps):
            action_probs = policy_table[state].numpy()
            action = torch.multinomial(torch.tensor(action_probs), 1).item()
            next_state = torch.multinomial(self.solver.p_a[state, action], 1).item()
            reward = r_a[state, action]
            trajectory.append((state, action, reward))
            state = next_state

        return trajectory

    # Estimate the value function using every-visit Monte Carlo sampling
    def mc_value_estimation(self, policy_table):
        returns = {state: [] for state in range(self.solver.n_states)}
        value_estimates = torch.zeros(self.solver.n_states)
        for _ in range(self.n_episodes):
            trajectory = self.sample_trajectory(policy_table)
            G = 0
            for state, action, reward in reversed(trajectory):
                G = reward + self.solver.gamma * G
                returns[state].append(G)

            for state in range(self.solver.n_states):
                if returns[state]:
                    value_estimates[state] = torch.mean(torch.tensor(returns[state]))

        return value_estimates

    # Estimate the Q function using every-visit Monte Carlo sampling
    def mc_q_value_estimation(self, policy_table):
        q_values = torch.zeros((self.solver.n_states, self.solver.n_actions))
        returns = {(state, action): [] for state in range(self.solver.n_states) for action in
                   range(self.solver.n_actions)}
        for _ in range(self.n_episodes):
            trajectory = self.sample_trajectory(policy_table)
            G = 0
            for state, action, reward in reversed(trajectory):
                G = reward + self.solver.gamma * G
                returns[(state, action)].append(G)
                q_values[state, action] = torch.mean(torch.tensor(returns[(state, action)]))

        return torch.ravel(q_values)

    # Estimate the state distribution using Monte Carlo sampling avg the discounted visits
    def mc_mu(self, policy_table):
        mu = torch.zeros(self.solver.n_states)
        for _ in range(self.n_episodes):
            trajectory = self.sample_trajectory(policy_table)
            i = 0
            for state, _, _ in trajectory:
                mu[state] += 1 * (self.solver.gamma ** i)
                i += 1
        return mu / torch.sum(mu)

    def mc_gradient(self, policy_table, grad_log_table):
        grad = torch.zeros((self.solver.n_states, self.solver.n_actions))
        mu = self.mc_mu(policy_table)
        for _ in range(self.n_episodes):
            trajectory = self.sample_trajectory(policy_table)
            G = 0
            for state, action, reward in reversed(trajectory):
                G = reward + self.solver.gamma * G
                grad[state, action] += mu[state] * G
        grad = (grad_log_table * grad) / self.n_episodes
        return grad

    def check_value_function(self, policy_table):
        mc_v = self.mc_value_estimation(policy_table)
        P, r = self.solver.get_P_r(policy_table)
        solver_v = self.solver.get_v(P, r)

        mse = torch.mean((mc_v - solver_v) ** 2)
        return mse.item()

    def check_q_function(self, policy_table):
        mc_q = self.mc_q_value_estimation(policy_table)
        P, r = self.solver.get_P_r(policy_table)
        solver_q = self.solver.get_q(P, r)

        mse = torch.mean((mc_q - solver_q) ** 2)
        return mse.item()

    def check_mu(self, policy_table):
        mc_mu = self.mc_mu(policy_table)
        P, _ = self.solver.get_P_r(policy_table)
        solver_mu = self.solver.get_mu_pi(P)

        mse = torch.mean((mc_mu - solver_mu) ** 2)
        return mse.item()


# Check the Monte Carlo sampling

n_states = 16
n_actions = 4

mdp = Gym('FrozenLake-v1', is_slippery=False, gamma=.9, render_mode="human")
solver = Solver(mdp)

policy_table = torch.zeros((n_states, n_actions))

# Fill the policy table
for state in range(n_states):
    # Choose a primary action with high probability
    primary_action = state % n_actions
    policy_table[state, primary_action] = 0.9
    policy_table[state, :] += 0.1 / n_actions

mc_sampling = MCSampling(solver, n_episodes=100, max_steps=1000)
check_v = mc_sampling.check_value_function(policy_table)
check_q = mc_sampling.check_q_function(policy_table)
print(f"Check value function: {check_v}")
print(f"Check Q function: {check_q}")