# Bachelor_Thesis_RL

## Mission

This project implements a reinforcement learning framework for training and evaluating policy models in a gridworld
environment. The goal is to determine the effect of the semi-gradient bias.

## Overview

The project consists of the following components:

- **Gridworld Environment:**  
  A gridworld environment with a fixed layout and rewards at specific locations. The environment provides functions for
  stepping through the grid, receiving rewards, and checking for terminal states.
- **Policy Models:**  
  Policy models that define the agent's behavior in the gridworld. The models include a tabular policy and a neural
  network policy.
- **Reinforcement Learning Algorithms:**  
  Reinforcement learning algorithms for training the policy models. The algorithms include Q-learning and semi-gradient
  Q-learning.
- **Training and Evaluation:**  
  Scripts for training and evaluating the policy models in the gridworld environment. The evaluation includes
  visualizing the agent's path and calculating the total reward.