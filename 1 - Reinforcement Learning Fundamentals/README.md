# RL2024_Ex1
## Exercise 1 - Reinforcement Learning Foundamentals

This repository contains a set of experiments focused on balancing a CartPole system using classical control techniques and reinforcement learning (RL). The CartPole problem is a standard benchmark in control theory and machine learning, where the goal is to balance a pole on a moving cart by applying appropriate forces.

## Objectives
The main objectives of this experiment are:
1. Evaluate a Linear Quadratic Regulator (LQR) for controlling the CartPole system
2. Train reinforcement learning agents to balance the pole using reward-based learning
3. Compare the performance, flexibility, and reliability of these approaches

## Methods
**LQR Control**: A classical control approach based on linearizing the dynamics of the CartPole system. The regulator computes optimal control inputs by minimizing a cost function with weighting matrices Q and R.

**Reinforcement Learning**: A data-driven approach where agents learn control strategies through interaction with the environment. This was implemented using a standard RL algorithm (e.g., DQN or PPO) with reward functions tailored for balancing, centering, or inducing specific cart behaviors.

## Results

The results and analysis are presented in the report file contained in this repository. In summary:

- **LQR** provided stable and predictable control performance, with outcomes dependent on the selection of weighting matrices. It demonstrated low computational cost but limited adaptability to nonlinearities or changes in dynamics.
2. **Reinforcement Learning**: Achieved successful balancing through trial and error. Custom reward functions enabled flexible behavior tuning, but training required substantial computational effort and showed variability due to stochastic learning.

## Files
- `Lab1 Report s331998.pdf`: Report file containing the results and analysis
- `lqr_control.py`: Implementation of the LQR controller
- `rl_training.py`: Reinforcement learning training and evaluation script
- `agent.py`: RL agent class definition
- `multiple_cartpoles_rl.py`: Script for training multiple RL agents and plotting results
- `utils.py`: Utility functions
- `requirements.txt`: List of dependencies
- `README.md`: Overview of the project