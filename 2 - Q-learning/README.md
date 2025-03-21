# RL2024_Ex2
## Exercise 2 - Q-learning for CartPole

This repository contains a set of experiments focused on balancing a CartPole system using Tabular Q-Learning and Deep Q-Learning. The CartPole problem is a standard benchmark in control theory and machine learning, where the goal is to balance a pole on a moving cart by applying appropriate forces.

## Objectives
The main objectives of this experiment are:
1. Evaluate Tabular Q-Learning by discretizing the state and action spaces of the CartPole environment
2. Implement Deep Q-Learning using a neural network to approximate the Q-function
3. Compare the performance, flexibility, and reliability of these approaches and the parameters that influence them

## Methods
**Tabular Q-Learning**: A reinforcement learning algorithm that learns the optimal action-value function by updating a Q-table based on state-action pairs. This was implemented using a discretized state space and a simple epsilon-greedy policy, with different epsilon schedules.

**Deep Q-Learning**: A variant of Q-Learning that uses a neural network to approximate the Q-function. This approach was implemented using a simple feedforward neural network with a single hidden layer, trained using the DQN algorithm with experience replay and a target network.

## Results

The results and analysis are presented in the report file contained in this repository. In summary:

- **Tabular Q-Learning** achieved sufficient balancing of the CartPole system with a discretized state space. The performance was sensitive to the discretization parameters, the exploration-exploitation trade-off (epsilon schedule), and the initial Q-values.
2. **Deep Q-Learning**: The Deep Q-Learning approach was able to learn a policy that balanced the CartPole system effectively. The neural network architecture, learning rate, and exploration strategy were critical for the performance and stability of the learning process.

## Files
- `Lab2 Report s331998.pdf`: Report file containing the results and analysis
- `qlearning.py`: Tabular Q-Learning implementation
- `deep_qlearning.py`: Deep Q-Learning implementation
- `README.md`: Overview of the project
- `plots/`: Directory containing plots generated during the experiments
- `models/`: Directory containing saved models from the experiments