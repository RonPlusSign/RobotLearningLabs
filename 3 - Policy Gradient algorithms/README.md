# RL2024_Ex3

## Exercise 3 - Policy Gradient Algorithms

This experiment focuses on solving the CartPole balancing problem using policy gradient reinforcement learning techniques. The goal is to directly optimize the policy by maximizing the expected reward, and to compare a custom implementation with one based on the Stable-Baselines3 library.

## Objectives

The main objectives of this experiment are:
1. **Implement Policy Gradient Methods:** Develop a policy gradient agent to learn an optimal control strategy for the CartPole using the **REINFORCE** algorithm.
2. **Leverage Stable-Baselines3:** Utilize the Stable-Baselines3 library for an alternative implementation, providing a higher-level and more robust training framework, using the **PPO** and **SAC** algorithms.
3. **Analyze and Compare:** Evaluate both approaches in terms of training stability, sample efficiency, and overall performance in balancing the CartPole.

## Methods

**Custom Policy Gradient Approach:**  
The custom agent is defined in `agent.py` and executed via `cartpole.py`.
It directly updates the policy parameters based on the observed rewards.
It defines a manual optimization of the policy following the REINFORCE algorithm, including detailed logging of training progress for analysis.

**Stable-Baselines3 Implementation:**  
The script `cartpole_sb3.py` uses the Stable-Baselines3 library to train the CartPole agent with the PPO and SAC algorithms.
This approach leverages pre-built policy gradient algorithms with additional features like experience replay and adaptive learning.
This method improves training stability and efficiency, providing a more robust and scalable solution for reinforcement learning tasks.

## Results

The experimental outcomes and detailed analysis are documented in the provided PDF report (`Lab3 Report s331998.pdf`). The custom policy gradient agent and the Stable-Baselines3 implementations were compared in terms of training performance, sample efficiency, and overall learning dynamics.

Both methods succeeded in learning effective policies for balancing the CartPole, with the Stable-Baselines3 approach yielding smoother training curves and better sample efficiency.

## Files

  - `Lab3 Report s331998.pdf`: Report file containing detailed analysis and results of the experiment
  - `agent.py`: Custom policy gradient agent definition using REINFORCE algorithm
  - `cartpole.py`: Runs the training/testing in the CartPole environment using the custom REINFORCE agent
  - `cartpole_sb3.py`: Trains and tests PPO and SAC algorithms using Stable-Baselines3
  - `cp_cont.py`: Defines the continuous CartPole environment
  - `multiple_cartpoles.py`: Runs batch experiments on multiple CartPole instances
  - `required_python_libs.txt`: List of required Python libraries
  - `checkpoints/`: Saved model checkpoints during training
  - `models/`: Stored trained models
  - `monitor/`: Logs and monitoring outputs
  - `plots/`: Generated plots illustrating training performance and reward progression