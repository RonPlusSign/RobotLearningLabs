"""
    Robot Learning
    Exercise 1

    Reinforcement Learning 

    Polito A-Y 2024-2025
"""
import torch
import gym
import numpy as np
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import sys
from agent import Agent, Policy
from utils import get_space_dim

import sys


# Parse script arguments
def parse_args(args=sys.argv[1:]):
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", "-t", type=str, default=None,
                        help="Model to be tested")
    parser.add_argument("--env", type=str, default="CartPole-v0",
                        help="Environment to use")
    parser.add_argument("--train_episodes", type=int, default=500,
                        help="Number of episodes to train for")
    parser.add_argument("--render_training", action='store_true',
                        help="Render each frame during training. Will be slower.")
    parser.add_argument("--render_test", action='store_true', help="Render test")
    parser.add_argument("--central_point", type=float, default=0.0,
                        help="Point x0 to fluctuate around")
    parser.add_argument("--random_policy", action='store_true', help="Applying a random policy training")
    return parser.parse_args(args)


# Policy training function
def train(agent, env, train_episodes, early_stop=True, render=False, silent=False, train_run_id=0, x0=0, random_policy=False):
    
    # Arrays to keep track of rewards
    reward_history, timestep_history = [], []
    average_reward_history = []

    # Run actual training
    for episode_number in range(train_episodes):
        reward_sum, timesteps = 0, 0
        done = False
        # Reset the environment and observe the initial state (it's a random initial state with small values)
        observation = env.reset()

        # Loop until the episode is over
        while not done:
            # Get action from the agent
            action, action_probabilities = agent.get_action(observation)

            if random_policy:
                # Task 1.1
                """
                Sample a random action from the action space
                """
                action = np.random.choice(env.action_space.n)

            previous_observation = observation

            # Perform the action on the environment, get new state and reward
            # note that after env._max_episode_steps the episode is over, if we stay alive that long
            observation, reward, done, info = env.step(action)

            # Task 3.1
            """
                Use a different reward, overwriting the original one
            """
            # reward = reward_stable_in_position(observation, x0)
            reward = reward_oscillation(previous_observation, observation)
            # reward = reward_oscillation_v2(observation)

            # Store action's outcome (so that the agent can improve its policy)
            agent.store_outcome(previous_observation, action_probabilities, action, reward)

            # Draw the frame, if desired
            if render:
                env.render()

            # Store total episode reward
            reward_sum += reward
            timesteps += 1

        if not silent:
            print("Episode {} finished. Total reward: {:.3g} ({} timesteps)"
                  .format(episode_number, reward_sum, timesteps))

        # Bookkeeping (mainly for generating plots)
        reward_history.append(reward_sum)
        timestep_history.append(timesteps)
        if episode_number > 100:
            avg = np.mean(reward_history[-100:])
        else:
            avg = np.mean(reward_history)
        average_reward_history.append(avg)

        # Early stopping: stop training if the agent has learned the task well enough
        if early_stop and (
            np.mean(reward_history[-15:]) >= env._max_episode_steps * 0.75 and # Good reward
            np.mean(timestep_history[-15:]) == env._max_episode_steps  and  # We need to reach the end of the episode
            episode_number > train_episodes/2  # We need to have trained for a while
        ):
            if not silent:
                print("Looks like it's learned. Finishing up early")
            break

        # Let the agent do its magic (update the policy)
        agent.episode_finished(episode_number)

    # Store the data in a Pandas dataframe for easy visualization
    data = pd.DataFrame({"episode": np.arange(len(reward_history)),
                         "train_run_id": [train_run_id]*len(reward_history),
                         "reward": reward_history,
                         "mean_reward": average_reward_history})
    return data


# Function to test a trained policy
def test(agent, env, episodes, render=False, x0=0):
    test_reward, test_len = 0, 0

    episodes = 100
    print('Num testing episodes:', episodes)
    
    max_speed = 0

    for ep in range(episodes):
        done = False
        observation = env.reset()
        while not done:
        # Task 1.2
            """
            Test on 500 timesteps
            """
            action, _ = agent.get_action(observation, evaluation=True)  # Similar to the training loop above -
                                                                        # get the action, act on the environment, save total reward
                                                                        # (evaluation=True makes the agent always return what it thinks to be
                                                                        # the best action - there is no exploration at this point)
            previous_observation = observation
            observation, reward, done, info = env.step(action)
            max_speed = max(max_speed, abs(observation[1])) # Keep track of the maximum speed reached

            # Task 3.1
            """
                Use a different reward, overwriting the original one
            """
            # reward = reward_stable_in_position(observation, x0)
            reward = reward_oscillation(previous_observation, observation)
            # reward = reward_oscillation_v2(observation)

            if render:
                env.render()
            test_reward += reward
            test_len += 1
    print("Average test reward:", test_reward/episodes, ", episode length:", test_len/episodes, ", max speed:", max_speed)
    
    
def reward_oscillation(previous_state, state):
    """
    Reward function to encourage oscillation (left-right movement) while keeping the pole balanced.

    Args:
        previous_state: A tuple containing the previous (position, position_velocity, angle, angle_velocity).
        state: A tuple containing the current (position, position_velocity, angle, angle_velocity).

    Returns: reward: A scalar reward.
    """
    
    prev_position, prev_position_velocity, prev_angle, prev_angle_velocity = previous_state
    position, position_velocity, angle, angle_velocity = state
    
    angle_penalty = abs(angle) # Penalize large pole angles
    if abs(angle) > 0.1:
        angle_penalty *= 2  # Apply a stronger penalty for large deviations from balance
    
    # Reward for oscillation: Encourage movement from left to right with enough speed
    if (prev_position_velocity > 0 and position_velocity < 0) or (prev_position_velocity < 0 and position_velocity > 0):
        direction_change_reward = 1.0  # Reward for changing direction
    else:
        direction_change_reward = 0.1
    
    speed_reward = abs(position_velocity) # Speed reward: The faster the better
    
    edge_penalty = 0 # Penalize being too close to the edge (-2.4, 2.4).
    if abs(position) > 2.0:  # Penalize if position is within 0.4 units of the edge
        edge_penalty = 3.0 * (abs(position) - 2.0)  # Penalize more as the cart gets closer to the edge
    
    return -angle_penalty + direction_change_reward + speed_reward - edge_penalty


def reward_oscillation_v2(state):
    """
    Reward function to encourage oscillation (left-right movement) while keeping the pole balanced.
    Args: state: A tuple containing the current (position, position_velocity, angle, angle_velocity).
    Returns: reward: A scalar reward.
    """
    return 1 + abs(state[1])    # Reward for high speed



def reward_stable_in_position(state, x0):
    # Balance the pole close to the given point x0
    # The closer the pole is to x0, the higher the reward
    pos, pos_velocity, angle, angle_velocity = state
    return 1 - 0.5*abs(pos-x0) -abs(angle) - 0.1*abs(pos_velocity)    # Reward is 1 if the pole is in x0 and straight, less otherwise
    
    
# The main function
def main(args):
    # Create a Gym environment with the argument CartPole-v0 (already embedded in)
    env = gym.make(args.env)

    # Task 1.2
    """
    # For CartPole-v0 - change the maximum episode length
    """
    env._max_episode_steps = 1000

    # Get dimensionalities of actions and observations
    action_space_dim = get_space_dim(env.action_space)
    observation_space_dim = get_space_dim(env.observation_space)

    # Instantiate agent and its policy
    policy = Policy(observation_space_dim, action_space_dim)
    agent = Agent(policy)

    # Print some stuff
    print("Environment:", args.env)
    print("Training device:", agent.train_device)
    print("Observation space dimensions:", observation_space_dim)
    print("Action space dimensions:", action_space_dim)

    # If no model was passed, train a policy from scratch.
    # Otherwise load the policy from the file and go directly to testing.
    if args.test is None:
        # Train
        training_history = train(agent, env, args.train_episodes, True, args.render_training, x0=args.central_point, random_policy=args.random_policy)

        # Save the model
        model_file = "%s_params.ai" % args.env
        torch.save(policy.state_dict(), model_file)
        print("Model saved to", model_file)

        # Plot rewards
        sns.lineplot(x="episode", y="reward", data=training_history, color='blue', label='Reward')
        sns.lineplot(x="episode", y="mean_reward", data=training_history, color='orange', label='100-episode average')
        plt.legend()
        plt.title("Reward history (%s)" % args.env)
        plt.savefig('reward_history_CartPole_rl.png')
        plt.show()
        print("Training finished.")
    else:
        # Test
        print("Loading model from", args.test, "...")
        state_dict = torch.load(args.test)
        policy.load_state_dict(state_dict)
        print("Testing...")
        test(agent, env, args.train_episodes, args.render_test)


# Entry point of the script
if __name__ == "__main__":
    args = parse_args()
    main(args)

