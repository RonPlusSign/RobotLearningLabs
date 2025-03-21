import torch
import gym
import numpy as np
import argparse
import matplotlib.pyplot as plt
from agent import Agent, Policy
from cp_cont import CartPoleEnv
import pandas as pd

import sys

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


# Policy training function
def train(env_name, print_things=True, train_run_id=0, train_episodes=1000):
    # Create a Gym environment
    env = gym.make(env_name)

    # Get dimensionalities of actions and observations
    action_space_dim = env.action_space.shape[-1] # it's only 1 now, but it's real valued
    observation_space_dim = env.observation_space.shape[-1]

    print('action_space_dim:', action_space_dim)
    print('observation_space_dim:', observation_space_dim)

    # Instantiate agent and its policy
    policy = Policy(observation_space_dim, action_space_dim)
    agent = Agent(policy)

    # Arrays to keep track of rewards
    reward_history, timestep_history = [], []
    average_reward_history = []

    # Run actual training
    for episode_number in range(train_episodes):
        reward_sum, timesteps = 0, 0
        done = False
        # Reset the environment and observe the initial state
        observation = env.reset()

        # Loop until the episode is over
        while not done:
            # Get action from the agent
            action, action_probabilities = agent.get_action(observation, episode_number=episode_number)
            previous_observation = observation

            # Perform the action on the environment, get new state and reward
            observation, reward, done, info = env.step(action.detach().numpy())

            # Store action's outcome (so that the agent can improve its policy)
            agent.store_outcome(previous_observation, action_probabilities, action, reward)

            # Store total episode reward
            reward_sum += reward
            timesteps += 1

        if print_things and episode_number % 100 == 0:
            last_loss = agent.losses[-1] if len(agent.losses) > 0 else np.nan
            print("Episode {} finished. Total reward: {:.3g}, loss: {:-3g} ({} timesteps)"
                  .format(episode_number, reward_sum, last_loss, timesteps))

            print('Current sigma squared:', agent.policy.sigmasquared)

        # Bookkeeping (mainly for generating plots)
        reward_history.append(reward_sum)
        timestep_history.append(timesteps)
        if episode_number > 100:
            avg = np.mean(reward_history[-100:])
        else:
            avg = np.mean(reward_history)
        average_reward_history.append(avg)

        # Let the agent do its magic (update the policy)
        agent.episode_finished(episode_number)

    # Training is finished - plot rewards
    if print_things:
        # Plot reward history
        plt.plot(reward_history)
        plt.plot(average_reward_history)
        plt.legend(["Reward", "100-episode average"])

        plt.xlabel('episode', labelpad=12, fontweight='bold')
        plt.ylabel('cumulative reward', labelpad=12, fontweight='bold')
        plt.title("Reward history with REINFORCE")
        plt.tight_layout()
        plt.savefig("reward_history_%s_%d.png" % (env_name, train_run_id))
        plt.show()
        plt.close()
        
        # Plot loss history
        plt.plot(agent.losses)
        plt.xlabel('episode', labelpad=12, fontweight='bold')
        plt.ylabel('loss', labelpad=12, fontweight='bold')
        plt.title("Loss history with REINFORCE")
        plt.tight_layout()
        plt.savefig("loss_history_%s_%d.png" % (env_name, train_run_id))
        plt.show()
        plt.close()
        
        print("Training finished.")

    data = pd.DataFrame({"episode": np.arange(len(reward_history)),
                         "train_run_id": [train_run_id]*len(reward_history),
                         # Change algorithm name for plots, if you want
                         "algorithm": ["PG"]*len(reward_history),
                         "reward": reward_history})
    torch.save(agent.best_policy, "model_%s_%d.mdl" % (env_name, train_run_id))
    return data


# Function to test a trained policy
def test(env_name, episodes, params, render=True):
    # Create a Gym environment
    env = gym.make(env_name)

    # Get dimensionalities of actions and observations
    action_space_dim = env.action_space.shape[-1]
    observation_space_dim = env.observation_space.shape[-1]

    # Instantiate agent and its policy
    policy = Policy(observation_space_dim, action_space_dim)
    policy.load_state_dict(params)
    agent = Agent(policy)

    test_reward, test_len = 0, 0
    for ep in range(episodes):
        done = False
        observation = env.reset()
        while not done:
            # Similar to the training loop above -
            # get the action, act on the environment, save total reward
            # (evaluation=True makes the agent always return what it thinks to be
            # the best action - there is no exploration at this point)
            action, _ = agent.get_action(observation, evaluation=True)
            observation, reward, done, info = env.step(action.detach().cpu().numpy())

            if render:
                env.render()
            test_reward += reward
            test_len += 1
    print("Average test reward:", test_reward/episodes, "episode length:", test_len/episodes)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", "-t", type=str, default=None, help="Model to be tested")
    parser.add_argument("--env", type=str, default="ContinuousCartPole-v0", help="Environment to use")
    parser.add_argument("--train_episodes", type=int, default=1000, help="Number of episodes to train for")
    parser.add_argument("--render_test", action='store_true', help="Render test")
    args = parser.parse_args()

    # If no model was passed, train a policy from scratch.
    # Otherwise load the policy from the file and go directly to testing.
    if args.test is None:
        try:
            train(args.env, train_episodes=args.train_episodes)
        # Handle Ctrl+C - save model and go to tests
        except KeyboardInterrupt:
            print("Interrupted!")
    else:
        state_dict = torch.load(args.test)
        print("Testing...")
        test(args.env, 100, state_dict, args.render_test)

