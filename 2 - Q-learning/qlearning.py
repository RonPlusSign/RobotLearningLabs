# ------------------------------ Q-learning ------------------------------

import gym
import numpy as np
from matplotlib import pyplot as plt
from time import sleep
import random
import seaborn as sns
import pandas as pd
import sys
import argparse

# Parameters
gamma = 0.98
alpha = 0.1
constant_eps = 0.2
b = 2222.22 # Choose b so that with GLIE we get an epsilon of 0.1 after 20'000 episodes (2000/0.9)
num_of_actions = 2  # 2 discrete actions for Cartpole

# Reasonable values for Cartpole discretization
discr = 16
x_min, x_max = -2.4, 2.4
v_min, v_max = -3, 3
th_min, th_max = -0.3, 0.3
av_min, av_max = -4, 4

# Create discretization grid
x_grid = np.linspace(x_min, x_max, discr)
v_grid = np.linspace(v_min, v_max, discr)
th_grid = np.linspace(th_min, th_max, discr)
av_grid = np.linspace(av_min, av_max, discr)


def parse_args(args=sys.argv[1:]):
    """Parse arguments from command line"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", "-t", type=str, default=None, help="Model to be tested")
    parser.add_argument("--env", type=str, default="CartPole-v0", help="Environment to use")
    parser.add_argument("--train_episodes", type=int, default=20000, help="Number of episodes to train for")
    parser.add_argument("--render_training", action='store_true', help="Render each frame during training. Will be slower.")
    parser.add_argument("--render_test", action='store_true', help="Render test")
    parser.add_argument("--eps_type", type=str, default="GLIE", help="Type of epsilon to use: constant, GLIE, greedy")
    return parser.parse_args(args)


def find_nearest(array, value):
    """Find nearest value in array"""
    return np.argmin(np.abs(array - value))

def get_cell_index(state):
    """Returns discrete state from continuous state"""
    x = find_nearest(x_grid, state[0])
    v = find_nearest(v_grid, state[1])
    th = find_nearest(th_grid, state[2])
    av = find_nearest(av_grid, state[3])
    return x, v, th, av


def get_action(state, q_values, epsilon=constant_eps, greedy=False):
    """Returns the action based on the greedy or epsilon-greedy policy"""
    x, v, th, av = get_cell_index(state)

    if greedy: # TEST -> greedy policy
        return np.argmax(q_values[x, v, th, av, :]) # greedy w.r.t. q_grid

    else: # TRAINING -> epsilon-greedy policy
        if np.random.rand() < epsilon: # Random action
            return np.random.choice(num_of_actions) # choose random action with equal probability among all actions
        else: # Greedy action
            return np.argmax(q_values[x, v, th, av, :]) # greedy w.r.t. q_grid


def update_q_value(old_state, action, new_state, reward, done, q_array):
    old_cell_index = get_cell_index(old_state)
    new_cell_index = get_cell_index(new_state)

    # Target value used for updating our current Q-function estimate at Q(old_state, action)
    if done is True:
        target_value = reward  # HINT: if the episode is finished, there is not next_state. Hence, the target value is simply the current reward.
    else:
        # HINT: if the episode is not finished, the target value is the sum of the current reward and the discounted maximum Q-value for the next state.
        target_value = reward + gamma * np.max(q_array[new_cell_index[0], new_cell_index[1], new_cell_index[2], new_cell_index[3], :]) 

    # Update Q value
    q_array[old_cell_index[0], old_cell_index[1], old_cell_index[2], old_cell_index[3], action] = (1 - alpha) * q_array[old_cell_index[0], old_cell_index[1], old_cell_index[2], old_cell_index[3], action] + alpha * target_value
    return q_array
    
def compute_value_function(q_grid):
    # Compute the value function by taking the max Q-value for each state
    value_function = np.max(q_grid, axis=-1)
    return value_function

def plot_value_function_heatmap(value_function, fig_name: str = 'value_function_heatmap.png'):
    # Assuming the value function is 4D, we need to reduce it to 2D for plotting
    # Here we average over two of the dimensions for simplicity
    value_function_2d = np.mean(value_function, axis=(1,3))
    
    plt.figure(figsize=(10, 8))
    plt.imshow(value_function_2d, cmap='viridis', interpolation='none', extent=[th_min, th_max, x_min, x_max, th_min], aspect='auto')
    plt.colorbar(label='State Value function')
    # plt.clim(0, 5) # color range from 0 to 5
    plt.title(r"State Value function heatmap (averaged over velocity $\dot x$ and angular velocity $\dot \theta$)")
    plt.xlabel(r"CartPole angle $\theta$")
    plt.ylabel(r"CartPole position $x$")
    plt.xticks(np.round(th_grid, 2))
    plt.yticks(np.round(x_grid, 2))
    plt.tight_layout()
    plt.savefig(fig_name)
    plt.show()

def plot_reward_history(reward_history, average_reward_history):
    # Store the data in a Pandas dataframe for easy visualization
    training_history = pd.DataFrame({"episode": np.arange(len(reward_history)), "reward": reward_history, "mean_reward": average_reward_history})

    # Plot rewards
    sns.lineplot(x="episode", y="reward", data=training_history, color='blue', label='Reward')
    sns.lineplot(x="episode", y="mean_reward", data=training_history, color='orange', label='100-episode average')
    plt.legend()
    plt.title("Reward history (CartPole-v0)")
    plt.savefig('reward_history_CartPole.png')
    plt.tight_layout()
    plt.show()
    plt.close()

def plot_avg_optimal_action_percentage(average_optimal_action_percentage, label: str):
    plt.plot(average_optimal_action_percentage, label=label)
    plt.title("Average greedy action percentage")
    plt.xlabel("Episodes")
    plt.ylabel("Percentage")
    plt.legend()
    plt.tight_layout()
    plt.savefig('optimal_action_percentage.png')
    plt.show()
    plt.close()

def training(env: gym.Env, q_grid, epsilon_type: str = 'constant', plot_heatmaps: bool = False, episodes: int = 20000, render_training: bool = False):
    
    reward_history = []
    timestep_history = []
    average_reward_history = []
    optimal_action_percentage = []
    average_optimal_action_percentage = []

    # Training loop
    ep_lengths, epl_avg = [], []
    for ep in range(episodes):
        state = env.reset()
        done = False

        # Define epsilon
        if epsilon_type == 'constant':  # Constant epsilon
            epsilon = constant_eps
        elif epsilon_type == 'GLIE':
            epsilon = b / (b + ep) # Change epsilon following GLIE schedule (task 3.1) 
        else:
            epsilon = 0 # Change epsilon to 0 to test the greedy policy
        
        # Save heatmap of state value function at episode 0, 1, 10000
        if plot_heatmaps and ep in [0, 1, episodes/2]:
            value_function = compute_value_function(q_grid)
            plot_value_function_heatmap(value_function, fig_name=f'value_function_heatmap_constant_eps_{ep}_episodes.png')

        # Run episode
        reward_sum, timesteps = 0, 0
        optimal_decisions = 0
        while not done:
            action = get_action(state, q_grid, epsilon)
            new_state, reward, done, _ = env.step(action)
            
            q_grid = update_q_value(state, action, new_state, reward, done, q_grid) # Update Q-value matrix

            # Check if the action was optimal, and store the percentage of optimal actions
            optimal_decisions += action == np.argmax(q_grid[get_cell_index(state)])
            
            timesteps += 1
            state = new_state
            reward_sum += reward


        # Bookkeeping (mainly for generating plots)
        ep_lengths.append(timesteps)
        epl_avg.append(np.mean(ep_lengths[max(0, ep-500):])) # Episode length average over the last 500 episodes
        if ep % 200 == 0:
            print("Episode {}, average timesteps: {:.2f}".format(ep, np.mean(ep_lengths[max(0, ep-200):])))
            print('Epsilon:', epsilon)
            
        reward_history.append(reward_sum)
        timestep_history.append(timesteps)
        avg = np.mean(reward_history[-100:]) if ep > 100 else np.mean(reward_history)    
        average_reward_history.append(avg)
        
        optimal_action_percentage.append(100 * optimal_decisions / timesteps)
        average_optimal_action_percentage.append(np.mean(optimal_action_percentage[max(0, ep-100):])) # Average over the last 100 episodes

    # Save the Q-values
    np.save("q_values.npy", q_grid)

    # Plot the reward history
    plot_reward_history(reward_history, average_reward_history)

    # Plot the avg optimal action percentage
    plot_avg_optimal_action_percentage(average_optimal_action_percentage, label='Q-learning')


def test(env, q_grid, test_episodes=100, render=False):
    
    total_length = 0
    for ep in range(test_episodes):
        # Test the agent
        state, done, steps = env.reset(), False, 0
        while not done:
            action = get_action(state, q_grid, greedy=True)
            state, _, done, _ = env.step(action)
            if render:
                env.render()
            steps += 1
        print("Test episode finished after {} timesteps".format(steps))
        env.close()
        total_length += steps
    print("Average episode length over {} test episodes: {:.2f}".format(test_episodes, total_length / test_episodes))
    
    
    # Calculate the value function using the q_values and the greedy policy (max Q-value for all actions)
    value_function = compute_value_function(q_grid)
    
    # Plot the value function heatmap
    plot_value_function_heatmap(value_function, fig_name='value_function_heatmap_test.png')
        
def main(args):   
    np.random.seed(123)
    env = gym.make(args.env)
    env.seed(321)
    
    if args.test is not None:
        q_grid = np.load(args.test)
        test(env, q_grid, render=args.render_test)
    else:
        # Initialize Q values
        q_grid = np.zeros((discr, discr, discr, discr, num_of_actions)) # Initial values to 0
        # q_grid = np.full((discr, discr, discr, discr, num_of_actions), 50) # Initial values to 50
        
        training(env, q_grid, epsilon_type=args.eps_type, plot_heatmaps=True, episodes=args.train_episodes)

    
if __name__ == "__main__":
    args = parse_args()
    main(args)
    