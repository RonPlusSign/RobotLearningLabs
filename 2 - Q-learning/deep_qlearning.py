# ------------------------------ Deep Q-learning with PyTorch ------------------------------

import gym
import numpy as np
from matplotlib import pyplot as plt
from time import sleep
import random
import seaborn as sns
import pandas as pd
import sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from qlearning import plot_reward_history

gamma = 0.98 # Discount factor
update_target_period = 10 # Update target network every few episodes
batch_size = 32
epsilon = 1.0 # Initial epsilon value
epsilon_decay = 0.995 # Decay rate for epsilon
epsilon_min = 0.01 # Minimum epsilon value

def parse_args(args=sys.argv[1:]):
    """Parse arguments from command line"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", "-t", type=str, default=None, help="Model to be tested")
    parser.add_argument("--env", type=str, default="CartPole-v0", help="Environment to use")
    parser.add_argument("--train_epochs", type=int, default=1000, help="Number of epochs to train for")
    parser.add_argument("--render_training", action='store_true', help="Render each frame during training. Will be slower.")
    parser.add_argument("--render_test", action='store_true', help="Render test")
    return parser.parse_args(args)

class ExperienceReplay:
    """Experience Replay Memory
    Store the transitions that the agent observes while interacting with the environment
    typle with (state, action, reward, next_state, done)
    """
    
    def __init__(self, capacity: int=10000):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
    
    def add(self, experience: tuple):
        """Add experience to memory"""
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = experience
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size: int):
        """Sample a batch of experiences from memory"""
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)
    

class QNet(nn.Module):
    def __init__(self, input_size, output_size):
        super(QNet, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_size)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    

def get_action(q_net: QNet, env:gym.Env, current_state, eps):
    """Returns the action based on the epsilon-greedy policy"""

    if np.random.rand() < eps: # Random action
        return np.random.choice(range(env.action_space.n)) # choose random action with equal probability among all actions
    else: # Greedy action (ask the network)
        with torch.no_grad(): # Don't compute gradients
            q_values = q_net(torch.tensor(current_state, dtype=torch.float32)) # Forward pass
            return torch.argmax(q_values).item() # Choose the action with the highest Q-value

def test(env: gym.Env, q_net: QNet, test_episodes: int=100, render: bool=True):
    """ Test the given Q-network performance """
    
    reward_history = []
    
    # Test the agent
    for ep in range(test_episodes):
        state, done, steps = env.reset(), False, 0
        total_reward = 0
        while not done:
            action = get_action(q_net, env, state, eps=0) # Choose the best action (greedy)
            state, reward, done, _ = env.step(action)
            if render:
                env.render()
            steps += 1
            total_reward += reward

        print(f"Test episode finished after {steps} timesteps")
        reward_history.append(total_reward)
        env.close()

    print("\nAverage reward over {} test episodes: {:.2f}".format(test_episodes, np.mean(reward_history)))


def train(env: gym.Env, q_net: QNet, episodes: int=2000):
    """ Train the Q-network using Q-learning """
    
    global epsilon
    
    target_q_net = QNet(env.observation_space.shape[0], env.action_space.n) # Initialize target (in: state space, out: action space)
    target_q_net.load_state_dict(q_net.state_dict()) # Copy weights from Q-net to target Q-net
    
    optimizer = optim.Adam(q_net.parameters(), lr=0.0001) # Initialize optimizer
    loss = nn.MSELoss() # Initialize loss function
    experience_buffer = ExperienceReplay() # Initialize experience replay buffer
    
    # Save history
    reward_history = []
    timestep_history = []
    average_reward_history = []
    optimal_action_percentage = []
    average_optimal_action_percentage = []
    ep_lengths, epl_avg = [], []

    # Training loop
    for ep in range(episodes):
        state = env.reset()
        done = False

        # Run episode
        reward_sum, timesteps = 0, 0
        optimal_decisions = 0
        while not done:
            action = get_action(q_net, env, state, eps=epsilon) # Choose action using epsilon-greedy policy
            new_state, reward, done, _ = env.step(action) # Take action
            experience_buffer.add((state, action, reward, new_state, done)) # Store experience
            
            timesteps += 1
            state = new_state
            reward_sum += reward
            
            # Sample a batch of experiences from the buffer
            if len(experience_buffer) < batch_size: # Wait until we have enough experiences in the buffer
                continue
            batch = experience_buffer.sample(batch_size)
            states, actions, rewards, next_states, dones = zip(*batch) # Unpack the batch
            
            # Convert to tensors
            states = torch.tensor(states, dtype=torch.float32)
            actions = torch.tensor(actions, dtype=torch.int64).unsqueeze(1)
            rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1)
            next_states = torch.tensor(next_states, dtype=torch.float32)
            dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1)
            
            # Compute Q-values and targets
            q_values = q_net(states).gather(1, actions) # Q-values for the actions taken
            with torch.no_grad():
                target_q_values = target_q_net(next_states).max(1, keepdim=True)[0] # Q-values for the next states, predicted by the target network
                target_q_values = rewards + gamma * target_q_values * (1 - dones) # Apply Bellman equation: Q(s, a) = r + gamma * max_a' Q(s', a')
                
            # Update the trained Q-network
            loss_value = loss(q_values, target_q_values) # Compute loss
            optimizer.zero_grad()
            loss_value.backward() # Compute gradients
            optimizer.step() # Update weights

            # Check if the action was optimal, and store the percentage of optimal actions
            with torch.no_grad():
                optimal_decisions += 1 if action == np.argmax(q_values) else 0
            
        epsilon = max(epsilon_min, epsilon * epsilon_decay) # Update epsilon
        
        # Update target network after a certain number of episodes
        if ep % update_target_period == 0:
            target_q_net.load_state_dict(q_net.state_dict())
        
        # Bookkeeping (mainly for generating plots)
        ep_lengths.append(timesteps)
        epl_avg.append(np.mean(ep_lengths[max(0, ep-500):])) # Episode length average over the last 500 episodes

        print_interval = 10
        if ep % print_interval == 0:    # Print every few episodes
            print(f"Episode {ep}, epsilon: {epsilon}, avg reward: {reward_sum}")
            
        reward_history.append(reward_sum)
        timestep_history.append(timesteps)
        avg = np.mean(reward_history[-100:]) if ep > 100 else np.mean(reward_history)    
        average_reward_history.append(avg)
        
        optimal_action_percentage.append(100 * optimal_decisions / timesteps)
        average_optimal_action_percentage.append(np.mean(optimal_action_percentage[max(0, ep-100):])) # Average over the last 100 episodes
    
    # Save the model
    torch.save(q_net.state_dict(), 'q_net.pth')
    
    # Plot the reward history
    plot_reward_history(reward_history, average_reward_history)
    
    
def main(args):
    env = gym.make('CartPole-v0')
    env.seed(321)
    # env.spec.max_episode_steps = 500    # Increase the max episode steps (default is 200)
    q_net = QNet(env.observation_space.shape[0], env.action_space.n) # Initialize Q-network (in: state space, out: action space)
    
    if args.test is not None: # Test the model
        q_net.load_state_dict(torch.load(args.test if args.test is not None else 'q_net.pth'))
        test(env, q_net, render=args.render_test)

    else: # Train the model
        input_size = env.observation_space.shape[0]
        output_size = env.action_space.n
        train(env, q_net, episodes=args.train_epochs)
    

if __name__ == "__main__":
    args = parse_args()
    main(args)