import torch
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np
from utils import discount_rewards
import sys
import math
import copy

class Policy(torch.nn.Module):
    def __init__(self, state_space, action_space):
        super().__init__()
        self.state_space = state_space
        self.action_space = action_space
        self.hidden = 64
        self.fc1 = torch.nn.Linear(state_space, self.hidden)
        self.fc2_mean = torch.nn.Linear(self.hidden, action_space)

        self.sigmasquared = 5 # std deviation

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if type(m) is torch.nn.Linear:
                torch.nn.init.normal_(m.weight)
                torch.nn.init.zeros_(m.bias)

    def forward(self, x, episode_number=0):
        x = self.fc1(x)
        x = F.relu(x)
        action_mean = self.fc2_mean(x)

        # --- Instantiate and return a normal distribution with mean as network output
        normal_dist = Normal(action_mean, self.sigmasquared)
        return normal_dist

class Agent(object):
    def __init__(self, policy):
        self.train_device = "cpu"
        self.policy = policy.to(self.train_device)
        self.optimizer = torch.optim.Adam(policy.parameters(), lr=5e-3)
        self.gamma = 0.98
        self.states = []
        self.action_probs = []
        self.rewards = []
        self.losses = []
        self.best_policy = None
        self.best_reward = -math.inf

    def episode_finished(self, episode_number):
        action_probs = torch.stack(self.action_probs, dim=0).to(self.train_device).squeeze(-1)
        rewards = torch.stack(self.rewards, dim=0).to(self.train_device).squeeze(-1)

        self.states, self.action_probs, self.rewards = [], [], []

        # T1 a) basic REINFORCE algorithm
        discounted_rewards = discount_rewards(rewards, self.gamma)

        # T1 b)  REINFORCE with a constant baseline b=20
        # discounted_rewards -= 20

        # T1 c) REINFORCE with discounted rewards normalized to zero mean and unit variance
        discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / max(discounted_rewards.std(), 1e-9)

        # Compute the optimization term, loss function to optimize (T1)
        estimated_loss_batch = -action_probs * discounted_rewards # Change sign in order to do gradient ascent instead of descent

        # estimated_loss = torch.sum(estimated_loss_batch)
        # Using the mean is more stable as it results in more consistent and "predictable" update sizes (different episodes have different number of timesteps)
        estimated_loss = torch.mean(estimated_loss_batch)

        # Compute the gradients of loss w.r.t. network parameters (T1)
        self.optimizer.zero_grad()
        estimated_loss.backward()

        # Update network parameters using self.optimizer and zero gradients (T1)
        self.optimizer.step()

        # Save loss and check
        self.losses.append(estimated_loss.item())
        
        # Save the best policy
        current_reward = rewards.sum().item()
        if current_reward > self.best_reward:
            self.best_reward = current_reward
            self.best_policy = copy.deepcopy(self.policy.state_dict())

        return

    def get_action(self, observation, episode_number=0, evaluation=False):
        x = torch.from_numpy(observation).float().to(self.train_device)

        normal_dist = self.policy.forward(x, episode_number=episode_number)

        if evaluation:
            # Return mean if evaluation, else sample from the distribution returned by the policy (T1)
            return normal_dist.mean, None
        else:
            action = normal_dist.sample() # Sample from the distribution (T1)
            action_log_prob = normal_dist.log_prob(action) # Calculate the log probability of the action (T1)

        return action, action_log_prob

    def store_outcome(self, observation, action_prob, action_taken, reward):
        self.states.append(observation)

        # They are log_action probabilities actually
        self.action_probs.append(action_prob)

        self.rewards.append(torch.Tensor([reward]))

