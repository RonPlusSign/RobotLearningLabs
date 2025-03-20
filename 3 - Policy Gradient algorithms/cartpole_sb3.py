import torch
import gym
import numpy as np
import argparse
import matplotlib.pyplot as plt
from agent import Agent, Policy
from cp_cont import CartPoleEnv
import pandas as pd
import time

import glob
import sys
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

from stable_baselines3 import PPO, SAC
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback

from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy

def set_seed(seed):
    if seed > 0:
        np.random.seed(seed)

def create_model(args, env):
    # T4 TODO
    if args.algo == 'ppo':
        model = PPO("MlpPolicy", env, learning_rate=args.lr, verbose=1, n_steps=2048, batch_size=64, n_epochs=10, gae_lambda=0.95, clip_range=0.2)

    elif args.algo == 'sac':
        model = SAC("MlpPolicy", env, learning_rate=args.lr, verbose=1, gradient_steps=args.gradient_steps, batch_size=64, buffer_size=1000000, learning_starts=100, train_freq=1, tau=0.005, gamma=0.99, ent_coef='auto', target_update_interval=1)
    else:
        raise ValueError(f"RL Algo not supported: {args.algo}")
    return model
        
def load_model(args, env):
    # T4
    if args.algo == 'ppo':
        model = PPO.load(args.test)
    elif args.algo == 'sac':
        model = SAC.load(args.test)
    else:
        raise ValueError(f"RL Algo not supported: {args.algo}")
    return model

def moving_average(values, window):
    """
    Smooth values by doing a moving average
    :param values: (numpy array)
    :param window: (int)
    :return: (numpy array)
    """
    weights = np.repeat(1.0, window) / window
    return np.convolve(values, weights, "valid")


def plot_results(monitor_folder, title="Learning Curve"):
    """
    plot the results

    :param log_folder: (str) the save location of the results to plot
    :param title: (str) the title of the task to plot
    """
    x, y = ts2xy(load_results(monitor_folder), "timesteps")
    y = moving_average(y, window=50)
    # Truncate x
    x = x[len(x) - len(y) :]

    fig = plt.figure(title)
    plt.plot(x, y)
    plt.xlabel("Number of Timesteps")
    plt.ylabel("Rewards")
    plt.title(title + " Smoothed")
    plt.savefig(f"reward_history {title}.png")
    plt.show()
    
def load_latest_checkpoint(log_dir, algo, env):
    checkpoint_files = glob.glob(os.path.join(log_dir, f"rl_model_{algo}_*.zip"))
    if not checkpoint_files:
        return None
    latest_checkpoint = max(checkpoint_files, key=os.path.getctime)
    if algo == 'ppo':
        model = PPO.load(latest_checkpoint, env) 
    elif algo == 'sac':
        model = SAC.load(latest_checkpoint, env)
    else:
        raise ValueError(f"RL Algo not supported: {algo}")
    
    print(f"Checkpoint found: {latest_checkpoint} with {model.num_timesteps} timesteps, resuming training...")
    return model
    
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", "-t", type=str, default=None, help="Model to be tested")
    parser.add_argument("--env", type=str, default="ContinuousCartPole-v0", help="Environment to use")
    parser.add_argument("--total_timesteps", type=int, default=25000, help="The total number of samples to train on")
    parser.add_argument("--render_test", action='store_true', help="Render test")
    parser.add_argument('--seed', default=0, type=int, help='Random seed')
    parser.add_argument('--algo', default='ppo', type=str, help='RL Algo [ppo, sac]')
    parser.add_argument('--lr', default=0.0003, type=float, help='Learning rate')
    parser.add_argument('--gradient_steps', default=-1, type=int, help='Number of gradient steps when policy is updated in sb3 using SAC. -1 means as many as --args.now')
    parser.add_argument('--test_episodes', default=100, type=int, help='# episodes for test evaluations')
    return parser.parse_args()

def train_model(model, args, env, log_dir, monitor_dir):
    try:
        # Create evaluation environment
        eval_env = gym.make(args.env)
        eval_env = Monitor(eval_env, monitor_dir)
        
        # Create EvalCallback
        eval_callback = EvalCallback(eval_env, best_model_save_path=log_dir, log_path=log_dir, eval_freq=10000, deterministic=True, render=False)

         # Create CheckpointCallback
        checkpoint_callback = CheckpointCallback(save_freq=10000, save_path=log_dir, name_prefix=f"rl_model_{args.algo}")

        # Policy training (T4)
        start = time.time()
        model.learn(total_timesteps=args.total_timesteps, callback=[eval_callback, checkpoint_callback])
        print(f"Training time: {time.time() - start} seconds")
        
        # Saving model (T4)
        model.save(f"{args.algo}_model")
        plot_results(monitor_dir)
        
    # Handle Ctrl+C - save model and go to tests
    except KeyboardInterrupt:
        print("Interrupted!")


def test_model(model, env, args):
    print("Testing...")
    # Policy evaluation (T4)
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=args.test_episodes)
    print(f"Test reward (avg +/- std): ({mean_reward} +/- {std_reward}) - Num episodes: {args.test_episodes}")

if __name__ == "__main__":
    args = parse_args()
    set_seed(args.seed)

    env = gym.make(args.env)

    log_dir = "./checkpoints/"
    monitor_dir = "./monitor/"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(monitor_dir, exist_ok=True)
    env = Monitor(env, monitor_dir) # Logs will be saved in log_dir/monitor.csv

    # If no model was passed, train a policy from scratch.
    # Otherwise load the model from the file and go directly to testing.
    if args.test is None: # Train
        # Check for the latest checkpoint
        model = load_latest_checkpoint(log_dir, args.algo, env)
        if model is None:
            model = create_model(args, env)
        train_model(model, args, env, log_dir, monitor_dir)

    else: # Test
        model = load_model(args, env)
        test_model(model, env, args)

    env.close()

