"""
    Robot Learning
    Exercise 1

    Linear Quadratic Regulator

    Polito A-Y 2024-2025
"""
import gym
import numpy as np
from scipy import linalg     # get riccati solver
import argparse
import matplotlib.pyplot as plt
import sys
from utils import get_space_dim, set_seed
import pdb 
import time
import os

# Parse script arguments
def parse_args(args=sys.argv[1:]):
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="CartPole-v0",
                        help="Environment to use")
    parser.add_argument('--seed', default=0, type=int, help='Random seed')
    parser.add_argument("--time_sleep", action='store_true',
                        help="Add timer for visualizing rendering with a slower frame rate")
    parser.add_argument("--mode", type=str, default="control",
                        help="Type of test ['control', 'multiple_R']")
    return parser.parse_args(args)

def linerized_cartpole_system(mp, mk, lp, g=9.81):
    mt=mp+mk
    # state matrix
    # a1 = 0
    a1 = (g*mp)/((mk + mp)*(mp/(mk + mp) - 4/3))
    a2 = - g /(lp*((mp/mt)-4/3))
    
    A = np.array([[0, 1, 0,  0],
                  [0, 0, a1, 0],
                  [0, 0, 0,  1],
                  [0, 0, a2, 0]])

    # input matrix
    # b1 = 1/mt
    b1 = -(mp/(mt*(mp/mt - 4/3)) - 1)/mt
    b2 = 1 / (lp*mt*((mp/mt)-4/3))
    B = np.array([[0], [b1], [0], [b2]])
    
    return A, B

def optimal_controller(A, B, R_value=1):
    R = R_value*np.eye(1, dtype=int)  # choose R (weight for input)
    Q = 5*np.eye(4, dtype=int)        # choose Q (weight for state)
   # solve ricatti equation
    P = linalg.solve_continuous_are(A, B, Q, R)

    # calculate optimal controller gain
    K = np.dot(np.linalg.inv(R),
            np.dot(B.T, P))
    return K

def apply_state_controller(K, x):
    # feedback controller
    u = -np.dot(K, x)   # u = -Kx
    if u > 0:
        return 1, u     # if force_dem > 0 -> move cart right
    else:
        return 0, u     # if force_dem <= 0 -> move cart left
    
    
def plot_forces(forces_history, R_values=[0.01, 0.1, 10, 100]):
    
    # Make all starting forces negative, by changing the sign of all next forces if the first force is positive
    for i in range(len(forces_history)):
        if forces_history[i][0] > 0:
            forces_history[i] = [-force for force in forces_history[i]]
    
    fig, ax = plt.subplots()
    for i, R_value in enumerate(R_values):
        ax.plot(forces_history[i], label=f'R = {R_value}')
    ax.set_xlabel('Timesteps')
    ax.set_ylabel('Force')
    ax.set_title('Force applied to the CartPole for different values of R')
    ax.legend()
    plt.tight_layout()
    plt.grid()
    plt.savefig('plots/cartpole_lqr_forces.png')
    plt.show()

def multiple_R(env, mp, mk, l, g, time_sleep=False, terminate=True):
    """
    Vary the value of R within the range [0.01, 0.1, 10, 100] and plot the forces 
    """
    
    forces_history = []
    for index, R_value in enumerate([0.01, 0.1, 10, 100]):
        
        set_seed(args.seed)    # seed for reproducibility
        env.env.seed(args.seed)
        
        A, B = linerized_cartpole_system(mp, mk, l, g)
        K = optimal_controller(A, B, R_value)    # Re-compute the optimal controller for the current R value
        
        forces_history.append([])
        obs = env.reset()    # Reset the environment for a new episode
        
        for i in range(400):
            env.render()
            if time_sleep:
                time.sleep(.1)
            
            # get force direction (action) and force value (force)
            action, force = apply_state_controller(K, obs)
            forces_history[index].append(force)
            
            # absolute value, since 'action' determines the sign, F_min = -10N, F_max = 10N
            abs_force = abs(float(np.clip(force, -10, 10)))
            
            # change magnitute of the applied force in CartPole
            env.env.force_mag = abs_force

            # apply action
            obs, reward, done, _ = env.step(action)
            
            # if terminate and done:
            #     print(f'Terminated after {i+1} iterations.')
            #     break
        
    plot_forces(forces_history)
    return

def plot_states(history: list):
    
    # Calculate from which timestep the system is considered to be in a stable equilibrium ([-threshold, +threshold])
    threshold = 0.003
    step = 0
    for i in range(len(history)):
        if abs(history[i][0]) < threshold and abs(history[i][1]) < threshold and abs(history[i][2]) < threshold and abs(history[i][3]) < threshold:
            step = i
            break
    
    # plot states: x, x_dot, theta, theta_dot
    history = np.array(history)
    plt.plot(history[:, 0], label=r'$x$')
    plt.plot(history[:, 1], label=r'$\dot x$')
    plt.plot(history[:, 2], label=r'$\theta$')
    plt.plot(history[:, 3], label=r'$\dot \theta$')
    plt.title('States of the CartPole system with LQR controller')
    plt.xlabel('Timesteps')
    plt.ylabel('State values')
    
    # Add 2 horizontal lines at +/- 0.003
    plt.axhline(y=threshold, color='black', linestyle='--', lw=1)
    plt.axhline(y=-threshold, color='black', linestyle='--', lw=1)
    
    # Add vertical line at the timestep where the system is considered to be in a stable equilibrium
    plt.axvline(x=step, color='green', linestyle='--', lw=1, label=f'Stability ({step} steps)')
    
    plt.legend()
    plt.tight_layout()
    plt.savefig('plots/cartpole_lqr_states.png')
    plt.show()

def control(env, mp, mk, l, g, time_sleep=False, terminate=True):
    """
    Control using LQR
    """

    obs = env.reset()    # Reset the environment for a new episode
    
    A, B = linerized_cartpole_system(mp, mk, l, g)
    K = optimal_controller(A, B)    # Re-compute the optimal controller for the current R value
    
    observation_history = [obs]

    for i in range(400):

        env.render()
        if time_sleep:
            time.sleep(.1)
        
        # get force direction (action) and force value (force)
        action, force = apply_state_controller(K, obs)
        
        # absolute value, since 'action' determines the sign, F_min = -10N, F_max = 10N
        abs_force = abs(float(np.clip(force, -10, 10)))
        
        # change magnitute of the applied force in CartPole
        env.env.force_mag = abs_force

        # apply action
        obs, reward, done, _ = env.step(action)
        observation_history.append(obs)
        
        # if terminate and done:
        #     print(f'Terminated after {i+1} iterations.')
        #     break
        
    plot_states(observation_history)

# The main function
def main(args):
    # Create a Gym environment with the argument CartPole-v0 (already embedded in)
    env = gym.make(args.env)

    # Get dimensionalities of actions and observations
    action_space_dim = get_space_dim(env.action_space)
    observation_space_dim = get_space_dim(env.observation_space)

    # Print some stuff
    print("Environment:", args.env)
    print("Observation space dimensions:", observation_space_dim)
    print("Action space dimensions:", action_space_dim)
    
    # Create the subfolder 'plots' if it does not exist
    if not os.path.exists('plots'):
        os.makedirs('plots')

    set_seed(args.seed)    # seed for reproducibility
    env.env.seed(args.seed)
    
    mp, mk, l, g = env.masspole, env.masscart, env.length, env.gravity

    if args.mode == "control":
        control(env, mp, mk, l, g, args.time_sleep, terminate=True)
    elif args.mode == "multiple_R":
        multiple_R(env, mp, mk, l, g, args.time_sleep, terminate=True)

    env.close()

# Entry point of the script
if __name__ == "__main__":
    args = parse_args()
    main(args)

