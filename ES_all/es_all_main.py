import gym
import gym_dssat_pdi
import sys
import torch
import numpy as np
import os

from es_algo import ES_Algo
from network import FeedForwardNN
from env_wrappers import make_env
from es_all_policy_eval import policy_eval

# Paths
LOG_DIR = 'logs/RF4'
SAVE_DIR = 'models/RF4'
POLICY_PATH = os.path.join(SAVE_DIR, 'policy.pth')
STATS_PATH = os.path.join(SAVE_DIR, 'normalization_stats.json')

# Config
Total_Iterations = 100
SEED = 123
MODE = 'train'

def train(env, policy):

    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)
    else:
        print(f"Log path {LOG_DIR} already exists.", flush=True)
        exit(0)

    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)
    else:
        print(f"Save path {SAVE_DIR} already exists.", flush=True)
        exit(0)

    if hasattr(env, 'training'):
        env.training = True
    if hasattr(env, 'norm_rew'):
        env.norm_rew = False
    if hasattr(env, 'norm_obs'):
        env.norm_obs = True

    # Create a model for ES
    model = ES_Algo(env, policy)
    print(f"Training from scratch...", flush=True)

    model.train(log_path=LOG_DIR, save_path=SAVE_DIR, iterations=Total_Iterations)

    if hasattr(env, 'save'):
        env.save(STATS_PATH)
        print(f"Successfully saved normalization statistics.", flush=True)

def test(env, policy):

    if not os.path.exists(POLICY_PATH):
        print(f"Actor model not found. Exiting.", flush=True)
        sys.exit(1)

    # Load stats and freeze normalization
    if hasattr(env, 'load') and os.path.exists(STATS_PATH):
        env.load(STATS_PATH)
        print(f"Successfully loaded normalization statistics from {STATS_PATH}.", flush=True)

    if hasattr(env, 'training'):
        env.training = False
    if hasattr(env, 'norm_rew'):
        env.norm_rew = False

    # Load the trained policy
    policy.load_state_dict(torch.load(POLICY_PATH))

    policy_eval(policy=policy, env=env, episodes=1)

def main():

    environment_arguments = {
    'mode': 'all',
    'seed': 123,
    'random_weather': False,
    'env_id': 'GymDssatPdi-v0',
    'max_episode_steps': 180,
}

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    env = make_env(environment_arguments)

    # Get the dimensions of the observation and action spaces
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    # Create a model for the policy
    policy = FeedForwardNN(state_dim, action_dim)


    if MODE == 'train':
        train(env, policy)

    elif MODE == 'test':
        test(env, policy)
    else:
        print(f"Invalid mode. Exiting.", flush=True)

    env.close()

if __name__ == "__main__":
    main()