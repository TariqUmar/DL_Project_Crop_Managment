import gym
import gym_dssat_pdi
import sys
import torch
import numpy as np
import random
import os

from ppo import PPO
from network import FeedForwardNN
from ppo_eval_policy import eval_policy
from env_wrappers import make_env

# Paths
LOG_DIR = 'logs/RF5'
SAVE_DIR = 'models/RF5'
ACTOR_PATH = os.path.join(SAVE_DIR, 'actor.pth')
CRITIC_PATH = os.path.join(SAVE_DIR, 'critic.pth')
STATS_PATH = os.path.join(SAVE_DIR, 'normalization_stats.json')

# Config
Total_Episodes = 2000
SEED = 123
MODE = 'train'  # 'train' or 'test'


def train(env):

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
        env.norm_rew = True

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    model = PPO(state_dim, action_dim, env, FeedForwardNN)
    print(f"Training  from scratch...", flush=True)

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    model.learn(total_episodes=Total_Episodes, log_path=LOG_DIR, save_path=SAVE_DIR)

    if hasattr(env, 'save'):
        env.save(STATS_PATH)
        print(f"Successfully saved normalization statistics.", flush=True)

def test(env):

    if not os.path.exists(ACTOR_PATH):
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

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    actor_network = FeedForwardNN(state_dim, action_dim)
    actor_network.load_state_dict(torch.load(ACTOR_PATH))
    
    eval_policy(actor_network=actor_network, env=env, episodes=1)
      

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

    if MODE == "train":
        train(env)
    elif MODE == "test":
        test(env)
    else:
        print("Invalid MODE, must be 'train' or 'test'.")

    env.close()

if __name__ == "__main__":

    main()