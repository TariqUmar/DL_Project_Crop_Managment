from env_wrappers import make_env
from network import FeedForwardNN

import torch
from torch.distributions import Categorical


client_seeds = [1001, 2211, 3120, 4155, 5185]

client_id = 1

models_path = "models/Weights_RF_1_r_50"


env_args = {
    'mode': 'all',
    'seed': client_seeds[client_id - 1],
    'random_weather': True,
    'env_id': 'GymDssatPdi-v0',
    'max_episode_steps': 180,
}

env = make_env(env_args)

# Load the environment normalization statistics
env.load(f'{models_path}/Client_{client_id}_env_normalization_stats.json')

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

policy = FeedForwardNN(state_dim, action_dim)

policy.load_state_dict(torch.load(f'{models_path}/Client_{client_id}_policy.pth'))

obs = env.reset(seed=client_seeds[client_id - 1])
done = False
episode_rew = 0
while not done:
    with torch.no_grad():
        logits = policy(obs)

    dist = Categorical(logits=logits)
    action = dist.sample()

    # Print the distribution
    print(f"Distribution: {dist.probs}")
    print(f"Distribution sum: {dist.probs.sum()}")
    print(f"Action: {action.item()}")

    obs, reward, done, info = env.step(action.item())
    episode_rew += reward

print(f"Episode Reward: {episode_rew}")
print(f"Episode Yield: {info['episode_metrics']['yield']}")
print(f"Episode Leach: {info['episode_metrics']['leach_total']}")
print(f"Episode Nitrogen: {info['episode_metrics']['N_total']}")
print(f"Episode Water: {info['episode_metrics']['W_total']}")

env.close()