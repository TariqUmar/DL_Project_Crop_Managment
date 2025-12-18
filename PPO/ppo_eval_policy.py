
import numpy as np
import torch

def eval_policy(actor_network, env, episodes=10):

    actor_network.eval()

    for episode in range(episodes):
        obs = env.reset()
        done = False
        ep_reward = 0

        
        while not done:
            with torch.no_grad():
                action = actor_network(obs)
            # Select the action with the highest probability
            action = action.argmax().item()
            obs, reward, done, info = env.step(action)

            print(f'Action: {action}, Reward: {info.get("raw_reward", 0)}, Done: {done}, leach: {info.get("leach", 0):.4f}')
            
            ep_reward += info.get("raw_reward", 0)

        if done:
            print('\n')
            metrics = info.get("episode_metrics", {})
            print(f"\nEpisode {episode + 1}/{episodes}, Total Reward={ep_reward:.2f}, "
                f"N_total={metrics.get('N_total', 0):.2f}, W_total={metrics.get('W_total', 0):.2f}, "
                f"Yield={metrics.get('yield', 0):.2f}, Leach={metrics.get('leach_total', 0):.2f}")

    actor_network.train()