def policy_eval(policy, env, episodes=1):

    policy.eval()

    for episode in range(episodes):
        obs = env.reset()
        done = False
        ep_reward = 0

        
        while not done:
            action = policy(obs)
            # Select the action with the highest probability
            action = action.argmax().item()
            obs, reward, done, info = env.step(action)

            print(f'Action: {action}, Reward: {reward}, Done: {done}, leach: {info["leach"]}')
            
            ep_reward += reward

        if done:
            print('\n')
            print(f"Episode {episode + 1}/{episodes}, Total Reward: {ep_reward}, "
                  f" Total Leach: {info['episode_metrics']['leach_total']}, Total Yield: {info['episode_metrics']['yield']}, "
                  f"Total Nitrogen: {info['episode_metrics']['N_total']}, Total Water: {info['episode_metrics']['W_total']}")
        

    policy.train()