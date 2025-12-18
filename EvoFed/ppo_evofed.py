from torch.distributions import Categorical
import torch
import numpy as np
from torch.optim import Adam
import torch.nn as nn
import time
import gym
import random

from torch.utils.tensorboard import SummaryWriter
from env_wrappers import make_env

from network import FeedForwardNN

class PPO:
    def __init__(self, state_dim: int, action_dim: int, env: gym.Env, client_id: int):

        # Initialize the hyperparameters
        self.episodes_per_batch = 20                                    # episodes per batch for training
        self.max_timesteps_per_episode = 180                            # max timesteps per episode
        self.gamma = 0.99                                               # discount factor for future rewards
        self.mini_batch_size = 5 * self.max_timesteps_per_episode       # number of episodes per mini batch
        self.n_updates_per_batch = 5                                    # number of updates per batch
        self.clip = 0.2                                                 # clip parameter for PPO
        self.actor_lr = 3e-4                                            # learning rate for the actor network
        self.critic_lr = 3e-4                                           # learning rate for the critic network
        self.ent_coef = 0.0                                             # entropy coefficient for the actor network
        self.vf_coef = 0.5                                              # value function coefficient for the critic network
        self.kl_threshold = 0.05                                        # The threshold for the KL divergence

        # Get the environment and the dimensions of the observation and action spaces
        self.env = env
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.client_id = client_id

        # ALG - Step 1:
        # Initialize the actor and critic networks
        self.actor = FeedForwardNN(self.state_dim, self.action_dim)
        self.critic = FeedForwardNN(self.state_dim, 1)

        # Initialize the actor optimizer
        self.actor_optimizer = None

        # Initialize the critic optimizer
        self.critic_optimizer = None

        self.total_episodes = 0

    def get_action(self, obs):
        with torch.no_grad():
            # Query the actor network for a action
            # Same thing as calling self.actor.forward(obs)
            obs = torch.as_tensor(obs, dtype=torch.float32)
            logits = self.actor(obs)

            # Create a categorical distribution
            dist = Categorical(logits=logits)

            # Sample an action from the distribution and get its log probability
            action = dist.sample()
            log_prob = dist.log_prob(action)

            # Return the sampled action and the log prob of that action
            # Note that I'm calling detach() since the action and log_prob  
            # are tensors with computation graphs, so I want to get rid
            # of the graph and just convert the action to numpy array.
            # log prob as tensor is fine. Our computation graph will
            # start later down the line.
        return int(action.item()), log_prob.detach()

    def rollout(self, env_seed: int, round_idx: int, client_id: int):
        # Batch data
        batch_obs = []                           # batch observations (number of timesteps per batch)
        batch_acts = []                          # batch actions (number of timesteps per batch)
        batch_log_probs = []                     # log probabilities of actions (number of timesteps per batch)
        batch_rews_per_timestep = []             # batch rewards (number of timesteps per episode)
        batch_rews_per_episode = []              # batch rewards (number of episodes)
        batch_rews_per_timestep_as_lists = []    # batch rewards (number of episodes) as lists (for each episode)
        batch_rtgs = []                          # batch rewards to go (number of timesteps per batch)  
        batch_lens = []                          # episode lengths in batch (number of episodes)
        batch_yields = []                        # batch yields (number of episodes)
        batch_leaches = []                       # batch leaches (number of episodes)
        batch_nitrogens = []                     # batch nitrogens (number of episodes)
        batch_waters = []                        # batch waters (number of episodes)
        batch_raw_ep_returns = []                # batch raw episode returns (number of episodes)

        episodes = 0
        first_reset = True

        
        while episodes < self.episodes_per_batch:
            # Reward of this episode
            ep_rews = []

            episodes += 1
            self.total_episodes += 1
            
            # print(f"    [Client {self.client_id}] starting episode {episodes} / {self.episodes_per_batch}", flush=True)

            # Reset the environment
            if first_reset and env_seed is not None:
                obs = self.env.reset(seed=int(env_seed))
                first_reset = False
            else:
                obs = self.env.reset()

            # print(f"    [Client {self.client_id}] first step of episode {episodes}", flush=True)

            # obs = self.env.reset(seed=int(env_seed)) if env_seed is not None else self.env.reset()
            done = False

            # print(f"    [Client {self.client_id}] round_idx == 24 {round_idx == 24, round_idx}, episodes == 16 {episodes == 16, episodes}, client_id == 5 {client_id == 5, client_id}", flush=True)

            for ep_t in range(self.max_timesteps_per_episode):
                # Collect obs
                batch_obs.append(obs)
                
                # Calculate the action and log probability
                action, log_prob = self.get_action(obs)
                obs, reward, done, info = self.env.step(action)

                # if round_idx == 24 and episodes == 16 and client_id == 5:
                #     print(f"    [Client {self.client_id}] action: {action}, reward: {reward}, done: {done}", flush=True)

                # Collect action, reward, and log probability
                ep_rews.append(float(reward))                        # Used for computing the return for the whole episode
                batch_rews_per_timestep.append(float(reward))        # Used for computing the return for the each timestep
                batch_acts.append(action)
                batch_log_probs.append(log_prob)

                if done:
                    batch_yields.append(info['episode_metrics']['yield'])
                    batch_leaches.append(info['episode_metrics']['leach_total'])
                    batch_nitrogens.append(info['episode_metrics']['N_total'])
                    batch_waters.append(info['episode_metrics']['W_total'])
                    batch_raw_ep_returns.append(info['raw_ep_return'])
                    break

            # Collect episodic length and rewards
            batch_lens.append(ep_t + 1)
            batch_rews_per_episode.append(np.sum(ep_rews))           # Used for computing the return for the whole episode
            batch_rews_per_timestep_as_lists.append(ep_rews)

            # self.writer.add_scalar(f'PPO_Client{self.client_id}/Episode_raw_ep_return', info['raw_ep_return'], self.total_episodes)

        # Reshape the data as tensors in the shape specified before returning
        batch_obs = np.array(batch_obs)
        batch_obs = torch.tensor(batch_obs, dtype=torch.float32)
        batch_acts = np.array(batch_acts)
        batch_acts = torch.tensor(batch_acts, dtype=torch.long)
        batch_log_probs = torch.stack(batch_log_probs)

        # ALG - Step 4:
        batch_rtgs = self.compute_rtgs(batch_rews_per_timestep_as_lists)

        # Calculate the average length, reward, yield, leach, nitrogen, and water for the current batch
        self.current_batch_average_length = np.mean(batch_lens)
        self.current_batch_average_reward = np.mean(batch_rews_per_episode)
        self.current_batch_average_yield = np.mean(batch_yields)
        self.current_batch_average_leach = np.mean(batch_leaches)
        self.current_batch_average_nitrogen = np.mean(batch_nitrogens)
        self.current_batch_average_water = np.mean(batch_waters)
        self.current_batch_average_raw_ep_return = np.mean(batch_raw_ep_returns)

        # Return the batch data
        return batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens

    def compute_rtgs(self, batch_rews_per_timestep_as_lists):
        # The rewards to go (rtg) per episode per batch to return.
        # The shape will be (num timesteps per batch)
        batch_rtgs = []

        # Iterate through each episode backwards to maintain same order in batch rtgs
        for ep_rews in reversed(batch_rews_per_timestep_as_lists):

            discounted_reward = 0 # The discounted reward so far

            for rew in reversed(ep_rews):
                discounted_reward = rew + self.gamma * discounted_reward
                batch_rtgs.insert(0, discounted_reward)

        # Convert to tensor
        batch_rtgs = torch.tensor(batch_rtgs, dtype=torch.float32)

        return batch_rtgs

    def get_value_using_critic(self, batch_obs):
        # Query the critic network for the value of each observation in batch_obs
        V = self.critic(batch_obs).squeeze(-1)

        # Return the value of the observations
        return V

    def get_log_probability_using_actor(self, batch_obs, batch_acts):
        # Query the actor network for the log probability of each action in batch_acts
        logits = self.actor(batch_obs)
        dist = Categorical(logits=logits)
        log_probs = dist.log_prob(batch_acts)

        entropy = dist.entropy().mean()

        # Return the log probability of the actions
        return log_probs, entropy

    def learn(self, total_episodes, theta: torch.FloatTensor, phi: torch.FloatTensor, env_seed: int,
              round_idx: int, client_id: int, writer_obj: SummaryWriter):
        print(f"    [Client {self.client_id}] - Policy Gradient Steps... Running {self.episodes_per_batch} episodes per batch for a total of {total_episodes} episodes")

        # Load the parameters from the theta tensor to the actor and critic
        self.actor.vector_to_parameters(torch.as_tensor(theta, dtype=torch.float32))

        if phi is not None:
            self.critic.vector_to_parameters(torch.as_tensor(phi, dtype=torch.float32))

        # Initialize the actor optimizer
        self.actor_optimizer = Adam(self.actor.parameters(), lr=self.actor_lr, eps=1e-5)

        # Initialize the critic optimizer
        self.critic_optimizer = Adam(self.critic.parameters(), lr=self.critic_lr, eps=1e-5)

        self.actor.train()
        self.critic.train()

        episodes_so_far = 0
        i_so_far = 0

        # ALG - Step 2:
        while episodes_so_far < total_episodes:
            # ALG - Step 3:
            batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens = self.rollout(env_seed, round_idx, client_id)

            # Increment the number of episodes simulated so far
            episodes_so_far += len(batch_lens)

            # Increment the number of iterations so far
            i_so_far += 1

            # Log the current batch average length, reward, yield, leach, nitrogen, and water
            writer_obj.add_scalar(f'Client{self.client_id}/PPO/Average_Length', self.current_batch_average_length, episodes_so_far + (round_idx-1)*total_episodes)
            writer_obj.add_scalar(f'Client{self.client_id}/PPO/Average_Raw_Ep_Return', self.current_batch_average_raw_ep_return, episodes_so_far + (round_idx-1)*total_episodes)
            writer_obj.add_scalar(f'Client{self.client_id}/PPO/Average_Yield', self.current_batch_average_yield, episodes_so_far + (round_idx-1)*total_episodes)
            writer_obj.add_scalar(f'Client{self.client_id}/PPO/Average_Leach', self.current_batch_average_leach, episodes_so_far + (round_idx-1)*total_episodes)
            writer_obj.add_scalar(f'Client{self.client_id}/PPO/Average_Nitrogen', self.current_batch_average_nitrogen, episodes_so_far + (round_idx-1)*total_episodes)
            writer_obj.add_scalar(f'Client{self.client_id}/PPO/Average_Water', self.current_batch_average_water, episodes_so_far + (round_idx-1)*total_episodes)
            
            print(f"    [Client {self.client_id}] Episode: {episodes_so_far}/{total_episodes}, Iteration: {i_so_far}, Batch Length: {self.current_batch_average_length:.2f}, "
                  f"Raw Ep Return: {self.current_batch_average_raw_ep_return:.2f}, Yield: {self.current_batch_average_yield:.2f}, "
                  f"Leach: {self.current_batch_average_leach:.2f}, Nitrogen: {self.current_batch_average_nitrogen:.2f}, "
                  f"Water: {self.current_batch_average_water:.2f}", flush=True)

            # Calculate V_{phi, k}
            V = self.get_value_using_critic(batch_obs)

            # ALG - Step 5:
            # Calculate the advantage
            A_k = batch_rtgs - V.detach()

            # Normalize the advantage
            A_k = (A_k - A_k.mean()) / (A_k.std() + 1e-10)

            # --------------------------------------------------------------------------------------------------------------- #
            # Made changes here for mini-batch training
            
            num_samples = batch_obs.shape[0]
            indices = np.arange(num_samples)

            for epoch in range(self.n_updates_per_batch):
                np.random.shuffle(indices)

                # Early-stop checking
                approx_kl_epoch = []

                for start in range(0, num_samples, self.mini_batch_size):
                    end = start + self.mini_batch_size

                    mb_idx = indices[start:end]
                    mb_obs = batch_obs[mb_idx]
                    mb_acts = batch_acts[mb_idx]
                    mb_log_probs = batch_log_probs[mb_idx].detach()
                    mb_adv = A_k[mb_idx]
                    mb_rtgs = batch_rtgs[mb_idx]

                    # Current policy & value on the mini-batch
                    curr_log_prob, curr_entropy = self.get_log_probability_using_actor(mb_obs, mb_acts)
                    mb_V = self.get_value_using_critic(mb_obs)

                    # Calculate ratio
                    log_ratio = curr_log_prob - mb_log_probs
                    ratio = torch.exp(log_ratio)

                    # Calculate KL divergence
                    mb_kl = (ratio - 1 - log_ratio).mean()
                    approx_kl_epoch.append(mb_kl.detach().item())

                    surr1 = ratio * mb_adv
                    surr2 = ratio.clamp(1 - self.clip, 1 + self.clip) * mb_adv

                    # Calculate loss
                    actor_loss = (-torch.min(surr1, surr2)).mean() - self.ent_coef * curr_entropy
                    critic_loss = nn.MSELoss()(mb_V, mb_rtgs) * self.vf_coef

                    # Calculate gradients and perform backward propagation for actor network
                    self.actor_optimizer.zero_grad()
                    actor_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
                    self.actor_optimizer.step()

                    # Calculate gradients and perform backward propagation for critic network
                    self.critic_optimizer.zero_grad()
                    critic_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
                    self.critic_optimizer.step()

                mean_approx_kl_epoch = np.mean(approx_kl_epoch) if approx_kl_epoch else 0.0
                if mean_approx_kl_epoch > self.kl_threshold:
                    # print(f"Early stop @ update {epoch} due to mean KL {mean_approx_kl_epoch:.3e}")
                    break

            # after the 4 upd in range(self.n_updates_per_batch): loop finishes
            # with torch.no_grad():
            #     _, curr_entropy = self.get_log_probability_using_actor(batch_obs, batch_acts)
                # print(f'post-update entropy at update {self.n_updates_per_batch}: {curr_entropy.item()}')

            # print(f"    [Client {self.client_id}] progress: {episodes_so_far}/{total_episodes} episodes", flush=True)

        current_results = {'reward': self.current_batch_average_raw_ep_return,
                           'yield': self.current_batch_average_yield,
                           'leach': self.current_batch_average_leach,
                           'nitrogen': self.current_batch_average_nitrogen,
                           'water': self.current_batch_average_water}
            
        updated_theta = self.actor.parameters_to_vector()
        updated_phi = self.critic.parameters_to_vector()
        return updated_theta, updated_phi, current_results


if __name__ == "__main__":

    master_seed = 0

    np.random.seed(master_seed)
    random.seed(master_seed)
    torch.manual_seed(master_seed)

    env_seed_1 = 123


    environment_arguments_1 = {
    'mode': 'all',
    'seed': env_seed_1,
    'random_weather': True,
    'env_id': 'GymDssatPdi-v0',
    'max_episode_steps': 180,
}


    env_1 = make_env(environment_arguments_1)

    if hasattr(env_1, 'training'):
        env_1.training = True
    if hasattr(env_1, 'norm_rew'):
        env_1.norm_rew = True
    if hasattr(env_1, 'norm_obs'):
        env_1.norm_obs = True


    state_dim = env_1.observation_space.shape[0]
    action_dim = env_1.action_space.n

    ppo_1 = PPO(state_dim, action_dim, env_1, client_id=1)

    policy = FeedForwardNN(state_dim, action_dim)
    theta = policy.parameters_to_vector().clone()

    total_episodes = 2000

    updated_theta_1 = ppo_1.learn(total_episodes, theta, env_seed_1)


    env_1.close()

    print("You are running the ppo.py file. Run main.py is meant to be run.")
