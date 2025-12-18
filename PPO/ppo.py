from torch.distributions import Categorical
import torch
import numpy as np
from torch.optim import Adam
import torch.nn as nn
import time
import gym

from torch.utils.tensorboard import SummaryWriter

class PPO:
    def __init__(self, state_dim: int, action_dim: int, env: gym.Env, policy_class: nn.Module):

        # Initialize the hyperparameters
        self.episodes_per_batch = 20                                    # episodes per batch for training
        self.max_timesteps_per_episode = 155                            # max timesteps per episode
        self.gamma = 0.99                                               # discount factor for future rewards
        self.mini_batch_size = 5 * self.max_timesteps_per_episode       # number of episodes per mini batch
        self.n_updates_per_batch = 5                                    # number of updates per batch
        self.clip = 0.2                                                 # clip parameter for PPO
        self.actor_lr = 3e-4                                            # learning rate for the actor network
        self.critic_lr = 3e-4                                           # learning rate for the critic network
        self.ent_coef = 0.0                                             # entropy coefficient for the actor network
        self.vf_coef = 0.5                                              # value function coefficient for the critic network
        self.save_freq = 10                                             # The frequency to save the model after every
        self.kl_threshold = 0.05                                        # The threshold for the KL divergence

        # Get the environment and the dimensions of the observation and action spaces
        self.env = env
        self.state_dim = state_dim
        self.action_dim = action_dim

        # Class variable for TensorBoard writer
        self.writer = None

        # Variables I am going to monitor on tensorboard
        self.current_batch_average_length = 0
        self.current_batch_average_reward = 0
        self.current_batch_average_yield = 0
        self.current_batch_average_leach = 0
        self.current_batch_average_nitrogen = 0
        self.current_batch_average_water = 0

        # ALG - Step 1:
        # Initialize the actor and critic networks
        self.actor = policy_class(self.state_dim, self.action_dim)
        self.critic = policy_class(self.state_dim, 1)

        # Initialize the actor optimizer
        self.actor_optimizer = Adam(self.actor.parameters(), lr=self.actor_lr, eps=1e-5)

        # Initialize the critic optimizer
        self.critic_optimizer = Adam(self.critic.parameters(), lr=self.critic_lr, eps=1e-5)

        self.global_once_print_flag = True
        self.global_once_print_flag1 = True
        self.global_once_print_flag2 = True

        self.total_episodes = 0

    def get_action(self, obs):
        with torch.no_grad():
            # Query the actor network for a mean action
            # Same thing as calling self.actor.forward(obs)
            obs = torch.as_tensor(obs, dtype=torch.float32)
            logits = self.actor(obs)

        
            # if self.global_once_print_flag:
            #     print('\n')
            #     print('logits: ', logits)
            #     print('logits_sum: ', logits.sum())
            #     self.global_once_print_flag = False
            #     print('\n')

            # Create a categorical distribution
            dist = Categorical(logits=logits)

            # if self.global_once_print_flag1:
            #     print('\n')
            #     print('dist: ', dist.probs)
            #     print('dist_sum: ', dist.probs.sum())
            #     self.global_once_print_flag1 = False
            #     print('\n')

            # Sample an action from the distribution and get its log probability
            action = dist.sample()
            log_prob = dist.log_prob(action)

            # if self.global_once_print_flag2:
            #     print('\n')
            #     print('action: ', action)
            #     print('log_prob: ', log_prob)
            #     self.global_once_print_flag2 = False
            #     print('\n')

            # Return the sampled action and the log prob of that action
            # Note that I'm calling detach() since the action and log_prob  
            # are tensors with computation graphs, so I want to get rid
            # of the graph and just convert the action to numpy array.
            # log prob as tensor is fine. Our computation graph will
            # start later down the line.
        return int(action.item()), log_prob.detach()

    def rollout(self):
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

        # Number of timesteps ran so far this batch
        episodes = 0
        while episodes < self.episodes_per_batch:
            # Reward of this episode
            ep_rews = []

            episodes += 1
            self.total_episodes += 1
            # Reset the environment
            obs = self.env.reset()
            done = False

            for ep_t in range(self.max_timesteps_per_episode):
                # Collect obs
                batch_obs.append(obs)
                
                # Calculate the action and log probability
                action, log_prob = self.get_action(obs)
                obs, reward, done, info = self.env.step(action)

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

            self.writer.add_scalar('Environment/Episode_raw_ep_return', info['raw_ep_return'], self.total_episodes)

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

        # print('\n[=]Checking the shape of the batch data')
        # print('batch_obs: ', np.shape(batch_obs))
        # print('batch_acts: ', np.shape(batch_acts))
        # print('batch_log_probs: ', np.shape(batch_log_probs))
        # print('batch_rtgs: ', np.shape(batch_rtgs))
        # print('batch_lens: ', np.shape(batch_lens))

        # print('\n')
        # print('batch_rews_per_timestep_as_lists: ', np.shape(batch_rews_per_timestep_as_lists))
        # print('batch_rews_per_episode: ', np.shape(batch_rews_per_episode))
        # print('batch_rews_per_timestep: ', np.shape(batch_rews_per_timestep))
        # print('batch_yields: ', np.shape(batch_yields))
        # print('batch_leaches: ', np.shape(batch_leaches))
        # print('batch_nitrogens: ', np.shape(batch_nitrogens))
        # print('batch_waters: ', np.shape(batch_waters))
        # print('[=]Done checking the shape of the batch data\n')

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

    def learn(self, total_episodes, log_path: str = 'logs', save_path: str = 'models'):
        print(f"Learning... Running {self.max_timesteps_per_episode} timesteps per episode, ", end='')
        print(f"{self.episodes_per_batch} episodes per batch for a total of {total_episodes} episodes")

        # Initialize TensorBoard writer
        if log_path is not None:            
            # Initialize TensorBoard writer
            self.writer = SummaryWriter(log_dir=log_path)

            # create a csv file to log the training progress
            with open(f'{log_path}/training_progress.csv', 'w') as f:
                f.write('episodes,mean_raw_ep_return,mean_yield,mean_leach_total,mean_N_total,mean_W_total\n')

        episodes_so_far = 0
        i_so_far = 0

        # ALG - Step 2:
        while episodes_so_far < total_episodes:
            # ALG - Step 3:
            batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens = self.rollout()

            # Increment the number of episodes simulated so far
            episodes_so_far += len(batch_lens)

            # Increment the number of iterations so far
            i_so_far += 1

            # Log the current batch average length, reward, yield, leach, nitrogen, and water
            self.writer.add_scalar('Environment/Current_Batch_Average_Length', self.current_batch_average_length, episodes_so_far)
            self.writer.add_scalar('Environment/Current_Batch_Average_Raw_Ep_Return', self.current_batch_average_raw_ep_return, episodes_so_far)
            self.writer.add_scalar('Environment/Current_Batch_Average_Yield', self.current_batch_average_yield, episodes_so_far)
            self.writer.add_scalar('Environment/Current_Batch_Average_Leach', self.current_batch_average_leach, episodes_so_far)
            self.writer.add_scalar('Environment/Current_Batch_Average_Nitrogen', self.current_batch_average_nitrogen, episodes_so_far)
            self.writer.add_scalar('Environment/Current_Batch_Average_Water', self.current_batch_average_water, episodes_so_far)
            
            if log_path is not None:
                with open(f'{log_path}/training_progress.csv', 'a') as f:
                    f.write(f'{episodes_so_far},{self.current_batch_average_reward},{self.current_batch_average_raw_ep_return},{self.current_batch_average_yield},'
                            f'{self.current_batch_average_leach},{self.current_batch_average_nitrogen},{self.current_batch_average_water}\n')

            # Print the current batch average length, reward, yield, leach, nitrogen, and water
            print()
            print(f"Episode {episodes_so_far}, Iteration {i_so_far}, Current Batch Average Length: {self.current_batch_average_length}, "
                  f"Current Batch Average Raw Ep Return: {self.current_batch_average_raw_ep_return}, "
                  f"Current Batch Average Yield: {self.current_batch_average_yield}, Current Batch Average Leach: {self.current_batch_average_leach}, "
                  f"Current Batch Average Nitrogen: {self.current_batch_average_nitrogen}, Current Batch Average Water: {self.current_batch_average_water}")


            # Calculate V_{phi, k}
            V = self.get_value_using_critic(batch_obs)

            # print('\n')
            # print('V: ', np.shape(V))
            # print('batch_rtgs: ', np.shape(batch_rtgs))
            # print('[=]Done checking the shape of the batch data\n')

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
                    print(f"Early stop @ update {epoch} due to mean KL {mean_approx_kl_epoch:.3e}")
                    break

            # --------------------------------------------------------------------------------------------------------------- #

            # Previous code for full batch training

            # for upd in range(self.n_updates_per_batch):
            #     # Calculate pi_theta(a_t | s_t)
            #     V = self.get_value_using_critic(batch_obs)
            #     curr_log_prob, curr_entropy = self.get_log_probability_using_actor(batch_obs, batch_acts)

                
            #     # Calculate ratio
            #     log_ratio = curr_log_prob - batch_log_probs
            #     ratio = torch.exp(log_ratio)
            #     approx_kl = (ratio - 1 - log_ratio).mean()

            #     if approx_kl.item() > self.kl_threshold:   # target_kl ~ 0.05
            #         print(f"Early stop @ update {upd} due to KL {approx_kl.item():.3e}")
            #         break

            #     print(f'curr_entropy at update {upd}: {curr_entropy}')

            #     # Calculate surrogate loss
            #     surr1 = ratio * A_k
            #     surr2 = ratio.clamp(1 - self.clip, 1 + self.clip) * A_k

            #     # Calculate loss
            #     actor_loss = (-torch.min(surr1, surr2)).mean()
            #     critic_loss = nn.MSELoss()(V, batch_rtgs)
            #     # print(f'actor_loss at update {upd}: {actor_loss.item()}')
            #     # print(f'critic_loss at update {upd}: {critic_loss.item()}')

            #     # Calculate gradients and perform backward propagation for actor network
            #     self.actor_optimizer.zero_grad()
            #     actor_loss.backward()
            #     torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
            #     self.actor_optimizer.step()

            #     # Calculate gradients and perform backward propagation for critic network
            #     self.critic_optimizer.zero_grad()
            #     critic_loss.backward()
            #     torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
            #     self.critic_optimizer.step()

            self.writer.add_scalar('Loss/Actor_Loss', actor_loss.item(), i_so_far)
            self.writer.add_scalar('Loss/Critic_Loss', critic_loss.item(), i_so_far)
            self.writer.add_scalar('Loss/Mean_Approx_KL_Epoch', mean_approx_kl_epoch, i_so_far)

            # after the 4 upd in range(self.n_updates_per_batch): loop finishes
            with torch.no_grad():
                _, curr_entropy = self.get_log_probability_using_actor(batch_obs, batch_acts)
                print(f'post-update entropy at update {self.n_updates_per_batch}: {curr_entropy.item()}')

            self.writer.add_scalar('Loss/Post_Update_Entropy', curr_entropy.item(), i_so_far)

            # Save our model if it's time
            if i_so_far % self.save_freq == 0:
                torch.save(self.actor.state_dict(), f'{save_path}/actor.pth')
                torch.save(self.critic.state_dict(), f'{save_path}/critic.pth')


    def get_writer(self):
        '''
        Get the TensorBoard writer for external logging
        Returns:
            SummaryWriter: The TensorBoard writer if initialized, None otherwise
        '''
        return self.writer

    def log_custom_metric(self, tag, value, step):
        '''
        Log a custom metric to TensorBoard
        Args:
            tag (str): The tag for the metric
            value (float): The value to log
            step (int): The step/episode number
        '''
        if self.writer is not None:
            self.writer.add_scalar(tag, value, step)