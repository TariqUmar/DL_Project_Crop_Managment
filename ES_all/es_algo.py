import torch
import numpy as np
from torch.nn.utils import parameters_to_vector, vector_to_parameters

from network import FeedForwardNN
from env_wrappers import make_env

from torch.utils.tensorboard import SummaryWriter

class ES_Algo:
    def __init__(self, env, policy):

        self.env = env
        
        # Algo Step 1:
        self.policy = policy
        self.learning_rate = 1e-3
        self.population_size = 128
        self.sigma = 0.5

        # Class variable for TensorBoard writer
        self.writer = None
        

    def train(self, log_path, save_path, iterations):

        # Initialize TensorBoard writer
        if log_path is not None:            
            # Initialize TensorBoard writer
            self.writer = SummaryWriter(log_dir=log_path)

            # create a csv file to log the training progress
            with open(f'{log_path}/training_progress.csv', 'w') as f:
                f.write('iterations,mean_reward,mean_yield,mean_leach_total,mean_N_total,mean_W_total\n')


        # Aglo - Step 2: 
        for t in range(iterations):
            with torch.no_grad():
                theta = parameters_to_vector(self.policy.parameters())

            # Algo - Step 3:
            noise = torch.randn(self.population_size//2, theta.shape[0])
            negative_noise = -noise
            self.noise_list = torch.cat([negative_noise, noise], dim=0)

            # print(f"Noise List: {self.noise_list.shape}")

            # print(f"Shape of Noise List: {self.noise_list.shape}")
            # for i in range(self.noise_list.shape[0]):
            #     for j in range(self.noise_list.shape[0]):
            #         print(f"Relative distance: L2(eps {i} - eps {j})/L2(eps {i}): {torch.norm(self.noise_list[i] - self.noise_list[j])}/{torch.norm(self.noise_list[i])}")

            iteration_rewards = 0
            iteration_yields = 0
            iteration_leaches = 0
            iteration_nitrogens = 0
            iteration_waters = 0

            # Algo - Step 4:
            with torch.no_grad():
                self.reward_list = torch.zeros(self.population_size)
                theta_ns = self.noise_list * self.sigma + theta

            # print(f"Shape of Theta: {theta}")
            # print(f"Shape of Noise List: {self.noise_list}")
            # print(f"Shape of Theta_ns: {theta_ns}")

            for n in range(self.population_size):

                with torch.no_grad():
                    theta_n = theta_ns[n]
                    vector_to_parameters(theta_n, self.policy.parameters())

                # print(f"L2 norm of theta - theta_{n}: {torch.norm(theta - theta_n)}")

                obs = self.env.reset()
                done = False
                ep_reward = 0
                ep_len = 0
                
                while not done:
                    with torch.no_grad():
                        action_vec = self.policy(obs)
                        # print(f"Shape of Action Vector: {action_vec}")
                        
                    action_argmax = action_vec.argmax().item()
                    # print(f"Action Argmax: {action_argmax}")
                    # print()

                    obs, reward, done, info = self.env.step(action_argmax)
                    ep_reward += reward
                    ep_len += 1

                # print(f"Reward for episode {n}: {ep_reward}")
                
                self.writer.add_scalar('Environment/Episode_raw_reward', info['raw_ep_return'], (n+1)*(t+1))

                self.reward_list[n] = ep_reward
                iteration_rewards += ep_reward
                iteration_yields += info['episode_metrics']['yield']
                iteration_leaches += info['episode_metrics']['leach_total']
                iteration_nitrogens += info['episode_metrics']['N_total']
                iteration_waters += info['episode_metrics']['W_total']

            average_reward_per_iteration = iteration_rewards/self.population_size
            average_yield_per_iteration = iteration_yields/self.population_size
            average_leach_per_iteration = iteration_leaches/self.population_size
            average_nitrogen_per_iteration = iteration_nitrogens/self.population_size
            average_water_per_iteration = iteration_waters/self.population_size

            self.log_custom_metric('Environment/Average_Reward', average_reward_per_iteration, t)
            self.log_custom_metric('Environment/Average_Yield', average_yield_per_iteration, t)
            self.log_custom_metric('Environment/Average_Leach', average_leach_per_iteration, t)
            self.log_custom_metric('Environment/Average_Nitrogen', average_nitrogen_per_iteration, t)
            self.log_custom_metric('Environment/Average_Water', average_water_per_iteration, t)

            print(f"Iteration {t + 1}, Average Reward: {average_reward_per_iteration}, Average Yield: {average_yield_per_iteration}, "
                  f"Average Leach: {average_leach_per_iteration}, Average Nitrogen: {average_nitrogen_per_iteration}"
                  f"Average Water: {average_water_per_iteration}")

            with open(f'{log_path}/training_progress.csv', 'a') as f:
                f.write(f'{t + 1},{average_reward_per_iteration},{average_yield_per_iteration}, '
                        f'{average_leach_per_iteration},{average_nitrogen_per_iteration},{average_water_per_iteration}\n')

            # normalized_reward_list = (self.reward_list - torch.mean(self.reward_list)) / (torch.std(self.reward_list) + 1e-8)
            # print(f"Std of Reward List: {torch.std(self.reward_list)}")

            # Algo - Step 5:
            update_term = ((self.noise_list.T @ self.reward_list.reshape(-1, 1)).squeeze())/(self.population_size * self.sigma)
            print(f"L2 Norm of Update Term: {torch.norm(update_term)}")
            
            theta_new = theta + self.learning_rate * update_term

            print(f"L2 Norm of Theta - Theta_new: {torch.norm(theta - theta_new)}")
            
            theta = theta_new
            print(f"L2 Norm of Theta: {torch.norm(theta)}")
            print()

            with torch.no_grad():
                vector_to_parameters(theta, self.policy.parameters())

        torch.save(self.policy.state_dict(), f'{save_path}/policy.pth')


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


if __name__ == "__main__":
    
    environment_arguments = {
    'mode': 'all',
    'seed': 123,
    'random_weather': False,
    'env_id': 'GymDssatPdi-v0',
    'max_episode_steps': 180,
}

    env = make_env(environment_arguments)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    policy_class = FeedForwardNN(state_dim, action_dim)

    es_algo = ES_Algo(state_dim, action_dim, env, policy_class)
   
    for i in range(es_algo.noise_list.shape[0]):

        element = es_algo.noise_list[i]
        # print(f"Element {i} in noise list: {element}")

        negative_element = -element
        # print(f"Negative of element {i}: {negative_element}")

        idx = np.where(es_algo.noise_list == negative_element)[0][0]
        print(f"Index = {i}, Negative Index = {idx}")