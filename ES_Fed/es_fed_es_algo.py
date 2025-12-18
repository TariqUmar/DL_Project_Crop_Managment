import torch
import numpy as np
from torch.nn.utils import parameters_to_vector, vector_to_parameters
import gym

from network import FeedForwardNN
from env_wrappers import make_env

from torch.distributions import Categorical

from torch.utils.tensorboard import SummaryWriter

class ES_Algo:
    def __init__(self, env: gym.Env, client_id: int, population_size: int, sigma: float, es_learning_rate: float):

        self.client_id = client_id
        self.env = env
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.n
        
        # Algo Step 1:
        self.es_learning_rate = es_learning_rate
        self.population_size = population_size
        self.sigma = sigma

        self.policy = FeedForwardNN(self.state_dim, self.action_dim)
        self.theta_dim = self.policy.parameters_to_vector().numel()

        self.eps = None


    def make_epsilons(self, N: int, round_seed: int) -> None:
        """
        Generate identical perturbations from a shared round_seed.
        Returns eps with shape [N, theta_dim], dtype float32.

        Args:
            N:              number of perturbations (population size) always even
            round_seed:     seed for the perturbations
            sigma:          standard deviation of the perturbations

        Returns:
            eps:        perturbations with shape [N, theta_dim], dtype float32
        """

        rng = np.random.Generator(np.random.PCG64(round_seed))
        half_N = N // 2
        e = rng.standard_normal(size=(half_N, self.theta_dim), dtype=np.float64)
        e = e.astype(np.float32, copy=False)
        eps = np.concatenate([e, -e], axis=0)
        eps = eps * self.sigma
        self.eps = torch.tensor(eps)


    def train(self, theta: torch.FloatTensor, round_seed: int, round_idx: int,  writer_obj: SummaryWriter) -> torch.FloatTensor:
        """
        This function will take the current theta and make perturbations using the epsilons.
        It will then evaluate the fitness of the perturbations.
        Return the fitnesses of the perturbations, that are basically the rewards from running the perturbations.

        Args:
            theta: current theta
            round_seed: round seed
            round_idx: round index
            writer_obj: writer object

        Returns:
            fitnesses of the perturbations
        """

        self.make_epsilons(N=self.population_size, round_seed=round_seed)

        theta_ns = self.eps + theta     
        # [population_size, theta_dim] + [theta_dim,] = [population_size, theta_dim]

        reward_list = []
        yield_list = []
        leach_list = []
        nitrogen_list = []
        water_list = []

        for n in range(self.population_size):
            ep_reward, episode_yield, episode_leach, episode_nitrogen, episode_water = self.run_theta(theta_ns[n])
            reward_list.append(ep_reward)
            yield_list.append(episode_yield)
            leach_list.append(episode_leach)
            nitrogen_list.append(episode_nitrogen)
            water_list.append(episode_water)

        average_reward = torch.mean(torch.tensor(reward_list))
        average_yield = torch.mean(torch.tensor(yield_list))
        average_leach = torch.mean(torch.tensor(leach_list))
        average_nitrogen = torch.mean(torch.tensor(nitrogen_list))
        average_water = torch.mean(torch.tensor(water_list))

        population_stats = {
            'average_reward': average_reward,
            'average_yield': average_yield,
            'average_leach': average_leach,
            'average_nitrogen': average_nitrogen,
            'average_water': average_water,
        }

        writer_obj.add_scalar(f'Client{self.client_id}/Average_Reward', average_reward, round_idx)
        writer_obj.add_scalar(f'Client{self.client_id}/Average_Yield', average_yield, round_idx)
        writer_obj.add_scalar(f'Client{self.client_id}/Average_Leach', average_leach, round_idx)
        writer_obj.add_scalar(f'Client{self.client_id}/Average_Nitrogen', average_nitrogen, round_idx)
        writer_obj.add_scalar(f'Client{self.client_id}/Average_Water', average_water, round_idx)

        print(f"        [Client{self.client_id}] - Average Reward: {average_reward:.2f}, "
              f"Average Yield: {average_yield:.2f}, Average Leach: {average_leach:.2f}, "
              f"Average Nitrogen: {average_nitrogen:.2f}, Average Water: {average_water:.2f}")

        return torch.tensor(reward_list), population_stats


    def run_theta(self, theta: torch.FloatTensor) -> torch.FloatTensor:
        """
        This function will run the theta on the environment and return the reward.
        """

        with torch.no_grad():
            vector_to_parameters(theta, self.policy.parameters())

        obs = self.env.reset()
        done = False
        ep_reward = 0
        ep_len = 0

        while not done:
            with torch.no_grad():
                logits = self.policy(obs)
                dist = Categorical(logits=logits)
                action = dist.sample()
                # action = torch.argmax(logits)

            obs, reward, done, info = self.env.step(action.item())
            ep_reward += reward
            ep_len += 1

            if done:
                episode_yield = info['episode_metrics']['yield']
                episode_leach = info['episode_metrics']['leach_total']
                episode_nitrogen = info['episode_metrics']['N_total']
                episode_water = info['episode_metrics']['W_total']

                return ep_reward, episode_yield, episode_leach, episode_nitrogen, episode_water


    def update_theta(self, theta: torch.FloatTensor, fitness: torch.FloatTensor) -> None: # fitness will come from the server
        """
        Update the theta of the client.
        Args:
            fitness: The fitnesses of the clients. Shape: (1, population_size).
        """
        upd_theta = theta + ((self.es_learning_rate / (self.population_size * self.sigma)) * (fitness @ self.eps).squeeze())
        return upd_theta


if __name__ == "__main__":
    
    environment_arguments = {
    'mode': 'all',
    'seed': 123,
    'random_weather': False,
    'env_id': 'GymDssatPdi-v0',
    'max_episode_steps': 180,
}

    master_seed = 123
    np.random.seed(master_seed)
    torch.manual_seed(master_seed)

    

    round_seed = 456

    env = make_env(environment_arguments)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    policy = FeedForwardNN(state_dim, action_dim)

    theta = policy.parameters_to_vector()

    es_algo = ES_Algo(env, client_id=0, population_size=10, sigma=0.5, learning_rate=0.01)

    writer = SummaryWriter(log_dir='logs')

    fitnesses = es_algo.train(theta, round_seed, round_idx=0, writer_obj=writer)

    print("fitnesses: ", fitnesses.shape)