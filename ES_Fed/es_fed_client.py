import torch
import numpy as np
import random

from torch.utils.tensorboard import SummaryWriter
from env_wrappers import make_env
from network import FeedForwardNN
from es_fed_es_algo import ES_Algo

class Client:
    def __init__(self, env_args: dict, client_id: int, master_seed: int, 
                 population_size: int, sigma: float, es_learning_rate: float) -> None:
        
        """
        env_args:   a dictionary of environment arguments
        client_id:  int (1,2,3,...,M)
        master_seed: seed for syncing theta_0
        population_size: population size
        sigma: standard deviation of the perturbations
        learning_rate: learning rate for the ES algorithm
        """
        
        self.env_args = env_args
        self.client_id = client_id
        self.master_seed = master_seed
        self.population_size = population_size
        self.sigma = sigma
        self.es_learning_rate = es_learning_rate

        self.env = make_env(self.env_args)
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.n

        torch.manual_seed(self.master_seed)

        self.policy = FeedForwardNN(self.state_dim, self.action_dim)

        self.es_algo = ES_Algo(self.env, self.client_id, self.population_size, self.sigma,
                               self.es_learning_rate)

        with torch.no_grad():
            self.theta = self.policy.parameters_to_vector().clone()

        self.theta_dim = self.theta.numel()

    def run_round(self, round_seed: int, round_idx: int, writer_obj: SummaryWriter) -> torch.FloatTensor:
        """
        Run one round of client side of the algorithm.
        1) Run the ES algorithm to get the fitnesses
        2) Send the fitnesses to the server
        
        
        return list of fitnesses (1, N)
        """

        if hasattr(self.env, 'training'):
            self.env.training = True
        if hasattr(self.env, 'norm_obs'):
            self.env.norm_obs = True
        if hasattr(self.env, 'norm_rew'):
            self.env.norm_rew = False

        fitnesses, population_stats = self.es_algo.train(self.theta, round_seed, round_idx, writer_obj)

        fitnesses = fitnesses.view(1, -1) # [1, population_size]

        return fitnesses, population_stats

    def update_theta(self, fitness: torch.FloatTensor) -> None: # fitness will come from the server
        """
        Update the theta of the client.
        Args:
            fitness: The fitnesses of the clients. Shape: (1, population_size).
        """
        updated_theta = self.es_algo.update_theta(theta= self.theta, fitness=fitness)
        self.theta = updated_theta

if __name__ == "__main__":

    master_seed = 123

    np.random.seed(master_seed)
    random.seed(master_seed)
    torch.manual_seed(master_seed)

    client_seeds = [1001, 2211, 3120, 4155, 5185]

    round_seed = 456

    environment_arguments_1 = {
    'mode': 'all',
    'seed': client_seeds[0],
    'random_weather': True,
    'env_id': 'GymDssatPdi-v0',
    'max_episode_steps': 180,
    }

    environment_arguments_2 = {
        'mode': 'all',
        'seed': client_seeds[1],
        'random_weather': True,
        'env_id': 'GymDssatPdi-v0',
        'max_episode_steps': 180,
    }

    client_1 = Client(environment_arguments_1, 1, master_seed, population_size=10, sigma=0.5, es_learning_rate=0.01)
    client_2 = Client(environment_arguments_2, 2, master_seed, population_size=10, sigma=0.5, es_learning_rate=0.01)

    print("client_1.theta == client_2.theta: ", torch.equal(client_1.theta, client_2.theta))

    writer = SummaryWriter(log_dir='logs')

    fitnesses_1, population_stats_1 = client_1.run_round(round_seed=round_seed, round_idx=0, writer_obj=writer)
    print("fitnesses_1: ", fitnesses_1)
    print("population_stats_1: ", population_stats_1)

    fitnesses_2, population_stats_2 = client_2.run_round(round_seed=round_seed, round_idx=0, writer_obj=writer)
    print("fitnesses_2: ", fitnesses_2)
    print("population_stats_2: ", population_stats_2)

    print("fitnesses_1 == fitnesses_2: ", torch.equal(fitnesses_1, fitnesses_2))

    fitnesses_concat = torch.cat([fitnesses_1, fitnesses_2], dim=0)
    print("fitnesses_concat: ", fitnesses_concat.shape)

    Fitness = torch.mean(fitnesses_concat, dim=0)
    Fitness = Fitness.reshape(1, -1)
    print("Fitness: ", Fitness.shape)


    client_1.update_theta(Fitness)
    client_2.update_theta(Fitness)

    print("client_1.theta == client_2.theta: ", torch.equal(client_1.theta, client_2.theta)) ## check is the updated thetas are still the same