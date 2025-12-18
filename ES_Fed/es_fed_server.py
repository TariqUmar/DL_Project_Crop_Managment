import torch
import numpy as np
import random
import time

from torch.utils.tensorboard import SummaryWriter

from network import FeedForwardNN
from es_fed_client import Client   # your class
from env_wrappers import make_env
from hashlib import sha256

def H(*parts: object, bits: int = 32) -> int:
    """
    Deterministic hash -> integer seed.
    parts: anything hashable/serializable to str (ints, strs, tuples)
    bits:  32 or 64 typically
    """
    s = "|".join(str(p) for p in parts)
    digest = sha256(s.encode("utf-8")).digest()
    val = int.from_bytes(digest[:8], "big")   # 64-bit
    return val % (2**bits)


class Server:
    def __init__(self, env_args: dict, master_seed: int, client_seeds: list[int], num_clients: int,
                 population_size: int, num_rounds: int, sigma: float, es_learning_rate: float):

        assert population_size % 2 == 0, "Population size must be even"

        self.env_args = env_args

        self.master_seed = master_seed                      # controls theta_0 and per-round epsilon seeds
        self.client_seeds = client_seeds                    # seeds for the clients
        self.num_clients = num_clients                      # number of clients
        self.population_size = population_size              # population size
        self.num_rounds = num_rounds                        # number of rounds
        self.sigma = sigma                                  # standard deviation of the perturbations
        self.es_learning_rate = es_learning_rate            # learning rate for the ES algorithm

        # Initialize the random seeds for reproducibility
        np.random.seed(self.master_seed)
        random.seed(self.master_seed)
        torch.manual_seed(self.master_seed)

        self.clients = []

        # TensorBoard writer for logging
        self.writer = None

    def initialize_clients(self):
        """
        Initialize the clients with their respective seeds.
        """

        print(f"[Server] - Initializing {self.num_clients} clients.")
        self.clients.clear()
        for client_id in range(1, self.num_clients + 1):
            client_env_args = self.env_args.copy()
            client_env_args['seed'] = self.client_seeds[client_id - 1]
            client = Client(client_env_args, client_id, self.master_seed,
                            self.population_size, self.sigma, self.es_learning_rate)
            self.clients.append(client)


    def aggregate_fitnesses(self, fitnesses: torch.Tensor) -> torch.Tensor:
        """
        Aggregate the fitnesses.

        Args:
            fitnesses: The fitnesses of the clients. Shape: (num_clients, population_size).

        Returns:
            The aggregated fitnesses. Shape: (1, population_size).
        """
        return torch.mean(fitnesses, dim=0).view(1, -1) # [1, population_size]


    def run_round_and_collect_fitnesses(self, round_seed: int, round_idx: int) -> torch.Tensor:
        """
        Run the round and collect the fitnesses.
        Args:
            round_seed: The seed for the perturbations.
            round_idx: The round index.
        Returns:
            The fitnesses of the clients. Shape: (num_clients, population_size).
        """
        total_reward = 0
        total_yield = 0
        total_leach = 0
        total_nitrogen = 0
        total_water = 0

        fitnesses_list = []

        for client in self.clients:
            fitness, population_stats = client.run_round(round_seed=round_seed, round_idx=round_idx, 
                                                           writer_obj=self.writer)
            fitnesses_list.append(fitness)
            total_reward += population_stats['average_reward']
            total_yield += population_stats['average_yield']
            total_leach += population_stats['average_leach']
            total_nitrogen += population_stats['average_nitrogen']
            total_water += population_stats['average_water']

        total_reward = total_reward / self.num_clients
        total_yield = total_yield / self.num_clients
        total_leach = total_leach / self.num_clients
        total_nitrogen = total_nitrogen / self.num_clients
        total_water = total_water / self.num_clients

        population_stats_aggregated = {
            'average_reward': total_reward,
            'average_yield': total_yield,
            'average_leach': total_leach,
            'average_nitrogen': total_nitrogen,
            'average_water': total_water,
        }

        fitnesses = torch.cat(fitnesses_list, dim=0) # [num_clients, population_size]
        return fitnesses, population_stats_aggregated
        
    def train_es_fed(self, log_path: str, save_path: str) -> None:

        self.initialize_clients()  # initialize the clients - Algo line 2

        # Check if all clients have the same theta
        client_1_theta = self.clients[0].theta
        clients_have_same_theta = all(torch.equal(client.theta, client_1_theta) for client in self.clients)
        print(f"Initialization all clients have theta - clients_have_same_theta: {clients_have_same_theta}")

        if log_path is not None:
            self.writer = SummaryWriter(log_dir=log_path)

            # with open(f'{log_path}/Server_training_progress.csv', 'w') as f:
            #     f.write('Round,Reward,Yield,Leach,Nitrogen,Water\n')

            # for client_idx in range(1, self.num_clients + 1):
            #     with open(f'{log_path}/Client{client_idx}_training_progress.csv', 'w') as f:
            #         f.write('Round,Reward,Yield,Leach,Nitrogen,Water\n')

        for round_idx in range(self.num_rounds):       # round_idx is the round number - Algo line 3

            start_time = time.time()
            print(f"[Server] - Round {round_idx} - Training ES-Fed")
            round_seed = H(self.master_seed, 'round', round_idx)

            # run the round and collect the fitnesses - Algo line 4-9
            fitnesses, population_stats_aggregated = self.run_round_and_collect_fitnesses(round_seed, round_idx)
            # Shape: [num_clients, population_size]

            print(f"[Server] - Round {round_idx} - Fitnesses: {fitnesses.shape}")

            self.log_custom_metric('Server/Average_reward', population_stats_aggregated['average_reward'], round_idx)
            self.log_custom_metric('Server/Average_yield', population_stats_aggregated['average_yield'], round_idx)
            self.log_custom_metric('Server/Average_leach', population_stats_aggregated['average_leach'], round_idx)
            self.log_custom_metric('Server/Average_nitrogen', population_stats_aggregated['average_nitrogen'], round_idx)
            self.log_custom_metric('Server/Average_water', population_stats_aggregated['average_water'], round_idx)

            print(f"Round{round_idx} - Reward: {population_stats_aggregated['average_reward']:.2f}, Yield: {population_stats_aggregated['average_yield']:.2f}, Leach: {population_stats_aggregated['average_leach']:.2f}, "
                  f"Nitrogen: {population_stats_aggregated['average_nitrogen']:.2f}, Water: {population_stats_aggregated['average_water']:.2f}")

            # aggregate the fitnesses - Algo line 10
            fitness = self.aggregate_fitnesses(fitnesses) # [1, population_size]
            
            # update the theta of the clients - Algo line 12-15
            for client in self.clients:
                client.update_theta(fitness)

            client_1_theta = self.clients[0].theta
            clients_have_same_theta = all(torch.equal(client.theta, client_1_theta) for client in self.clients)
            print(f"Round {round_idx} all clients have theta - clients_have_same_theta: {clients_have_same_theta}")

            end_time = time.time()
            print(f"Round {round_idx} - Time taken: {end_time - start_time:.2f} seconds")

            # Save the models
            for client in self.clients:
                torch.save(client.policy.state_dict(), f'{save_path}/Client_{client.client_id}_policy.pth')
                client.env.save(f'{save_path}/Client_{client.client_id}_env_normalization_stats.json')
                print(f"Successfully saved client {client.client_id} policy and environment normalization statistics.")
           
           
        for client in self.clients:
            client.env.close()

        if self.writer is not None:
            self.writer.close()

        print(f"Successfully trained ES-Fed.")

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
    pass