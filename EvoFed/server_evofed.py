import torch
import numpy as np
import random
import time

from torch.utils.tensorboard import SummaryWriter

from network import FeedForwardNN
from client_evofed import Client   # your class
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
                 ppo_total_episodes: int, population_size: int, num_rounds: int, sigma: float, es_learning_rate: float):

        assert population_size % 2 == 0, "Population size must be even"

        self.env_args = env_args

        self.master_seed = master_seed                      # controls theta_0 and per-round epsilon seeds
        self.client_seeds = client_seeds                    # seeds for the clients
        self.num_clients = num_clients                      # number of clients
        self.ppo_total_episodes = ppo_total_episodes        # number of episodes for each client to train
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
            client = Client(client_env_args, client_id, self.master_seed)
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
        fitnesses = []
        current_reward = 0
        current_yield = 0
        current_leach = 0
        current_nitrogen = 0
        current_water = 0
        for client in self.clients:
            fitness, current_results = client.run_round(ppo_total_episodes=self.ppo_total_episodes, 
                             population_size=self.population_size, sigma=self.sigma,
                             round_seed=round_seed, round_idx=round_idx, writer_obj=self.writer) # [1, population_size]
            fitnesses.append(fitness)
            current_reward += current_results['reward']
            current_yield += current_results['yield']
            current_leach += current_results['leach']
            current_nitrogen += current_results['nitrogen']
            current_water += current_results['water']

        current_reward = current_reward / self.num_clients
        current_yield = current_yield / self.num_clients
        current_leach = current_leach / self.num_clients
        current_nitrogen = current_nitrogen / self.num_clients
        current_water = current_water / self.num_clients

        current_results_aggregated = {'reward': current_reward,
                                      'yield': current_yield,
                                      'leach': current_leach,
                                      'nitrogen': current_nitrogen,
                                      'water': current_water}

        fitnesses = torch.cat(fitnesses, dim=0) # [num_clients, population_size]
        return fitnesses, current_results_aggregated
        
    def train_evofed(self, log_path: str, save_path: str, load_path: str = None, load_models: bool = False) -> None:

        self.initialize_clients()  # initialize the clients - Algo line 2

        if load_models:
            self.load_models(load_path)
            print("Successfully loaded models.")

        # Check if all clients have the same theta
        client_1_theta = self.clients[0].theta
        clients_have_same_theta = all(torch.equal(client.theta, client_1_theta) for client in self.clients)
        print(f"Initialization all clients have theta - clients_have_same_theta: {clients_have_same_theta}")

        if log_path is not None:
            self.writer = SummaryWriter(log_dir=log_path)

            with open(f'{log_path}/Server_training_progress.csv', 'w') as f:
                f.write('Round,Reward,Yield,Leach,Nitrogen,Water\n')

            for client_idx in range(1, self.num_clients + 1):
                with open(f'{log_path}/Client{client_idx}_training_progress.csv', 'w') as f:
                    f.write('Round,Reward,Yield,Leach,Nitrogen,Water\n')

        for round_idx in range(0, self.num_rounds):       # round_idx is the round number - Algo line 3
            start_time = time.time()
            print(f"[Server] - Round {round_idx} - Training EvoFed")
            round_seed = H(self.master_seed, 'round', round_idx)

            # run the round and collect the fitnesses - Algo line 4-9
            fitnesses, current_results_aggregated = self.run_round_and_collect_fitnesses(round_seed, round_idx)
            # Shape: [num_clients, population_size]

            print(f"[Server] - Round {round_idx} - Fitnesses: {fitnesses.shape}")

            self.log_custom_metric('Environment/Episode_rew', current_results_aggregated['reward'], round_idx)
            self.log_custom_metric('Environment/Episode_yield', current_results_aggregated['yield'], round_idx)
            self.log_custom_metric('Environment/Episode_leach', current_results_aggregated['leach'], round_idx)
            self.log_custom_metric('Environment/Episode_nitrogen', current_results_aggregated['nitrogen'], round_idx)
            self.log_custom_metric('Environment/Episode_water', current_results_aggregated['water'], round_idx)

            print(f"Round{round_idx} - Reward: {current_results_aggregated['reward']:.2f}, Yield: {current_results_aggregated['yield']:.2f}, Leach: {current_results_aggregated['leach']:.2f}, "
                  f"Nitrogen: {current_results_aggregated['nitrogen']:.2f}, Water: {current_results_aggregated['water']:.2f}")

            # with open(f'{log_path}/Server_training_progress.csv', 'a') as f:
            #     f.write(f'{round_idx},{current_results_aggregated['reward']:.2f},{current_results_aggregated['yield']:.2f},{current_results_aggregated['leach']:.2f},{current_results_aggregated['nitrogen']:.2f},{current_results_aggregated['water']:.2f}\n')

            # aggregate the fitnesses - Algo line 10
            fitness = self.aggregate_fitnesses(fitnesses) # [1, population_size]
            

            # update the theta of the clients - Algo line 12-15
            for client in self.clients:
                client.update_theta(fitness, self.population_size, self.sigma, self.es_learning_rate)

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

        print(f"Successfully trained EvoFed.", flush=True)

    def load_models(self, load_path: str) -> None:
        """
        Load the models from the load path.
        """
        for client in self.clients:
            client.policy.load_state_dict(torch.load(f'{load_path}/Client_{client.client_id}_policy.pth'))
            client.env.load(f'{load_path}/Client_{client.client_id}_env_normalization_stats.json')
            print(f"Successfully loaded client {client.client_id} policy and environment normalization statistics.")


    # def server_evaluate(self, round_idx: int, log_path: str) -> tuple[float, float, float, float, float, float]:

    #     """
    #     Call the evaluate function for all clients.
    #     Clients will evaluate the current policy that they have.

    #     Average the rewards, yields, leaches, nitrogens, and waters from all clients.

    #     Returns:
    #         The average reward, yield, leach, nitrogen, and water.
    #     """
    #     client_rews = []
    #     client_yields = []
    #     client_leaches = []
    #     client_nitrogens = []
    #     client_waters = []

    #     for client_idx, client in enumerate(self.clients):
    #         Rew, Yield, Leach, Nitrogen, Water = client.evaluate()
    #         self.log_custom_metric(f'Client{client_idx + 1}/Eval_rew', Rew, round_idx)
    #         self.log_custom_metric(f'Client{client_idx + 1}/Eval_yield', Yield, round_idx)
    #         self.log_custom_metric(f'Client{client_idx + 1}/Eval_leach', Leach, round_idx)
    #         self.log_custom_metric(f'Client{client_idx + 1}/Eval_nitrogen', Nitrogen, round_idx)
    #         self.log_custom_metric(f'Client{client_idx + 1}/Eval_water', Water, round_idx)
    #         with open(f'{log_path}/Client{client_idx + 1}_training_progress.csv', 'a') as f:
    #             f.write(f'{round_idx},{Rew:.2f},{Yield:.2f},{Leach:.2f},{Nitrogen:.2f},{Water:.2f}\n')
    #         client_rews.append(Rew)
    #         client_yields.append(Yield)
    #         client_leaches.append(Leach)
    #         client_nitrogens.append(Nitrogen)
    #         client_waters.append(Water)

    #     avg_rew = np.mean(client_rews)
    #     avg_yield = np.mean(client_yields)
    #     avg_leach = np.mean(client_leaches)
    #     avg_nitrogen = np.mean(client_nitrogens)
    #     avg_water = np.mean(client_waters)

    #     return avg_rew, avg_yield, avg_leach, avg_nitrogen, avg_water

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