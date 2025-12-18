import torch
import numpy as np
import random

import time

from torch.utils.tensorboard import SummaryWriter

from network import FeedForwardNN
from ppo_fed_client import Client   # your class
from env_wrappers import make_env
from hashlib import sha256

# def H(*parts: object, bits: int = 32) -> int:
#     """
#     Deterministic hash -> integer seed.
#     parts: anything hashable/serializable to str (ints, strs, tuples)
#     bits:  32 or 64 typically
#     """
#     s = "|".join(str(p) for p in parts)
#     digest = sha256(s.encode("utf-8")).digest()
#     val = int.from_bytes(digest[:8], "big")   # 64-bit
#     return val % (2**bits)


class Server:
    def __init__(self, env_args: dict, master_seed: int, num_clients: int,
                 ppo_total_episodes: int, num_rounds: int, client_seeds: list[int]):

        self.env_args = env_args

        self.env = make_env(self.env_args)
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.n

        self.master_seed = master_seed
        self.num_clients = num_clients
        self.ppo_total_episodes = ppo_total_episodes
        self.num_rounds = num_rounds
        self.client_seeds = client_seeds
        
        np.random.seed(self.master_seed)
        random.seed(self.master_seed)
        torch.manual_seed(self.master_seed)

        # Initialize the actor and critic networks
        self.actor = FeedForwardNN(self.state_dim, self.action_dim)
        self.critic = FeedForwardNN(self.state_dim, 1)

        self.actor_vector = self.actor.parameters_to_vector()
        self.critic_vector = self.critic.parameters_to_vector()

        self.clients = []

        self.writer = None


    def initialize_clients(self):
        """
        Initialize the clients.
        """

        print(f"[Server] - Initializing {self.num_clients} clients.")
        self.clients.clear()
        for client_id in range(1, self.num_clients + 1):
            client_env_args = self.env_args.copy()
            client_env_args['seed'] = self.client_seeds[client_id - 1]
            print(f"[Server] - Client {client_id} seed: {client_env_args['seed']}")
            client = Client(client_env_args, client_id, self.actor, self.critic)
            self.clients.append(client)


    def aggregate_actors_and_critics(self, actors: list[torch.FloatTensor],
                                      critics: list[torch.FloatTensor]) -> tuple[torch.FloatTensor, torch.FloatTensor]:
        """
        Aggregate the actors and critics.

        Args:
            actors: The actors of the clients. Shape: (num_clients, vector(Actor)).
            critics: The critics of the clients. Shape: (num_clients, vector(Critic)).

        Returns:
            The aggregated actors and critics. Shape: (vector(Actor)), (vector(Critic)).
        """
        return torch.mean(actors, dim=0), torch.mean(critics, dim=0)


    def run_round_and_collect_fitnesses(self, round_idx: int) -> torch.Tensor:
        """
        Run the round and collect the fitnesses.
        """
        actors = []
        critics = []
        ppo_stats = []
        for client in self.clients:
            actor, critic, ppo_stat = client.run_round(ppo_total_episodes=self.ppo_total_episodes,
                             round_idx=round_idx, writer_obj=self.writer) # [1, vector(Actor)], [1, vector(Critic)]
            actors.append(actor)
            critics.append(critic)
            ppo_stats.append(ppo_stat)   

        reward = 0
        final_yield = 0
        leach = 0
        nitrogen = 0
        water = 0

        for ppo_stat in ppo_stats:
            reward += ppo_stat['reward']
            final_yield += ppo_stat['yield']
            leach += ppo_stat['leach']
            nitrogen += ppo_stat['nitrogen']
            water += ppo_stat['water']

        reward = reward / self.num_clients
        final_yield = final_yield / self.num_clients
        leach = leach / self.num_clients
        nitrogen = nitrogen / self.num_clients
        water = water / self.num_clients

        actors = torch.cat(actors, dim=0) # [num_clients, vector(Actor)]
        critics = torch.cat(critics, dim=0) # [num_clients, vector(Critic)]

        return actors, critics, reward, final_yield, leach, nitrogen, water
        
    def train_ppo_fed(self, log_path: str, save_path: str) -> None:

        self.initialize_clients()

        # Check if clients and server intialized theta is the same
        actors_same_as_server = all(torch.equal(client.actor_vector, self.actor_vector) for client in self.clients)
        critics_same_as_server = all(torch.equal(client.critic_vector, self.critic_vector) for client in self.clients)
        print(f"Initialization all clients have actors and critics - actors_same_as_server: {actors_same_as_server}, critics_same_as_server: {critics_same_as_server}")

        if log_path is not None:
            self.writer = SummaryWriter(log_dir=log_path)


        for round_idx in range(1, self.num_rounds + 1):       # round_idx is the round number - Algo line 3
            start_time = time.time()
            print(f"[Server] - Round {round_idx} - Training PPO-Fed")

            # run the round and collect the fitnesses - Algo line 4-9
            actors, critics, reward, final_yield, leach, nitrogen, water = self.run_round_and_collect_fitnesses(round_idx)

            print(f"[Server] - Round {round_idx} - Actors: {actors.shape}, Critics: {critics.shape}")

            # aggregate the normalization stats from the clients and update the server's env
            # total_obs_per_client = []
            # mean_obs_per_client = []
            # var_obs_per_client = []

            # for client in self.clients:
            #     total_obs_per_client.append(client.env.obs_rms.count)
            #     mean_obs_per_client.append(client.env.obs_rms.mean)
            #     var_obs_per_client.append(client.env.obs_rms.var)

            # update the obs_rms of the clients
            # for client in self.clients:
            #     client.env.obs_rms.count = obs_rms[0]
            #     client.env.obs_rms.mean = obs_rms[1]
            #     client.env.obs_rms.var = obs_rms[2]

            # aggregate the actors and critics - Algo line 10
            aggregated_actor, aggregated_critic = self.aggregate_actors_and_critics(actors, critics) 
            
            # update the actor and critic - Algo line 11
            self.actor_vector = aggregated_actor
            self.critic_vector = aggregated_critic
            self.actor.vector_to_parameters(aggregated_actor)
            self.critic.vector_to_parameters(aggregated_critic)

            # update the theta of the clients - Algo line 12-15
            for client in self.clients:
                client.update_actor_and_critic(aggregated_actor, aggregated_critic)

            # Check if clients and server intialized theta is the same
            actors_same_as_server = all(torch.equal(client.actor_vector, self.actor_vector) for client in self.clients)
            critics_same_as_server = all(torch.equal(client.critic_vector, self.critic_vector) for client in self.clients)
            print(f"Round {round_idx} all clients have actors and critics - actors_same_as_server: {actors_same_as_server}, critics_same_as_server: {critics_same_as_server}")

            self.log_custom_metric('Environment/Episode_rew', reward, round_idx)
            self.log_custom_metric('Environment/Episode_yield', final_yield, round_idx)
            self.log_custom_metric('Environment/Episode_leach', leach, round_idx)
            self.log_custom_metric('Environment/Episode_nitrogen', nitrogen, round_idx)
            self.log_custom_metric('Environment/Episode_water', water, round_idx)

            print(f"Round{round_idx} - Reward: {reward:.2f}, Yield: {final_yield:.2f}, Leach: {leach:.2f}, Nitrogen: {nitrogen:.2f}, Water: {water:.2f}")
            end_time = time.time()
            print(f"Round {round_idx} - Time taken: {end_time - start_time:.2f} seconds")

        torch.save(self.actor.state_dict(), f'{save_path}/actor.pth')
        torch.save(self.critic.state_dict(), f'{save_path}/critic.pth')

        for client in self.clients:
            torch.save(client.actor.state_dict(), f'{save_path}/Client_{client.client_id}_actor.pth')
            torch.save(client.critic.state_dict(), f'{save_path}/Client_{client.client_id}_critic.pth')
            client.env.save(f'{save_path}/Client_{client.client_id}_env_normalization_stats.json')
            print(f"Successfully saved client {client.client_id} actor and critic and environment normalization statistics.")

        for client in self.clients:
            client.env.close()

        self.env.close()

        if self.writer is not None:
            self.writer.close()

        print(f"Successfully trained PPO-Fed.", flush=True)


    def aggregate_normalization_stats(self, total_obs_per_client: list, mean_obs_per_client: list,
                                      var_obs_per_client: list) -> tuple[int, np.ndarray, np.ndarray]:

        n = 0 
        mean = None
        M2 = None

        for i in range(len(total_obs_per_client)):
            if total_obs_per_client[i] <= 0:
                continue
            
            M2k = var_obs_per_client[i] * total_obs_per_client[i]
            
            if mean is None: # first client
                n = total_obs_per_client[i]
                mean = mean_obs_per_client[i].copy()
                M2 = M2k.copy()
            else:
                delta = mean_obs_per_client[i] - mean
                n_new = n + total_obs_per_client[i]
                mean = mean + delta * total_obs_per_client[i] / n_new
                M2 = M2 + M2k + (delta**2) * ((n * total_obs_per_client[i]) / n_new)
                n = n_new
                
        if mean is None:
            return 0, None, None

        var = M2 / max(n, 1)

        return n, mean, var

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
    pass