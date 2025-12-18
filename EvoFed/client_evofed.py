import torch
import numpy as np
import random

from torch.utils.tensorboard import SummaryWriter
from env_wrappers import make_env
from network import FeedForwardNN
from ppo_evofed import PPO

class Client:
    def __init__(self, env_args: dict, client_id: int, master_seed: int) -> None:
        
        """
        env_args:   a dictionary of environment arguments
        client_id:  int (1,2,3,...,M)
        """
        
        self.env_args = env_args
        self.client_id = client_id
        self.master_seed = master_seed

        self.env = make_env(self.env_args)
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.n

        self.eps = None

        torch.manual_seed(self.master_seed)

        self.policy = FeedForwardNN(self.state_dim, self.action_dim)

        self.ppo = PPO(self.state_dim, self.action_dim, self.env, self.client_id)

        with torch.no_grad():
            self.theta = self.policy.parameters_to_vector().clone()

        self.phi = None

        self.theta_dim = self.theta.numel()

        self.writer = None


    def make_epsilons(self, N: int, round_seed: int, sigma: float) -> None:
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
        eps = eps * sigma
        self.eps = torch.tensor(eps)


    def run_round(self, ppo_total_episodes: int, population_size: int, sigma: float,
                  round_seed: int, round_idx: int, writer_obj: SummaryWriter) -> torch.FloatTensor:
        """
        Run one round of client side of the algorithm.
        1) Using PPO, update the policy network and get theta prime
        2) Using the round seed, generate epsilons (Noises)
        3) Using epsilons, generate theta_eps (population of theta_eps)
        4) Compute the fitness of each theta_eps, using - (L2 norm)^2 of theta_eps - theta_prime
        
        return list of fitnesses (1, N)
        """

        if hasattr(self.env, 'training'):
            self.env.training = True
        if hasattr(self.env, 'norm_obs'):
            self.env.norm_obs = True
        if hasattr(self.env, 'norm_rew'):
            self.env.norm_rew = True

        theta_prime, upd_phi, current_results = self.ppo.learn(ppo_total_episodes, self.theta, self.phi,
                                                self.env_args['seed'], round_idx, self.client_id, writer_obj)

        theta_prime = theta_prime.clone().detach()
        upd_phi = upd_phi.clone().detach()
        self.phi = upd_phi

        self.make_epsilons(N=population_size, round_seed=round_seed, sigma=sigma)

        theta_eps = self.theta + self.eps  # (population_size, theta_dim) by broadcasting

        # self.eps [population_size, theta_dim] + self.theta [theta_dim,] = [population_size, theta_dim]

        fitnesses = (-1) * torch.norm(theta_eps - theta_prime, dim=1) ** 2

        fitnesses = fitnesses.view(1, -1) # [1, population_size]
        return fitnesses, current_results

    def update_theta(self, fitness: torch.FloatTensor, population_size: int, sigma: float, es_learning_rate: float) -> None: # fitness will come from the server
        """
        Update the theta of the client.
        Args:
            fitness: The fitnesses of the clients. Shape: (1, population_size).
            population_size: The population size.
            sigma: The standard deviation of the perturbations.
            es_learning_rate: The learning rate for the ES algorithm.
        """
        self.theta = self.theta + ((es_learning_rate / (population_size * sigma)) * (fitness @ self.eps).squeeze())
         

    # def evaluate(self) -> tuple[float, float, float, float, float, float]:
    #     """
    #     Evaluate the policy on the environment.
    #     """
    #     with torch.no_grad():
    #         self.policy.vector_to_parameters(self.theta)


    #     if hasattr(self.env, 'training'):
    #         self.env.training = False
    #     if hasattr(self.env, 'norm_obs'):
    #         self.env.norm_obs = True
    #     if hasattr(self.env, 'norm_rew'):
    #         self.env.norm_rew = False

    #     episode_steps = 0
    #     done = False
    #     episode_rew = 0

    #     obs = self.env.reset(seed=self.env_args['seed'])

    #     while not done:
    #         logits = self.policy(obs)
    #         action = torch.argmax(logits).item()

    #         obs, rew, done, info = self.env.step(action)
            
    #         episode_rew += rew
    #         episode_steps += 1

    #         if done:
    #             episode_yield = info['episode_metrics']['yield']
    #             episode_leach = info['episode_metrics']['leach_total']
    #             episode_nitrogen = info['episode_metrics']['N_total']
    #             episode_water = info['episode_metrics']['W_total']

    #     return episode_rew, episode_yield, episode_leach, episode_nitrogen, episode_water


if __name__ == "__main__":

    master_seed = 0

    np.random.seed(master_seed)
    random.seed(master_seed)
    torch.manual_seed(master_seed)

    env_seed_1 = 123
    env_seed_2 = 456

    round_seed = 456

    environment_arguments_1 = {
    'mode': 'all',
    'seed': env_seed_1,
    'random_weather': True,
    'env_id': 'GymDssatPdi-v0',
    'max_episode_steps': 180,
    }

    environment_arguments_2 = {
        'mode': 'all',
        'seed': env_seed_2,
        'random_weather': True,
        'env_id': 'GymDssatPdi-v0',
        'max_episode_steps': 180,
    }

    client_1 = Client(environment_arguments_1, 1, master_seed)
    client_2 = Client(environment_arguments_2, 2, master_seed)

    print("client_1.theta == client_2.theta: ", torch.allclose(client_1.theta, client_2.theta))


    fitnesses_1 = client_1.run_round(ppo_total_episodes=40, population_size=4, sigma=0.5, round_seed=round_seed)
    print("fitnesses_1: ", fitnesses_1)

    # np.random.seed(master_seed)
    # random.seed(master_seed)
    # torch.manual_seed(master_seed)

    fitnesses_2 = client_2.run_round(ppo_total_episodes=40, population_size=4, sigma=0.5, round_seed=round_seed)
    print("fitnesses_2: ", fitnesses_2)
    print("fitnesses_1 == fitnesses_2: ", torch.allclose(fitnesses_1, fitnesses_2))

    fitnesses_concat = torch.cat([fitnesses_1, fitnesses_2], dim=0)
    print("fitnesses_concat: ", fitnesses_concat.shape)

    Fitness = torch.mean(fitnesses_concat, dim=0)
    Fitness = Fitness.reshape(1, -1)
    print("Fitness: ", Fitness.shape)

    client_1.make_epsilons(N=4, round_seed=round_seed, sigma=0.5)
    print("eps: ", client_1.eps.shape)

    client_1.update_theta(Fitness, 4, 0.5, 0.01)
    client_2.update_theta(Fitness, 4, 0.5, 0.01)

    print("client_1.theta == client_2.theta: ", torch.allclose(client_1.theta, client_2.theta)) ## check is the updated thetas are still the same