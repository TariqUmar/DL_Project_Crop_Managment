import torch
import numpy as np
import random

from torch.utils.tensorboard import SummaryWriter
from env_wrappers import make_env
from network import FeedForwardNN
from ppo_fed import PPO

class Client:
    def __init__(self, env_args: dict, client_id: int, actor: FeedForwardNN, critic: FeedForwardNN) -> None:
        
        """
        env_args:   a dictionary of environment arguments
        client_id:  int (1,2,3,...,M)
        """
        
        self.env_args = env_args
        self.client_id = client_id

        self.env = make_env(self.env_args)
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.n

        self.actor = actor
        self.critic = critic

        self.actor_vector = self.actor.parameters_to_vector()
        self.critic_vector = self.critic.parameters_to_vector()

        self.ppo = PPO(self.state_dim, self.action_dim, self.env, self.client_id)

        self.writer = None

    def run_round(self, ppo_total_episodes: int, round_idx: int, writer_obj: SummaryWriter) -> torch.FloatTensor:
        """
        Run one round of client side of the algorithm.
        Using PPO, update the actor and critic networks

        return policy
        """

        if hasattr(self.env, 'training'):
            self.env.training = True
        if hasattr(self.env, 'norm_obs'):
            self.env.norm_obs = True
        if hasattr(self.env, 'norm_rew'):
            self.env.norm_rew = True

        self.actor.train()
        self.critic.train()

        vector_actor = self.actor.parameters_to_vector()
        vector_critic = self.critic.parameters_to_vector()


        updated_vector_actor, updated_vector_critic, final_ppo_stats = self.ppo.learn(ppo_total_episodes,
                                        vector_actor, vector_critic, self.env_args['seed'], round_idx,
                                        self.client_id, writer_obj) 
        # [1, vector(Actor)], [1, vector(Critic)], 
        # {'reward': float, 'yield': float, 'leach': float, 'nitrogen': float, 'water': float}

        self.actor_vector = updated_vector_actor
        self.critic_vector = updated_vector_critic

        return updated_vector_actor.view(1, -1), updated_vector_critic.view(1, -1), final_ppo_stats

    def update_actor_and_critic(self, actor: torch.FloatTensor, critic: torch.FloatTensor) -> None:
        """
        Update the actor and critic networks.

        """

        self.actor_vector = actor
        self.critic_vector = critic

        self.actor.vector_to_parameters(actor)
        self.critic.vector_to_parameters(critic)


if __name__ == "__main__":

    from network import FeedForwardNN
    from env_wrappers import make_env

    client_id = 1

    env_args = {
        'mode': 'all',
        'seed': 456,
        'random_weather': False,
        'env_id': 'GymDssatPdi-v0',
        'max_episode_steps': 180,
    }

    env = make_env(env_args)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    actor = FeedForwardNN(state_dim, action_dim)
    critic = FeedForwardNN(state_dim, 1)

    client = Client(env_args, client_id, actor, critic)

    print("client.actor: ", client.actor)
    print("client.critic: ", client.critic)

    total_ppo_episodes = 20
    round_idx = 1
    writer_obj = SummaryWriter(log_dir=f"logs/client_{client_id}_round_{round_idx}")

    vector_actor, vector_critic = client.run_round(total_ppo_episodes, round_idx, writer_obj)

    print("vector_actor: ", vector_actor.shape)
    print("vector_critic: ", vector_critic.shape)
