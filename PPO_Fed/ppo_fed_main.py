import os

from ppo_fed_server import Server
import time

def main():
     # --- Environment config for DSSAT ---
    env_args = {
        'mode': 'all',
        'seed': 456,                 # only used for the server's eval env; clients get H(master_seed, 'client', id)
        'random_weather': False,
        'env_id': 'GymDssatPdi-v0',
        'max_episode_steps': 180,
    }

    client_seeds = [312, 221, 101, 415, 518]

    # --- EvoFed hyperparams ---
    master_seed = 345                 # controls theta_0, per-client env seeds, and per-round epsilon seeds
    num_clients = 5
    ppo_total_episodes = 50
    num_rounds = 100

    # --- Logging / checkpoints ---
    log_path = "logs/RF5"
    save_path = "models/RF5"

    if not os.path.exists(log_path):
        os.makedirs(log_path)
    else:
        print(f"Log path {log_path} already exists.", flush=True)
        exit(0)

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    else:
        print(f"Save path {save_path} already exists.", flush=True)
        exit(0)

     # --- Construct and train ---
    server = Server(
        env_args=env_args,
        master_seed=master_seed,
        num_clients=num_clients,
        ppo_total_episodes=ppo_total_episodes,
        num_rounds=num_rounds,
        client_seeds=client_seeds
    )
    total_start_time = time.time()
    server.train_ppo_fed(log_path=log_path, save_path=save_path)
    print("PPO-Fed training complete. Final policy saved to:", save_path)
    total_end_time = time.time()
    print(f"Total time taken: {total_end_time - total_start_time:.2f} seconds")

if __name__ == "__main__":
    main()