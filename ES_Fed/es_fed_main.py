import os
import time

from es_fed_server import Server

def main():
     # --- Environment config for DSSAT ---
    env_args = {
        'mode': 'all',
        'seed': 0,                 # Not used anywhere, client seeds are used instead
        'random_weather': True,
        'env_id': 'GymDssatPdi-v0',
        'max_episode_steps': 180,
    }

    client_seeds = [312, 221, 101, 415, 518]

    # --- EvoFed hyperparams ---
    master_seed = 124                    # controls theta_0 and per-round epsilon seeds
    num_clients = len(client_seeds)
    population_size = 64                 # must be even (server enforces)
    num_rounds = 100
    sigma = 0.5
    es_learning_rate = 0.001

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
        client_seeds=client_seeds,
        num_clients=num_clients,
        population_size=population_size,
        num_rounds=num_rounds,
        sigma=sigma,
        es_learning_rate=es_learning_rate
    )
    total_start_time = time.time()
    server.train_es_fed(log_path=log_path, save_path=save_path)
    total_end_time = time.time()
    print("ES-Fed training complete. Final policy saved to:", save_path)
    print(f"Total time taken: {total_end_time - total_start_time:.2f} seconds")
if __name__ == "__main__":
    main()