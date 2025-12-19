# Deep Learning Project: Crop Management with Reinforcement Learning

A comprehensive deep learning project for optimizing crop management strategies using reinforcement learning algorithms. This project implements multiple RL approaches (PPO, Evolution Strategy, and their federated variants) to optimize fertilizer application in crop management using the GymDSSAT-PDI environment.

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Algorithms Implemented](#algorithms-implemented)
- [Environment](#environment)
- [Installation](#installation)
- [Usage](#usage)
- [Project Organization](#project-organization)
- [Results](#results)
- [License](#license)

## Overview

This project explores the application of reinforcement learning techniques to optimize crop management decisions, specifically focusing on fertilizer application strategies. The goal is to maximize crop yield while minimizing usage of resources, fertilizer(nitrogen) and irrigation water, and environmental impact (nitrogen leaching) through intelligent decision-making policies learned from crop simulation data.

The project implements both centralized and federated learning approaches, allowing for comparison of different training paradigms in agricultural decision-making scenarios.

## Features

- **RL Algorithms**: Implementation of PPO (Proximal Policy Optimization) and Evolution Strategy (ES) algorithms
- **Federated Learning Support**: Federated variants of both PPO and ES for distributed training
- **Hybrid Approach**: EvoFed combines PPO and Evolution Strategy in a federated setting
- **Environment Wrappers**: Custom wrappers to adapt GymDSSAT-PDI environment for RL algorithms
- **Comprehensive Logging**: Training progress tracking and model checkpointing
- **Policy Evaluation**: Built-in evaluation tools for trained policies

## Repository Structure

```
Codes/
├── PPO/                   # Proximal Policy Optimization (Centralized)
│   ├── ppo_main.py        # Main training script
│   ├── ppo.py             # PPO algorithm implementation
│   ├── network.py         # Neural network architecture
│   ├── env_wrappers.py    # Environment wrapper functions
│   ├── ppo_eval_policy.py # Policy evaluation script
│   ├── logs/              # Training logs
│   └── models/            # Saved model checkpoints
│
├── PPO_Fed/               # PPO Federated Learning
│   ├── ppo_fed_main.py    # Main federated training script
│   ├── ppo_fed_server.py  # Federated server implementation
│   ├── ppo_fed_client.py  # Federated client implementation
│   ├── ppo_fed.py         # PPO algorithm for federated setting
│   ├── network.py         # Neural network architecture
│   ├── env_wrappers.py    # Environment wrappers
│   ├── logs/              # Training logs
│   └── models/            # Saved model checkpoints
│
├── ES_all/                # Evolution Strategy (Centralized)
│   ├── es_all_main.py     # Main training script
│   ├── es_algo.py         # Evolution Strategy algorithm
│   ├── network.py         # Neural network architecture
│   ├── env_wrappers.py    # Environment wrappers
│   ├── es_all_policy_eval.py # Policy evaluation
│   ├── logs/              # Training logs
│   └── models/            # Saved model checkpoints
│
├── ES_Fed/                # Evolution Strategy Federated Learning
│   ├── es_fed_main.py     # Main federated training script
│   ├── es_fed_server.py   # Federated server implementation
│   ├── es_fed_client.py   # Federated client implementation
│   ├── es_fed_es_algo.py  # ES algorithm for federated setting
│   ├── network.py         # Neural network architecture
│   ├── env_wrappers.py    # Environment wrappers
│   ├── logs/              # Training logs
│   └── models/            # Saved model checkpoints
│
├── EvoFed/                # Evolutionary Federated (PPO + ES Hybrid)
│   ├── main_evofed.py     # Main training script
│   ├── server_evofed.py    # Federated server implementation
│   ├── client_evofed.py   # Federated client implementation
│   ├── ppo_evofed.py      # PPO component
│   ├── network.py         # Neural network architecture
│   ├── env_wrappers.py    # Environment wrappers
│   ├── evaluation_evofed.py # Evaluation script
│   ├── logs/              # Training logs
│   └── models/            # Saved model checkpoints
│
└── env_wrapper_explanation.txt  # Detailed explanation of environment wrappers
```

## Environment

The project uses **GymDSSAT-PDI** (`GymDssatPdi-v0`), a crop simulation environment that models:
- Crop growth dynamics
- Soil conditions
- Weather patterns
- Fertilizer application effects

### Environment Wrappers

Custom wrappers are used to adapt the environment for RL algorithms:

1. **DictToArrayWrapper**: Converts dictionary observations to 1D vectors
2. **TimeLimit**: Sets maximum episode length (180 steps)
3. **RewardTupleAdapter**: Handles multi-objective rewards (yield and leaching)
4. **ActionToDictWrapper**: Converts discrete actions to environment dictionary format
5. **NormalizationWrapper**: Normalizes observations and rewards for stable training

See `env_wrapper_explanation.txt` for detailed documentation.

### Action Space
- **Type**: Discrete(25)
- **Actions**: Represent combinations of fertilizer application rates (`anfer`, `amir`)

### Observation Space
- **Type**: Box (1D normalized vector)
- **Components**: Crop state variables (DAP, temperature, soil water, etc.)

### Reward Structure
- **Primary reward**: Crop yield (maximize)
- **Secondary metric**: Nitrogen leaching (minimize, tracked in info dict)

## Installation

### Prerequisites
- Python 3.10.2
- PyTorch 2.3.0+cpu
- Gym 0.21.0
- gym-dssat-pdi 0.0.5        
- NumPy 1.24.1
- TensorBoard

### Setup

1. Follow the instruction on Gym-Dssat website to install Ubuntu 22.04 (Jammy Jellyfish) Package
    https://rgautron.gitlabpages.inria.fr/gym-dssat-docs/Installation/packages.html#id1


2. Install required packages:

Activate the environment using: 
```bash
source /opt/gym_dssat_pdi/bin/activate
```

Then install the libraries in prerequisities in this environment.

3. Some changes in the codes:
Navigate to: /opt/gym_dssat_pdi/lib/python3.10/site-packages/gym_dssat_pdi/envs/configs

Then run:
```bash
sudo nano rewards.py
```

Here we need to modify the current all_reward() function with:
``` python
def all_reward(_previous_state, _next_state, _history, _cultivar):

    weights = {
                "maize"  : {"w1":0.2, "w2":1.0, "w3":1.1, "w4":5.0},
              }

    w1 = weights[_cultivar]["w1"]
    w2 = weights[_cultivar]["w2"]
    w3 = weights[_cultivar]["w3"]
    w4 = weights[_cultivar]["w4"]
    last_action_N = _history['action'][-1]['anfer']
    last_action_W = _history['action'][-1]['amir']

    if _next_state:

        N_leach = _next_state['tleachd']
        reward = - (w2 * last_action_N) - (w3 * last_action_W) - (w4 * N_leach)

        return [reward, N_leach]

    N_leach = _previous_state['tleachd']
    Yield = _previous_state['grnwt']
    last_reward = (w1 * Yield) - (w2 * last_action_N) - (w3 * last_action_W) - (w4 * N_leach)

    return [last_reward, N_leach]
```

This is a modified version of the reward function that I use with my code. Every time we need to change the reward function(RF1-RF5) we can come here and change weights.

Then navigate to /opt/gym_dssat_pdi/lib/python3.10/site-packages/gym_dssat_pdi/envs/

Run:
```bash
sudo nano dssat_pdi.py
```
Here comment out line 433 and 434.
``` python
# if done:
#     return None, self.reward, self.done, None
```
Thats all the changes we needed to make.

4. Clone the repository:
```bash
git clone https://github.com/TariqUmar/DL_Project_Crop_Managment.git
cd DL_Project_Crop_Managment/Codes
```

## Usage

### Training a any Model

```bash
cd model_folder
python model_main.py
```

Modify hyperparameters in `model_main.py` and the base algorithim file as modify these directories.
- `SEED`: Random seed for reproducibility
- `LOG_DIR`: Directory for training logs
- `SAVE_DIR`: Directory for model checkpoints

## Project Organization

Each algorithm directory contains:
- **Main script**: Entry point for training (`*_main.py`)
- **Algorithm implementation**: Core RL algorithm code
- **Network architecture**: Neural network definitions (`network.py`)
- **Environment wrappers**: Custom environment adaptations (`env_wrappers.py`)
- **Logs directory**: Training progress and TensorBoard logs
- **Models directory**: Saved policy checkpoints and normalization statistics

## Results

Training results are saved in the `logs/` directory of each algorithm:
- TensorBoard event files for visualization
- CSV files with training progress metrics
- Model checkpoints in `models/` directory

To visualize training progress:
```bash
tensorboard --logdir=<algorithm_directory>/logs
```

## Experimental Configurations

The project includes multiple experimental runs (RF1-RF5) with different hyperparameter configurations. 

## Notes

- Centralized algorithms use feedforward neural networks with architecture: `25 → 256 → 256 → 256 → 25`
- Federated algorithms use feedforward neural networks with architecture: `25 → 50 → 50 → 50 → 25`
- Environment normalization statistics are saved with each model for consistent evaluation
- Federated approaches use 5 clients with different random seeds for environment diversity
- Maximum episode length is set to 180 steps

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Authors

**Tariq Umar**
- GitHub: [@TariqUmar](https://github.com/TariqUmar)

**Mehraj Chhetri**

## Acknowledgments

- GymDSSAT-PDI environment developers
- Deep Learning course (EE5260) at Iowa State University

---