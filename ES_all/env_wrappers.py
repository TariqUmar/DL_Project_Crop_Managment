import gym
import gym_dssat_pdi
import numpy as np
import json

from gym.wrappers import TimeLimit

# Defining tuples for the actions and the observation keys
NITROGEN_ACTIONS = (0, 40, 80, 120, 160)
IRRIGATION_ACTIONS = (0, 6, 12, 18, 24)
OBS_SCALAR_KEYS = (
    "cumsumfert", "dap", "dtt", "ep", "grnwt", "istage", "nstres", "rtdep",
    "srad", "swfac", "tmax", "topwt", "totir", "vstage", "wtdep", "xlai", "sw"
)

class DictToArrayWrapper(gym.ObservationWrapper):
    """
    Convert a Dict observation into a single 1D float32 vector.
    Uses the key_order.

    Observation space is a Box(total,), where total is the sum of the sizes of the keys.
    The sizes are the product of the shape of the observation space for each key.
    The zero vectors are used to fill in the missing keys.
    The parts are concatenated along the first axis (axis=0) to form the final observation.
    """

    def __init__(self, env: gym.Env, key_order=OBS_SCALAR_KEYS) -> None:

        super().__init__(env)

        self.key_order = list(key_order)

        # Size of each observation part - the shapes are either () or (9,)
        self._sizes = {k: int(np.prod(env.observation_space.spaces[k].shape)) for k in self.key_order}     # np.prod(()) = 1, np.prod((9,)) = 9
        total = int(sum(self._sizes.values()))

        # Observation space is a Box(total,), where total is the sum of the sizes of the keys.
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(total,), dtype=np.float32)

        # prebuild zero vectors for each key in case the environment returns a partial dict
        self._zeros = {k: np.zeros(self._sizes[k], dtype=np.float32) for k in self.key_order}

    def observation(self, obs: dict | None) -> np.ndarray:

        # handle terminal step: env may return obs=None
        if obs is None:
            return np.concatenate([self._zeros[k] for k in self.key_order], axis=0)

        # handle partial dicts: missing keys -> zeros
        parts = []
        for k in self.key_order:
            v = obs[k] if k in obs else self._zeros[k]
            parts.append(np.asarray(v, dtype=np.float32).reshape(-1))
        return np.concatenate(parts, axis=0)

class ActionToDictWrapper(gym.ActionWrapper):
    """
    Map Discrete(25) index -> [anfer, amir] using 5x5 grids.
    Returns a dictionary with the following keys:
        - anfer: float
        - amir: float
    """
    def __init__(self, env: gym.Env, anfer_values = NITROGEN_ACTIONS, amir_values = IRRIGATION_ACTIONS) -> None:
        super().__init__(env)
        self.anfer_values = np.array(anfer_values, dtype=np.float32)
        self.amir_values = np.array(amir_values, dtype=np.float32)
        self._m = len(self.amir_values)
        self.action_space = gym.spaces.Discrete(len(self.anfer_values) * len(self.amir_values))

    def action(self, action: int) -> dict:
        # Convert action to integer
        a = int(np.asarray(action).item())
        # Calculate row and column indices
        i = int(a) // self._m    # row index
        j = int(a) % self._m     # column index
        # Return the corresponding N/W values
        return {"anfer": float(self.anfer_values[i]),
                "amir":  float(self.amir_values[j])}


class RewardTupleAdapter(gym.Wrapper):
    """
    In the environment settings, I modified the reward function to return a tuple of (main, leach).
    This wrapper will return the main reward and stash the leach reward into info['leach'].
    Main reward will used to train the DQN/PPO/DDPG agent while daily leaching is used to monitor the performance of the agent.

    This wrapper will accumulate the total amount of nitrogen and water applied across an episode.
    Accumulates across an episode:
      - N_total: sum of anfer doses
      - W_total: sum of amir amounts
      - leach_total: sum of info['leach'] if present
      - yield: last seen info['grnwt'] if present
    Emits info['episode_metrics'] at done.
    """

    def __init__(self, env: gym.Env, yield_index: int = 4) -> None:
        super().__init__(env)
        self._yield_index = yield_index

        self._reset_stats()
        self._yield_from_prev: float | None = None

    def _reset_stats(self):
        self._N_total = 0.0
        self._W_total = 0.0
        self._leach_total = 0.0
        self._yield_final = None

    def reset(self, **kwargs) -> np.ndarray:
        obs = self.env.reset(**kwargs)
        self._reset_stats()

        # obs should already be a flat vector (from DictToArrayWrapper)
        arr = np.asarray(obs).reshape(-1)
        if arr.size > self._yield_index:
            self._yield_from_prev = float(arr[self._yield_index])

        # return the initial observation
        return obs

    def step(self, action: dict) -> tuple[np.ndarray, float, bool, dict]:
        # Accumulate N and W from the dict action (default to 0.0 if missing)
        self._N_total += float(action.get('anfer', 0.0))
        self._W_total += float(action.get('amir', 0.0))

        # Yield before taking this step
        yield_before = self._yield_from_prev

        # Step the environment
        obs, rew, done, info = self.env.step(action)

        # Handle reward = (main, leach) or plain float
        leach_val = None
        if isinstance(rew, (tuple, list)) and len(rew) >= 2:
            main_rew, leach_val = rew[0], rew[1]
            rew = float(main_rew)
        else:
            rew = float(rew)

        # If env didn't provide leach in the tuple, fall back to info or 0.0
        if leach_val is None:
            leach_val = float(info.get('leach', 0.0))
        else:
            leach_val = float(leach_val)

        info['leach'] = leach_val
        self._leach_total += leach_val

        # Update yield buffer from the NEW obs (if any)
        if obs is not None:
            arr = np.asarray(obs, dtype=np.float32).reshape(-1)
            if arr.size > self._yield_index:
                self._yield_from_prev = float(arr[self._yield_index])
        # If obs is None, keep previous yield_from_prev as-is

        # On episode end, emit summary metrics
        if done:
            yfinal = yield_before if yield_before is not None else float("nan")
            info['episode_metrics'] = {
                'N_total': self._N_total,
                'W_total': self._W_total,
                'leach_total': self._leach_total,
                'yield': yfinal,
            }

        return obs, rew, done, info


class RunningMeanStd:
    " Tracking mean and variance with numerically stable updates"

    def __init__(self, shape: tuple, epsilon: float = 1e-8):
        self.mean = np.zeros(shape, dtype=np.float32)
        self.var = np.ones(shape, dtype=np.float32)
        self.count = epsilon
        self.epsilon = epsilon
        
    def update(self, x: np.ndarray):
        x = np.asarray(x, dtype=np.float32)
 
        # add a batch dimension so we can take mean/var over axis=0.
        if x.ndim == len(self.mean.shape):
            x = x[None, ...]                    # scalar -> (1,), (25,) -> (1,25), etc.

        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self._update_from_moments(batch_mean, batch_var, batch_count)
        
    def _update_from_moments(self, batch_mean: np.ndarray, batch_var: np.ndarray, batch_count: int):
        mean_diff = batch_mean - self.mean
        total_counts = self.count + batch_count

        new_mean = self.mean + mean_diff * batch_count / total_counts
        
        sum_sq_dev_old = self.var * self.count
        sum_sq_dev_new = batch_var * batch_count
        new_var = (sum_sq_dev_old + sum_sq_dev_new + np.square(mean_diff) * self.count * batch_count / total_counts) / total_counts
        
        new_count = total_counts
        
        self.mean, self.var, self.count = new_mean, new_var, new_count

    def state_dict(self):
        return {'mean': self.mean.tolist(), 'var': self.var.tolist(), 'count': self.count}

    def load_state_dict(self, state_dict: dict):
        self.mean = np.array(state_dict['mean'], dtype=np.float32)
        self.var = np.array(state_dict['var'], dtype=np.float32)
        self.count = state_dict['count']


class NormalizationWrapper(gym.Wrapper):
    """
    Normalize the observations and rewards.
    Assumes env.observation_space is a flat Box (use DictToArrayWrapper first).
    """

    def __init__(self, env: gym.Env, norm_obs: bool = True, norm_rew: bool = True,
                 clip_obs: float = 10.0, clip_rew: float = 10.0, gamma: float = 0.99, eps: float = 1e-8):

        super().__init__(env)

        assert isinstance(env.observation_space, gym.spaces.Box), \
            "NormalizationWrapper expects flattened Box observations (use DictToArrayWrapper earlier)."

        self.norm_obs = norm_obs
        self.norm_rew = norm_rew
        self.clip_obs = float(clip_obs)
        self.clip_rew = float(clip_rew)
        self.gamma = float(gamma)
        self.eps = float(eps)
        
        obs_space = env.observation_space.shape
        self.obs_rms = RunningMeanStd(shape=obs_space, epsilon=self.eps)       # Observation normalization (vector)
        self.ret_rms = RunningMeanStd(shape=(), epsilon=self.eps)              # Return normalization (scalar)
        
        self.training = True
        self._ret = 0.0           # Discounted return accumulator
        self._raw_ep_return = 0.0
        

    def _normalize_obs(self, obs: np.ndarray) -> np.ndarray:
        if not self.norm_obs or obs is None:
            return obs
        
        if self.training:
            self.obs_rms.update(obs)
        
        std = np.sqrt(self.obs_rms.var + self.eps)
        obs = (obs - self.obs_rms.mean) / std

        return np.clip(obs, -self.clip_obs, self.clip_obs).astype(np.float32)
    
    def _normalize_rew(self, rew: float) -> float:
        if not self.norm_rew:
            return rew

        self._ret = self._ret * self.gamma + rew
        if self.training:
            self.ret_rms.update(self._ret)

        std = np.sqrt(self.ret_rms.var + self.eps)
        rew = rew / std
        return np.clip(rew, -self.clip_rew, self.clip_rew).astype(np.float32)

    def reset(self, **kwargs) -> np.ndarray:
        obs = self.env.reset(**kwargs)
        self._ret = 0.0
        self._raw_ep_return = 0.0
        return self._normalize_obs(obs)
    
    def step(self, action: dict) -> tuple[np.ndarray, float, bool, dict]:
        # 'action' is whatever the agent passes (Discrete int in our setup).
        obs, rew, done, info = self.env.step(action)

        raw_rew = float(rew)
        self._raw_ep_return += raw_rew

        obs = self._normalize_obs(obs)
        rew = self._normalize_rew(raw_rew)

        info['raw_reward'] = raw_rew
        
        if done:            # Reset the return accumulator if the episode is done
            self._ret = 0.0
            info['raw_ep_return'] = self._raw_ep_return
            self._raw_ep_return = 0.0
            
        return obs, rew, done, info

    # Helper for logging and evaluation
    def denormalize_obs(self, obs: np.ndarray) -> np.ndarray:
        std = np.sqrt(self.obs_rms.var + self.eps)
        return (obs * std + self.obs_rms.mean).astype(np.float32)

    def denormalize_rew(self, rew: float) -> float:
        std = np.sqrt(self.ret_rms.var + self.eps)
        return (rew * std)

    def save(self, path: str):
        state_dictictionary = {
                    'obs_rms': self.obs_rms.state_dict(),
                    'ret_rms': self.ret_rms.state_dict(),
                    'norm_obs': self.norm_obs,
                    'norm_rew': self.norm_rew,
                    'clip_obs': self.clip_obs,
                    'clip_rew': self.clip_rew,
                    'gamma': self.gamma,
                    'eps': self.eps,
                }
        
        with open(path, 'w') as f:
            json.dump(state_dictictionary, f)

    def load(self, path: str):
        with open(path, 'r') as f:
            state_dict = json.load(f)
        self.obs_rms.load_state_dict(state_dict['obs_rms'])
        self.ret_rms.load_state_dict(state_dict['ret_rms'])
        self.norm_obs = state_dict['norm_obs']
        self.norm_rew = state_dict['norm_rew']
        self.clip_obs = state_dict['clip_obs']
        self.clip_rew = state_dict['clip_rew']
        self.gamma = state_dict['gamma']
        self.eps = state_dict['eps']


def make_env(environment_arguments: dict) -> gym.Env:
    '''
    Make the environment.
    Args:
        environment_arguments (dict): The environment arguments.
    '''
    env_args = {'mode': environment_arguments['mode'], 'seed': environment_arguments['seed'], 'random_weather': environment_arguments['random_weather']}
    env_id = environment_arguments['env_id']
    max_episode_steps = environment_arguments['max_episode_steps']

    base_env = gym.make(env_id, **env_args)                                 # Base environment
    time_limit = TimeLimit(base_env, max_episode_steps=max_episode_steps)   # Putting an upper limit on the number of steps in an episode
    dict_to_array = DictToArrayWrapper(time_limit)                          # Converting the dict observation to a 1D array
    reward_tuple_adapter = RewardTupleAdapter(dict_to_array)                # Adapting the reward function to return a tuple of (main, leach)
                                                                            # Accumulating the total amount of nitrogen and water applied across an episode
    action_to_dict = ActionToDictWrapper(reward_tuple_adapter)              # Mapping the discrete action to a dict action


    env = NormalizationWrapper(
        action_to_dict,
        norm_obs=True,
        norm_rew=True,
        clip_obs=10.0,
        clip_rew=10.0,
        gamma=0.99,
        eps=1e-8,
    )
    return env


if __name__ == "__main__":

    environment_arguments = {
        'mode': 'all',
        'seed': 123,
        'random_weather': True,
        'env_id': 'GymDssatPdi-v0',
        'max_episode_steps': 180,
    }

    env = make_env(environment_arguments)
    print(env)

    obs = env.reset()

    print(env.action_space.n)
    print(env.observation_space.shape[0])

    env.close()

    print("You are running the env_wrappers.py file. Run main.py is meant to be run.")