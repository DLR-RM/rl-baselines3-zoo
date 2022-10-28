from typing import Optional

import numpy as np
from envpool.python.protocol import EnvPool
from stable_baselines3.common.vec_env import VecEnvWrapper
from stable_baselines3.common.vec_env.base_vec_env import VecEnvObs, VecEnvStepReturn


class EnvPoolAdapter(VecEnvWrapper):
    """
    Convert EnvPool object to a Stable-Baselines3 (SB3) VecEnv.

    :param venv: The envpool object.
    """

    def __init__(self, venv: EnvPool):
        # Retrieve the number of environments from the config
        venv.num_envs = venv.spec.config.num_envs
        super().__init__(venv=venv)

    def step_async(self, actions: np.ndarray) -> None:
        self.actions = actions

    def reset(self) -> VecEnvObs:
        return self.venv.reset()

    def seed(self, seed: Optional[int] = None) -> None:
        # You can only seed EnvPool env by calling envpool.make()
        pass

    def step_wait(self) -> VecEnvStepReturn:
        obs, rewards, dones, info_dict = self.venv.step(self.actions)
        infos = []
        # Convert dict to list of dict
        # and add terminal observation
        for i in range(self.num_envs):
            infos.append({key: info_dict[key][i] for key in info_dict.keys() if isinstance(info_dict[key], np.ndarray)})
            if dones[i]:
                infos[i]["terminal_observation"] = obs[i]
                obs[i] = self.venv.reset(np.array([i]))

        return obs, rewards, dones, infos
