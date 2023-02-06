import inspect
from typing import Any, Iterable, List, Optional, Sequence, Type, Union

import gym
import numpy as np
from envpool.python.protocol import EnvPool
from gym import spaces
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.vec_env.base_vec_env import VecEnvObs, VecEnvStepReturn

# Used when we want to access one or more VecEnv
VecEnvIndices = Union[None, int, Iterable[int]]


class EnvPoolAdapter(VecEnv):
    """
    Vectorized environment base class

    :param venv: the vectorized environment to wrap
    :param observation_space: the observation space (can be None to load from venv)
    :param action_space: the action space (can be None to load from venv)
    """

    def __init__(self, venv: EnvPool) -> None:
        # Retrieve the number of environments from the config
        self.venv = venv
        action_space = venv.action_space
        observation_space = venv.observation_space
        # Tmp fix for https://github.com/DLR-RM/stable-baselines3/issues/1145
        for space in [observation_space, action_space]:
            if isinstance(space, spaces.Box) and space.dtype == np.float64:
                space.dtype = np.dtype(np.float32)

        super().__init__(
            num_envs=venv.spec.config.num_envs,
            observation_space=observation_space,
            action_space=action_space,
        )
        self.class_attributes = dict(inspect.getmembers(self.__class__))  # TODO: unused

    def reset(self) -> VecEnvObs:
        return self.venv.reset()

    def step_async(self, actions: np.ndarray) -> None:
        self.actions = actions

    def step_wait(self) -> VecEnvStepReturn:
        obs, rewards, dones, info_dict = self.venv.step(self.actions)
        infos = []
        # Convert dict to list of dict and add terminal observation
        for env_idx in range(self.num_envs):
            infos.append({key: info_dict[key][env_idx] for key in info_dict.keys() if isinstance(info_dict[key], np.ndarray)})
            if dones[env_idx]:
                infos[env_idx]["terminal_observation"] = obs[env_idx]
                obs[env_idx] = self.venv.reset(np.array([env_idx]))

        return obs, rewards, dones, infos

    def close(self) -> None:
        # No closing method in envpool
        return None

    def get_attr(self, attr_name: str, indices: VecEnvIndices = None) -> List[Any]:
        raise NotImplementedError()  # TODO: Does envpool support this?

    def set_attr(self, attr_name: str, value: Any, indices: VecEnvIndices = None) -> None:
        raise NotImplementedError()  # TODO: Does envpool support this?

    def env_method(self, method_name: str, *method_args, indices: VecEnvIndices = None, **method_kwargs) -> List[Any]:
        raise NotImplementedError()  # TODO: Does envpool support this?

    def env_is_wrapped(self, wrapper_class: Type[gym.Wrapper], indices: VecEnvIndices = None) -> List[bool]:
        return [False for _ in range(self.num_envs)]

    def get_images(self) -> Sequence[np.ndarray]:
        raise NotImplementedError()  # TODO: Does envpool support this?

    def render(self, mode: str = "human") -> Optional[np.ndarray]:
        raise NotImplementedError()  # TODO: Does envpool support this?

    def seed(self, seed: Optional[int] = None) -> Sequence[None]:  # type: ignore[override] # until SB3/#1318 is closed
        # You can only seed EnvPool env by calling envpool.make()
        return [None for _ in range(self.num_envs)]
