from typing import Any, ClassVar, Optional, SupportsFloat

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from gymnasium.core import ObsType
from sb3_contrib.common.wrappers import TimeFeatureWrapper  # noqa: F401 (backward compatibility)
from stable_baselines3.common.type_aliases import GymResetReturn, GymStepReturn


class TruncatedOnSuccessWrapper(gym.Wrapper):
    """
    Reset on success and offsets the reward.
    Useful for GoalEnv.
    """

    def __init__(self, env: gym.Env, reward_offset: float = 0.0, n_successes: int = 1):
        super().__init__(env)
        self.reward_offset = reward_offset
        self.n_successes = n_successes
        self.current_successes = 0

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> GymResetReturn:
        self.current_successes = 0
        assert options is None, "Options not supported for now"
        return self.env.reset(seed=seed)

    def step(self, action) -> GymStepReturn:
        obs, reward, terminated, truncated, info = self.env.step(action)
        if info.get("is_success", False):
            self.current_successes += 1
        else:
            self.current_successes = 0
        # number of successes in a row
        truncated = truncated or self.current_successes >= self.n_successes
        reward = float(reward) + self.reward_offset
        return obs, reward, terminated, truncated, info

    def compute_reward(self, achieved_goal, desired_goal, info):
        reward = self.env.compute_reward(achieved_goal, desired_goal, info)
        return reward + self.reward_offset


class ActionNoiseWrapper(gym.Wrapper[ObsType, np.ndarray, ObsType, np.ndarray]):
    """
    Add gaussian noise to the action (without telling the agent),
    to test the robustness of the control.

    :param env:
    :param noise_std: Standard deviation of the noise
    """

    def __init__(self, env: gym.Env, noise_std: float = 0.1):
        super().__init__(env)
        self.noise_std = noise_std

    def step(self, action: np.ndarray) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        assert isinstance(self.action_space, spaces.Box)
        noise = np.random.normal(np.zeros_like(action), np.ones_like(action) * self.noise_std)
        noisy_action = np.clip(action + noise, self.action_space.low, self.action_space.high)
        return self.env.step(noisy_action)


class ActionSmoothingWrapper(gym.Wrapper):
    """
    Smooth the action using exponential moving average.

    :param env:
    :param smoothing_coef: Smoothing coefficient (0 no smoothing, 1 very smooth)
    """

    def __init__(self, env: gym.Env, smoothing_coef: float = 0.0):
        super().__init__(env)
        self.smoothing_coef = smoothing_coef
        self.smoothed_action = None
        # from https://github.com/rail-berkeley/softlearning/issues/3
        # for smoothing latent space
        # self.alpha = self.smoothing_coef
        # self.beta = np.sqrt(1 - self.alpha ** 2) / (1 - self.alpha)

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> GymResetReturn:
        self.smoothed_action = None
        assert options is None, "Options not supported for now"
        return self.env.reset(seed=seed)

    def step(self, action) -> GymStepReturn:
        if self.smoothed_action is None:
            self.smoothed_action = np.zeros_like(action)
        assert self.smoothed_action is not None
        self.smoothed_action = self.smoothing_coef * self.smoothed_action + (1 - self.smoothing_coef) * action
        return self.env.step(self.smoothed_action)


class DelayedRewardWrapper(gym.Wrapper):
    """
    Delay the reward by `delay` steps, it makes the task harder but more realistic.
    The reward is accumulated during those steps.

    :param env:
    :param delay: Number of steps the reward should be delayed.
    """

    def __init__(self, env: gym.Env, delay: int = 10):
        super().__init__(env)
        self.delay = delay
        self.current_step = 0
        self.accumulated_reward = 0.0

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> GymResetReturn:
        self.current_step = 0
        self.accumulated_reward = 0.0
        assert options is None, "Options not supported for now"
        return self.env.reset(seed=seed)

    def step(self, action) -> GymStepReturn:
        obs, reward, terminated, truncated, info = self.env.step(action)

        self.accumulated_reward += float(reward)
        self.current_step += 1

        if self.current_step % self.delay == 0 or terminated or truncated:
            reward = self.accumulated_reward
            self.accumulated_reward = 0.0
        else:
            reward = 0.0
        return obs, reward, terminated, truncated, info


class HistoryWrapper(gym.Wrapper[np.ndarray, np.ndarray, np.ndarray, np.ndarray]):
    """
    Stack past observations and actions to give an history to the agent.

    :param env:
    :param horizon: Number of steps to keep in the history.
    """

    def __init__(self, env: gym.Env, horizon: int = 2):
        assert isinstance(env.observation_space, spaces.Box)
        assert isinstance(env.action_space, spaces.Box)

        wrapped_obs_space = env.observation_space
        wrapped_action_space = env.action_space

        low_obs = np.tile(wrapped_obs_space.low, horizon)
        high_obs = np.tile(wrapped_obs_space.high, horizon)

        low_action = np.tile(wrapped_action_space.low, horizon)
        high_action = np.tile(wrapped_action_space.high, horizon)

        low = np.concatenate((low_obs, low_action))
        high = np.concatenate((high_obs, high_action))

        # Overwrite the observation space
        env.observation_space = spaces.Box(low=low, high=high, dtype=wrapped_obs_space.dtype)  # type: ignore[arg-type]

        super().__init__(env)

        self.horizon = horizon
        self.low_action, self.high_action = low_action, high_action
        self.low_obs, self.high_obs = low_obs, high_obs
        self.low, self.high = low, high
        self.obs_history = np.zeros(low_obs.shape, low_obs.dtype)
        self.action_history = np.zeros(low_action.shape, low_action.dtype)

    def _create_obs_from_history(self) -> np.ndarray:
        return np.concatenate((self.obs_history, self.action_history))

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> tuple[np.ndarray, dict]:
        # Flush the history
        self.obs_history[...] = 0
        self.action_history[...] = 0
        assert options is None, "Options not supported for now"
        obs, info = self.env.reset(seed=seed)
        self.obs_history[..., -obs.shape[-1] :] = obs
        return self._create_obs_from_history(), info

    def step(self, action) -> tuple[np.ndarray, SupportsFloat, bool, bool, dict]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        last_ax_size = obs.shape[-1]

        self.obs_history = np.roll(self.obs_history, shift=-last_ax_size, axis=-1)
        self.obs_history[..., -obs.shape[-1] :] = obs

        self.action_history = np.roll(self.action_history, shift=-action.shape[-1], axis=-1)
        self.action_history[..., -action.shape[-1] :] = action
        return self._create_obs_from_history(), reward, terminated, truncated, info


class HistoryWrapperObsDict(gym.Wrapper):
    """
    History Wrapper for dict observation.

    :param env:
    :param horizon: Number of steps to keep in the history.
    """

    def __init__(self, env: gym.Env, horizon: int = 2):
        assert isinstance(env.observation_space, spaces.Dict)
        assert isinstance(env.observation_space.spaces["observation"], spaces.Box)
        assert isinstance(env.action_space, spaces.Box)

        wrapped_obs_space = env.observation_space.spaces["observation"]
        wrapped_action_space = env.action_space

        low_obs = np.tile(wrapped_obs_space.low, horizon)
        high_obs = np.tile(wrapped_obs_space.high, horizon)

        low_action = np.tile(wrapped_action_space.low, horizon)
        high_action = np.tile(wrapped_action_space.high, horizon)

        low = np.concatenate((low_obs, low_action))
        high = np.concatenate((high_obs, high_action))

        # Overwrite the observation space
        env.observation_space.spaces["observation"] = spaces.Box(
            low=low,
            high=high,
            dtype=wrapped_obs_space.dtype,  # type: ignore[arg-type]
        )

        super().__init__(env)

        self.horizon = horizon
        self.low_action, self.high_action = low_action, high_action
        self.low_obs, self.high_obs = low_obs, high_obs
        self.low, self.high = low, high
        self.obs_history = np.zeros(low_obs.shape, low_obs.dtype)
        self.action_history = np.zeros(low_action.shape, low_action.dtype)

    def _create_obs_from_history(self) -> np.ndarray:
        return np.concatenate((self.obs_history, self.action_history))

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> tuple[dict[str, np.ndarray], dict]:
        # Flush the history
        self.obs_history[...] = 0
        self.action_history[...] = 0
        assert options is None, "Options not supported for now"
        obs_dict, info = self.env.reset(seed=seed)
        obs = obs_dict["observation"]
        self.obs_history[..., -obs.shape[-1] :] = obs

        obs_dict["observation"] = self._create_obs_from_history()

        return obs_dict, info

    def step(self, action) -> tuple[dict[str, np.ndarray], SupportsFloat, bool, bool, dict]:
        obs_dict, reward, terminated, truncated, info = self.env.step(action)
        obs = obs_dict["observation"]
        last_ax_size = obs.shape[-1]

        self.obs_history = np.roll(self.obs_history, shift=-last_ax_size, axis=-1)
        self.obs_history[..., -obs.shape[-1] :] = obs

        self.action_history = np.roll(self.action_history, shift=-action.shape[-1], axis=-1)
        self.action_history[..., -action.shape[-1] :] = action

        obs_dict["observation"] = self._create_obs_from_history()

        return obs_dict, reward, terminated, truncated, info


class FrameSkip(gym.Wrapper):
    """
    Return only every ``skip``-th frame (frameskipping)

    :param env: the environment
    :param skip: number of ``skip``-th frame
    """

    def __init__(self, env: gym.Env, skip: int = 4):
        super().__init__(env)
        self._skip = skip

    def step(self, action) -> GymStepReturn:
        """
        Step the environment with the given action
        Repeat action, sum reward.

        :param action: the action
        :return: observation, reward, terminated, truncated, information
        """
        total_reward = 0.0
        for _ in range(self._skip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += float(reward)
            if terminated or truncated:
                break

        return obs, total_reward, terminated, truncated, info


class MaskVelocityWrapper(gym.ObservationWrapper):
    """
    Gym environment observation wrapper used to mask velocity terms in
    observations. The intention is the make the MDP partially observable.
    Adapted from https://github.com/LiuWenlin595/FinalProject.

    :param env: Gym environment
    """

    # Supported envs
    velocity_indices: ClassVar[dict[str, np.ndarray]] = {
        "CartPole-v1": np.array([1, 3]),
        "MountainCar-v0": np.array([1]),
        "MountainCarContinuous-v0": np.array([1]),
        "Pendulum-v1": np.array([2]),
        "LunarLander-v2": np.array([2, 3, 5]),
        "LunarLanderContinuous-v2": np.array([2, 3, 5]),
    }

    def __init__(self, env: gym.Env):
        super().__init__(env)

        assert env.unwrapped.spec is not None
        env_id: str = env.unwrapped.spec.id
        # By default no masking
        self.mask = np.ones_like(env.observation_space.sample())
        try:
            # Mask velocity
            self.mask[self.velocity_indices[env_id]] = 0.0
        except KeyError as e:
            raise NotImplementedError(f"Velocity masking not implemented for {env_id}") from e

    def observation(self, observation: np.ndarray) -> np.ndarray:
        return observation * self.mask
