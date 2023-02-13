"""
Patches for gym 0.26+ so RL Zoo3 keeps working as before
(notably TimeLimit wrapper and Pybullet envs)
"""
from typing import Any, Dict

import numpy as np

# Deprecation warning with gym 0.26 and numpy 1.24
np.bool8 = np.bool_  # type: ignore[attr-defined]

import gym  # noqa: E402
import gymnasium  # noqa: E402


class PatchedRegistry(dict):
    """
    gym.envs.registration.registry
    is now a dictionnary and no longer an EnvRegistry() object.
    """

    @property
    def env_specs(self) -> Dict[str, Any]:
        return self


class PatchedTimeLimit(gymnasium.wrappers.TimeLimit):
    """
    See https://github.com/openai/gym/issues/3102
    and https://github.com/Farama-Foundation/Gymnasium/pull/101:
    keep the behavior as before and provide additionnal info
    that the episode reached a timeout, but only
    when the episode is over because of that.
    """

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        self._elapsed_steps += 1

        if self._elapsed_steps >= self._max_episode_steps:
            done = truncated or terminated
            # TimeLimit.truncated key may have been already set by the environment
            # do not overwrite it
            # only set it when the episode is not over for other reasons
            episode_truncated = not done or info.get("TimeLimit.truncated", False)
            info["TimeLimit.truncated"] = episode_truncated
            # truncated may have been set by the env too
            truncated = truncated or episode_truncated

        return observation, reward, terminated, truncated, info


# Use gym as base class otherwise the patch_env won't work
class PatchedGymTimeLimit(gym.wrappers.TimeLimit):
    """
    See https://github.com/openai/gym/issues/3102
    and https://github.com/Farama-Foundation/Gymnasium/pull/101:
    keep the behavior as before and provide additionnal info
    that the episode reached a timeout, but only
    when the episode is over because of that.
    """

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        self._elapsed_steps += 1

        if self._elapsed_steps >= self._max_episode_steps:
            done = truncated or terminated
            # TimeLimit.truncated key may have been already set by the environment
            # do not overwrite it
            # only set it when the episode is not over for other reasons
            episode_truncated = not done or info.get("TimeLimit.truncated", False)
            info["TimeLimit.truncated"] = episode_truncated
            # truncated may have been set by the env too
            truncated = truncated or episode_truncated

        return observation, reward, terminated, truncated, info


# Patch Gym registry (for Pybullet)
patched_registry = PatchedRegistry()
patched_registry.update(gym.envs.registration.registry)
gym.envs.registry = patched_registry
gym.envs.registration.registry = patched_registry
# Patch gym TimeLimit
gym.wrappers.TimeLimit = PatchedGymTimeLimit  # type: ignore[misc]
gym.wrappers.time_limit.TimeLimit = PatchedGymTimeLimit  # type: ignore[misc]
gym.envs.registration.TimeLimit = PatchedGymTimeLimit  # type: ignore[misc]
# Patch Gymnasium TimeLimit
gymnasium.wrappers.TimeLimit = PatchedTimeLimit  # type: ignore[misc]
gymnasium.wrappers.time_limit.TimeLimit = PatchedTimeLimit  # type: ignore[misc]
gymnasium.envs.registration.TimeLimit = PatchedTimeLimit  # type: ignore[misc]
