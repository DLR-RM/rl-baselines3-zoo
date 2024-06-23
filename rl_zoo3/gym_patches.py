"""
Patches for gym 0.26+ so RL Zoo3 keeps working as before
(notably TimeLimit wrapper and Pybullet envs)
"""

import numpy as np

# Deprecation warning with gym 0.26 and numpy 1.24
np.bool8 = np.bool_  # type: ignore[attr-defined]

import gymnasium  # noqa: E402


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


# Patch Gymnasium TimeLimit
gymnasium.wrappers.TimeLimit = PatchedTimeLimit  # type: ignore[misc]
gymnasium.wrappers.time_limit.TimeLimit = PatchedTimeLimit  # type: ignore[misc]
gymnasium.envs.registration.TimeLimit = PatchedTimeLimit  # type: ignore[misc,attr-defined]
