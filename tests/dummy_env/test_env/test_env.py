from typing import ClassVar

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class TestEnv(gym.Env):
    metadata: ClassVar[dict] = {"render_modes": ["human"], "render_fps": 4}
    __test__ = False

    def __init__(self, render_mode=None):
        super().__init__()

        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(low=0, high=1, shape=(2,), dtype=np.float32)

    def step(self, action):
        return self.observation_space.sample(), 0.0, self.np_random.random() > 0.5, False, {}

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.observation_space.seed(seed)
        return self.observation_space.sample(), {}

    def render(self, mode="human"):
        pass


if __name__ == "__main__":
    from gymnasium.utils.env_checker import check_env as gym_check
    from stable_baselines3.common.env_checker import check_env

    check_env(TestEnv())
    gym_check(TestEnv())
