import gym
import numpy as np
from gym import spaces


class TestEnv(gym.Env):
    def __init__(self):
        super().__init__()

        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(low=0, high=1, shape=(2,), dtype=int)

    def step(self, action):
        return np.zeros(2), 0.0, False, {}

    def reset(self):
        return np.zeros(2)

    def render(self, mode="human"):
        pass
