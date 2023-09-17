from typing import Callable, Optional

import gymnasium as gym
from gymnasium.envs.registration import register

from rl_zoo3.wrappers import MaskVelocityWrapper

try:
    import pybullet_envs_gymnasium  # pytype: disable=import-error
except ImportError:
    pass

try:
    import highway_env  # pytype: disable=import-error
except ImportError:
    pass
else:
    # hotfix for highway_env
    import numpy as np

    np.float = np.float32  # type: ignore[attr-defined]

try:
    import custom_envs  # pytype: disable=import-error
except ImportError:
    pass

try:
    import gym_donkeycar  # pytype: disable=import-error
except ImportError:
    pass

try:
    import panda_gym  # pytype: disable=import-error
except ImportError:
    pass

try:
    import rocket_lander_gym  # pytype: disable=import-error
except ImportError:
    pass

try:
    import minigrid  # pytype: disable=import-error
except ImportError:
    pass


# Register no vel envs
def create_no_vel_env(env_id: str) -> Callable[[Optional[str]], gym.Env]:
    def make_env(render_mode: Optional[str] = None) -> gym.Env:
        env = gym.make(env_id, render_mode=render_mode)
        env = MaskVelocityWrapper(env)
        return env

    return make_env


for env_id in MaskVelocityWrapper.velocity_indices.keys():
    name, version = env_id.split("-v")
    register(
        id=f"{name}NoVel-v{version}",
        entry_point=create_no_vel_env(env_id),  # type: ignore[arg-type]
    )
