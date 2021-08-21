try:
    import pybullet_envs  # pytype: disable=import-error
except ImportError:
    pybullet_envs = None

try:
    import highway_env  # pytype: disable=import-error
except ImportError:
    highway_env = None

try:
    import neck_rl  # pytype: disable=import-error
except ImportError:
    neck_rl = None

try:
    import mocca_envs  # pytype: disable=import-error
except ImportError:
    mocca_envs = None

try:
    import custom_envs  # pytype: disable=import-error
except ImportError:
    custom_envs = None

try:
    import gym_donkeycar  # pytype: disable=import-error
except ImportError:
    gym_donkeycar = None

try:
    import panda_gym  # pytype: disable=import-error
except ImportError:
    panda_gym = None


"""Set up gym interface for simulation environments."""
import gym
from gym.envs.registration import registry, make, spec

def register(env_id, *args, **kvargs):
  if env_id in registry.env_specs:
    return
  else:
    return gym.envs.registration.register(env_id, *args, **kvargs)


register(
    env_id='A1GymEnv-v0',
    entry_point='blind_walking.envs.gym_envs:A1GymEnv',
    max_episode_steps=2000,
    reward_threshold=2000.0,
)
