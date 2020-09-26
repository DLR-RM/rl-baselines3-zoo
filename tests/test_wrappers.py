import gym
import pybullet_envs  # noqa: F401
import pytest
from stable_baselines3.common.env_checker import check_env

from utils.utils import get_wrapper_class
from utils.wrappers import ActionNoiseWrapper, DelayedRewardWrapper, HistoryWrapper, TimeFeatureWrapper


def test_wrappers():
    env = gym.make("HalfCheetahBulletEnv-v0")
    env = DelayedRewardWrapper(env)
    env = ActionNoiseWrapper(env)
    env = HistoryWrapper(env)
    env = TimeFeatureWrapper(env)
    check_env(env)


@pytest.mark.parametrize(
    "env_wrapper",
    [
        None,
        {"utils.wrappers.HistoryWrapper": dict(horizon=2)},
        [{"utils.wrappers.HistoryWrapper": dict(horizon=3)}, "utils.wrappers.TimeFeatureWrapper"],
    ],
)
def test_get_wrapper(env_wrapper):
    env = gym.make("HalfCheetahBulletEnv-v0")
    hyperparams = {"env_wrapper": env_wrapper}
    wrapper_class = get_wrapper_class(hyperparams)
    if env_wrapper is not None:
        env = wrapper_class(env)
    check_env(env)
