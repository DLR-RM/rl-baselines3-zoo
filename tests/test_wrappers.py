import gymnasium as gym
import numpy as np
import pytest
import stable_baselines3 as sb3
from sb3_contrib.common.wrappers import TimeFeatureWrapper
from stable_baselines3 import A2C
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.env_util import DummyVecEnv

import rl_zoo3.import_envs
import rl_zoo3.wrappers
from rl_zoo3.utils import SimpleLinearSchedule, get_wrapper_class, linear_schedule
from rl_zoo3.wrappers import ActionNoiseWrapper, DelayedRewardWrapper, HistoryWrapper


def test_wrappers():
    env = gym.make("Ant-v4")
    env = DelayedRewardWrapper(env)
    env = ActionNoiseWrapper(env)
    env = HistoryWrapper(env)
    env = TimeFeatureWrapper(env)
    check_env(env)


@pytest.mark.parametrize(
    "env_wrapper",
    [
        None,
        {"rl_zoo3.wrappers.HistoryWrapper": dict(horizon=2)},
        [{"rl_zoo3.wrappers.HistoryWrapper": dict(horizon=3)}, "rl_zoo3.wrappers.TimeFeatureWrapper"],
        [{rl_zoo3.wrappers.HistoryWrapper: dict(horizon=3)}, "rl_zoo3.wrappers.TimeFeatureWrapper"],
    ],
)
def test_get_wrapper(env_wrapper):
    env = gym.make("Ant-v4")
    hyperparams = {"env_wrapper": env_wrapper}
    wrapper_class = get_wrapper_class(hyperparams)
    if env_wrapper is not None:
        env = wrapper_class(env)
    check_env(env)


@pytest.mark.parametrize(
    "vec_env_wrapper",
    [
        None,
        {"stable_baselines3.common.vec_env.VecFrameStack": dict(n_stack=2)},
        {sb3.common.vec_env.VecFrameStack: dict(n_stack=2)},
        [{"stable_baselines3.common.vec_env.VecFrameStack": dict(n_stack=3)}, "stable_baselines3.common.vec_env.VecMonitor"],
    ],
)
def test_get_vec_env_wrapper(vec_env_wrapper):
    env = gym.make("Ant-v4")
    env = DummyVecEnv([lambda: env])
    hyperparams = {"vec_env_wrapper": vec_env_wrapper}
    wrapper_class = get_wrapper_class(hyperparams, "vec_env_wrapper")
    if wrapper_class is not None:
        env = wrapper_class(env)
    A2C("MlpPolicy", env).learn(16)


def test_linear_schedule():
    schedule = linear_schedule(100)
    assert isinstance(schedule, SimpleLinearSchedule)
    assert np.allclose(schedule(1.0), 100.0)
    assert np.allclose(schedule(0.5), 50.0)
    assert np.allclose(schedule(0.0), 0.0)
