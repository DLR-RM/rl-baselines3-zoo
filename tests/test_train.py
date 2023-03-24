import os
import subprocess

import pytest


def _assert_eq(left, right):
    assert left == right, f"{left} != {right}"


N_STEPS = 100

ALGOS = ("ppo", "a2c", "dqn")
# 'BreakoutNoFrameskip-v4'
ENV_IDS = ("CartPole-v1",)

experiments = {}

for algo in ALGOS:
    for env_id in ENV_IDS:
        experiments[f"{algo}-{env_id}"] = (algo, env_id)

# Test for vecnormalize and frame-stack
experiments["ppo-BipedalWalkerHardcore-v3"] = ("ppo", "BipedalWalkerHardcore-v3")
# Test for SAC
experiments["sac-Pendulum-v1"] = ("sac", "Pendulum-v1")
# for TD3
experiments["td3-Pendulum-v1"] = ("td3", "Pendulum-v1")
# for DDPG
experiments["ddpg-Pendulum-v1"] = ("ddpg", "Pendulum-v1")


@pytest.mark.parametrize("experiment", experiments.keys())
def test_train(tmp_path, experiment):
    algo, env_id = experiments[experiment]
    args = ["-n", str(N_STEPS), "--algo", algo, "--env", env_id, "--log-folder", tmp_path]

    return_code = subprocess.call(["python", "train.py", *args])
    _assert_eq(return_code, 0)


def test_continue_training(tmp_path):
    algo, env_id = "a2c", "CartPole-v1"
    args = [
        "-n",
        str(N_STEPS),
        "--algo",
        algo,
        "--env",
        env_id,
        "--log-folder",
        tmp_path,
        "-i",
        "rl-trained-agents/a2c/CartPole-v1_1/CartPole-v1.zip",
    ]

    return_code = subprocess.call(["python", "train.py", *args])
    _assert_eq(return_code, 0)


def test_save_load_replay_buffer(tmp_path):
    algo, env_id = "sac", "Pendulum-v1"
    args = [
        "-n",
        str(N_STEPS),
        "--algo",
        algo,
        "--env",
        env_id,
        "--log-folder",
        tmp_path,
        "--save-replay-buffer",
        "-params",
        "buffer_size:1000",
    ]

    return_code = subprocess.call(["python", "train.py", *args])
    _assert_eq(return_code, 0)

    assert os.path.isfile(os.path.join(tmp_path, "sac/Pendulum-v1_1/replay_buffer.pkl"))

    args = [*args, "-i", os.path.join(tmp_path, "sac/Pendulum-v1_1/Pendulum-v1.zip")]

    return_code = subprocess.call(["python", "train.py", *args])
    _assert_eq(return_code, 0)


def test_parallel_train(tmp_path):
    args = [
        "-n",
        str(1000),
        "--algo",
        "sac",
        "--env",
        "Pendulum-v1",
        "--log-folder",
        tmp_path,
        "-params",
        # Test custom argument for the monitor too
        "monitor_kwargs:'dict(info_keywords=(\"TimeLimit.truncated\",))'",
        "callback:'rl_zoo3.callbacks.ParallelTrainCallback'",
    ]

    return_code = subprocess.call(["python", "train.py", *args])
    _assert_eq(return_code, 0)


def test_custom_yaml(tmp_path):
    # Use A2C hyperparams for ppo
    args = [
        "-n",
        str(N_STEPS),
        "--algo",
        "ppo",
        "--env",
        "CartPole-v1",
        "--log-folder",
        tmp_path,
        "-conf",
        "hyperparams/a2c.yml",
        "-params",
        "n_envs:2",
        "n_steps:50",
        "n_epochs:2",
        "batch_size:4",
        # Test custom policy
        "policy:'stable_baselines3.ppo.MlpPolicy'",
    ]

    return_code = subprocess.call(["python", "train.py", *args])
    _assert_eq(return_code, 0)


@pytest.mark.parametrize("config_file", ["hyperparams.python.ppo_config_example", "hyperparams/python/ppo_config_example.py"])
def test_python_config_file(tmp_path, config_file):
    # Use the example python config file for training
    args = [
        "-n",
        str(N_STEPS),
        "--algo",
        "ppo",
        "--env",
        "MountainCarContinuous-v0",
        "--log-folder",
        tmp_path,
        "-conf",
        config_file,
    ]

    return_code = subprocess.call(["python", "train.py", *args])
    _assert_eq(return_code, 0)
