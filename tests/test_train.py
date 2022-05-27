import os
import subprocess

import optuna
import pytest
from optuna.trial import TrialState


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

    return_code = subprocess.call(["python", "train.py"] + args)
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

    return_code = subprocess.call(["python", "train.py"] + args)
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

    return_code = subprocess.call(["python", "train.py"] + args)
    _assert_eq(return_code, 0)

    assert os.path.isfile(os.path.join(tmp_path, "sac/Pendulum-v1_1/replay_buffer.pkl"))

    args = args + ["-i", os.path.join(tmp_path, "sac/Pendulum-v1_1/Pendulum-v1.zip")]

    return_code = subprocess.call(["python", "train.py"] + args)
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
        "callback:'utils.callbacks.ParallelTrainCallback'",
    ]

    return_code = subprocess.call(["python", "train.py"] + args)
    _assert_eq(return_code, 0)


def is_redis_available():
    try:
        import redis
    except ImportError:
        return False
    try:
        return redis.Redis(host='localhost', port=6379).ping()
    except (redis.ConnectionError, ImportError):
        return False


@pytest.mark.skipif(not is_redis_available(), reason="Redis not installed or no Redis server reachable")
def test_multiple_workers(tmp_path):
    study_name = "test-study"
    storage = "redis://localhost:6379"
    n_trials = 6
    args = [
        "-optimize",
        "--no-optim-plots",
        "--storage",
        storage,
        "--max-total-trials",
        str(n_trials),
        "--study-name",
        study_name,
        "--n-evaluations",
        str(1),
        "-n",
        str(100),
        "--algo",
        "ppo",
        "--env",
        "Pendulum-v1",
        "--log-folder",
        tmp_path,
    ]

    p1 = subprocess.Popen(["python", "train.py"] + args)
    p2 = subprocess.Popen(["python", "train.py"] + args)
    p3 = subprocess.Popen(["python", "train.py"] + args)
    p4 = subprocess.Popen(["python", "train.py"] + args)

    return_code1 = p1.wait()
    return_code2 = p2.wait()
    return_code3 = p3.wait()
    return_code4 = p4.wait()

    study = optuna.load_study(study_name=study_name, storage=storage)
    assert sum(t.state in (TrialState.COMPLETE, TrialState.PRUNED) for t in study.trials) == n_trials

    assert return_code1 == 0
    assert return_code2 == 0
    assert return_code3 == 0
    assert return_code4 == 0
