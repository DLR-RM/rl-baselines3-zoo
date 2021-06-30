import os
import subprocess

import pytest


def _assert_eq(left, right):
    assert left == right, f"{left} != {right}"


N_STEPS = 100
N_TRIALS = 2
N_JOBS = 1

ALGOS = ("ppo", "a2c")
ENV_IDS = ("CartPole-v1",)

experiments = {}

for algo in ALGOS:
    for env_id in ENV_IDS:
        experiments[f"{algo}-{env_id}"] = (algo, env_id)

# Test for SAC
experiments["sac-Pendulum-v0"] = ("sac", "Pendulum-v0")
# Test for TD3
experiments["td3-Pendulum-v0"] = ("td3", "Pendulum-v0")
# Test for HER
experiments["tqc-parking-v0"] = ("tqc", "parking-v0")
# Test for TQC
experiments["tqc-Pendulum-v0"] = ("tqc", "Pendulum-v0")


@pytest.mark.parametrize("sampler", ["random", "tpe"])
@pytest.mark.parametrize("pruner", ["none", "halving", "median"])
@pytest.mark.parametrize("experiment", experiments.keys())
def test_optimize(tmp_path, sampler, pruner, experiment):
    algo, env_id = experiments[experiment]

    # Skip slow tests
    if algo not in {"a2c", "ppo"} and not (sampler == "random" and pruner == "median"):
        pytest.skip("Skipping slow tests")

    args = ["-n", str(N_STEPS), "--algo", algo, "--env", env_id, "-params", 'policy_kwargs:"dict(net_arch=[32])"', "n_envs:1"]
    args += ["n_steps:10"] if algo == "ppo" else []
    args += [
        "--seed",
        "14",
        "--log-folder",
        tmp_path,
        "--n-trials",
        str(N_TRIALS),
        "--n-jobs",
        str(N_JOBS),
        "--sampler",
        sampler,
        "--pruner",
        pruner,
        "--n-evaluations",
        str(2),
        "--n-startup-trials",
        str(1),
        "-optimize",
    ]

    return_code = subprocess.call(["python", "train.py"] + args)
    _assert_eq(return_code, 0)


def test_optimize_log_path(tmp_path):
    algo, env_id = "a2c", "CartPole-v1"
    sampler = "random"
    pruner = "median"
    optimization_log_path = str(tmp_path / "optim_logs")

    args = ["-n", str(N_STEPS), "--algo", algo, "--env", env_id, "-params", 'policy_kwargs:"dict(net_arch=[32])"', "n_envs:1"]
    args += [
        "--seed",
        "14",
        "--log-folder",
        tmp_path,
        "--n-trials",
        str(N_TRIALS),
        "--n-jobs",
        str(N_JOBS),
        "--sampler",
        sampler,
        "--pruner",
        pruner,
        "--n-evaluations",
        str(2),
        "--n-startup-trials",
        str(1),
        "--optimization-log-path",
        optimization_log_path,
        "-optimize",
    ]

    return_code = subprocess.call(["python", "train.py"] + args)
    _assert_eq(return_code, 0)
    print(optimization_log_path)
    assert os.path.isdir(optimization_log_path)
    # Log folder of the first trial
    assert os.path.isdir(os.path.join(optimization_log_path, "trial_1"))
    assert os.path.isfile(os.path.join(optimization_log_path, "trial_1", "evaluations.npz"))
