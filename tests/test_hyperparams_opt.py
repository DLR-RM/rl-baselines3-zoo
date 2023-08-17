import glob
import os
import subprocess

import optuna
import pytest
from optuna.trial import TrialState


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
experiments["sac-Pendulum-v1"] = ("sac", "Pendulum-v1")
# Test for TD3
experiments["td3-Pendulum-v1"] = ("td3", "Pendulum-v1")
# Test for HER
experiments["tqc-parking-v0"] = ("tqc", "parking-v0")
# Test for TQC
experiments["tqc-Pendulum-v1"] = ("tqc", "Pendulum-v1")


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
        "--no-optim-plots",
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

    return_code = subprocess.call(["python", "train.py", *args])
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

    return_code = subprocess.call(["python", "train.py", *args])
    _assert_eq(return_code, 0)
    print(optimization_log_path)
    assert os.path.isdir(optimization_log_path)
    # Log folder of the first trial
    assert os.path.isdir(os.path.join(optimization_log_path, "trial_1"))
    assert os.path.isfile(os.path.join(optimization_log_path, "trial_1", "evaluations.npz"))

    study_path = next(iter(glob.glob(str(tmp_path / algo / "report_*.pkl"))))
    print(study_path)
    # Test reading best trials
    args = [
        "-i",
        study_path,
        "--print-n-best-trials",
        str(N_TRIALS),
        "--save-n-best-hyperparameters",
        str(N_TRIALS),
        "-f",
        str(tmp_path / "best_hyperparameters"),
    ]
    return_code = subprocess.call(["python", "scripts/parse_study.py", *args])
    _assert_eq(return_code, 0)


def test_multiple_workers(tmp_path):
    study_name = "test-study"
    storage = f"sqlite:///{tmp_path}/optuna.db"
    # n trials per worker
    n_trials = 2
    # max total trials
    max_trials = 3
    # 1st worker will do 2 trials
    # 2nd worker will do 1 trial
    # 3rd worker will do nothing
    n_workers = 3
    args = [
        "-optimize",
        "--no-optim-plots",
        "--storage",
        storage,
        "--n-trials",
        str(n_trials),
        "--max-total-trials",
        str(max_trials),
        "--study-name",
        study_name,
        "--n-evaluations",
        str(1),
        "-n",
        str(100),
        "--algo",
        "a2c",
        "--env",
        "Pendulum-v1",
        "--log-folder",
        tmp_path,
        "-params",
        "n_envs:1",
        "--seed",
        "12",
    ]

    # Sequencial execution to avoid race conditions
    workers = []
    for _ in range(n_workers):
        worker = subprocess.Popen(
            ["python", "train.py", *args], stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True
        )
        worker.wait()
        workers.append(worker)

    study = optuna.load_study(study_name=study_name, storage=storage)
    assert len(study.get_trials(states=(TrialState.COMPLETE, TrialState.PRUNED))) == max_trials

    for worker in workers:
        assert worker.returncode == 0, "STDOUT:\n{}\nSTDERR:\n{}\n".format(*worker.communicate())
