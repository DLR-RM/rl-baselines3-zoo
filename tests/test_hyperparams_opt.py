import glob
import os
import shlex
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
# Test for RecurrentPPO (ppo_lstm)
experiments["ppo_lstm-CartPoleNoVel-v1"] = ("ppo_lstm", "CartPoleNoVel-v1")


@pytest.mark.parametrize("sampler", ["random", "tpe"])
@pytest.mark.parametrize("pruner", ["none", "halving", "median"])
@pytest.mark.parametrize("experiment", experiments.keys())
def test_optimize(tmp_path, sampler, pruner, experiment):
    algo, env_id = experiments[experiment]

    # Skip slow tests
    if algo not in {"a2c", "ppo"} and not (sampler == "random" and pruner == "median"):
        pytest.skip("Skipping slow tests")

    maybe_params = "n_steps:10" if algo == "ppo" else ""
    cmd = (
        f"python train.py -n {N_STEPS} --algo {algo} --env {env_id} --log-folder {tmp_path} "
        f"-params policy_kwargs:'dict(net_arch=[32])' {maybe_params} "
        f"--no-optim-plots --seed 14 --n-trials {N_TRIALS} --n-jobs {N_JOBS} "
        f"--sampler {sampler} --pruner {pruner} --n-evaluations 2 --n-startup-trials 1 -optimize"
    )
    return_code = subprocess.call(shlex.split(cmd))
    _assert_eq(return_code, 0)


def test_optimize_log_path(tmp_path):
    algo, env_id = "a2c", "CartPole-v1"
    sampler = "random"
    pruner = "median"
    optimization_log_path = str(tmp_path / "optim_logs")

    cmd = (
        f"python train.py -n {N_STEPS} --algo {algo} --env {env_id} --log-folder {tmp_path} "
        f"-params policy_kwargs:'dict(net_arch=[32])' "
        f"--no-optim-plots --seed 14 --n-trials {N_TRIALS} --n-jobs {N_JOBS} "
        f"--sampler {sampler} --pruner {pruner} --n-evaluations 2 --n-startup-trials 1 "
        f"--optimization-log-path {optimization_log_path} -optimize"
    )
    return_code = subprocess.call(shlex.split(cmd))
    _assert_eq(return_code, 0)

    print(optimization_log_path)
    assert os.path.isdir(optimization_log_path)
    # Log folder of the first trial
    assert os.path.isdir(os.path.join(optimization_log_path, "trial_1"))
    assert os.path.isfile(os.path.join(optimization_log_path, "trial_1", "evaluations.npz"))

    study_path = next(iter(glob.glob(str(tmp_path / algo / "report_*.pkl"))))
    print(study_path)
    # Test reading best trials
    cmd = (
        "python scripts/parse_study.py "
        f"-i {study_path} --print-n-best-trials {N_TRIALS} "
        f"--save-n-best-hyperparameters {N_TRIALS} -f {tmp_path / 'best_hyperparameters'}"
    )
    return_code = subprocess.call(shlex.split(cmd))
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
    cmd = (
        f"python train.py -n 100 --algo a2c --env Pendulum-v1 --log-folder {tmp_path} "
        "-params n_envs:1 --n-evaluations 1 "
        f"--no-optim-plots --seed 12 --n-trials {n_trials} --max-total-trials {max_trials} "
        f"--storage {storage} --study-name {study_name} --no-optim-plots  -optimize"
    )

    # Sequencial execution to avoid race conditions
    workers = []
    for _ in range(n_workers):
        worker = subprocess.Popen(shlex.split(cmd), stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
        worker.wait()
        workers.append(worker)

    study = optuna.load_study(study_name=study_name, storage=storage)
    assert len(study.get_trials(states=(TrialState.COMPLETE, TrialState.PRUNED))) == max_trials

    for worker in workers:
        assert worker.returncode == 0, "STDOUT:\n{}\nSTDERR:\n{}\n".format(*worker.communicate())
