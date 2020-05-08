import os
import shutil
import subprocess

import pytest


def _assert_eq(left, right):
    assert left == right, '{} != {}'.format(left, right)


N_STEPS = 100

ALGOS = ('ppo', 'a2c')
# 'BreakoutNoFrameskip-v4'
ENV_IDS = ('CartPole-v1',)
LOG_FOLDER = 'logs/tests/'

experiments = {}

for algo in ALGOS:
    for env_id in ENV_IDS:
        experiments['{}-{}'.format(algo, env_id)] = (algo, env_id)

# Test for vecnormalize and frame-stack
experiments['ppo-BipedalWalkerHardcore-v3'] = ('ppo', 'BipedalWalkerHardcore-v3')
# Test for SAC
experiments['sac-Pendulum-v0'] = ('sac', 'Pendulum-v0')
# for TD3
experiments['td3-Pendulum-v0'] = ('td3', 'Pendulum-v0')

# Clean up
if os.path.isdir(LOG_FOLDER):
    shutil.rmtree(LOG_FOLDER)


@pytest.mark.parametrize("experiment", experiments.keys())
def test_train(experiment):
    algo, env_id = experiments[experiment]
    args = [
        '-n', str(N_STEPS),
        '--algo', algo,
        '--env', env_id,
        '--log-folder', LOG_FOLDER
    ]

    return_code = subprocess.call(['python', 'train.py'] + args)
    _assert_eq(return_code, 0)


def test_continue_training():
    algo, env_id = 'a2c', 'CartPole-v1'
    args = [
        '-n', str(N_STEPS),
        '--algo', algo,
        '--env', env_id,
        '--log-folder', LOG_FOLDER,
        '-i', 'rl-trained-agents/a2c/CartPole-v1_1/CartPole-v1.zip'
    ]

    return_code = subprocess.call(['python', 'train.py'] + args)
    _assert_eq(return_code, 0)
