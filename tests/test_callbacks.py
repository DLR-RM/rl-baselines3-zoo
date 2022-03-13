

import subprocess


def _assert_eq(left, right):
    assert left == right, f"{left} != {right}"


# def test_parallel_train():
args = [
    "-n",
    str(1000),
    "--algo",
    "ppo",
    "--env",
    "CartPole-v1",
    "-params",
    "callback:'utils.callbacks.RawStatisticsCallback'",
    "--tensorboard-log",
    "/tmp/stable-baselines/",
]

return_code = subprocess.call(["python", "train.py"] + args)
_assert_eq(return_code, 0)