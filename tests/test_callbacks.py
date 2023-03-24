import subprocess


def _assert_eq(left, right):
    assert left == right, f"{left} != {right}"


def test_raw_stat_callback(tmp_path):
    args = [
        "-n",
        str(200),
        "--algo",
        "ppo",
        "--env",
        "CartPole-v1",
        "-params",
        "callback:'rl_zoo3.callbacks.RawStatisticsCallback'",
        "--tensorboard-log",
        f"{tmp_path}",
    ]

    return_code = subprocess.call(["python", "train.py", *args])
    _assert_eq(return_code, 0)
