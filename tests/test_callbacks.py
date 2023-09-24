import shlex
import subprocess


def _assert_eq(left, right):
    assert left == right, f"{left} != {right}"


def test_raw_stat_callback(tmp_path):
    cmd = (
        f"python train.py -n 200 --algo ppo --env CartPole-v1 --log-folder {tmp_path} "
        f"--tensorboard-log {tmp_path} -params callback:\"'rl_zoo3.callbacks.RawStatisticsCallback'\""
    )
    return_code = subprocess.call(shlex.split(cmd))
    _assert_eq(return_code, 0)
