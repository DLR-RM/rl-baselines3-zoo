import os
import subprocess
import sys

import pytest

from rl_zoo3.utils import get_hf_trained_models, get_trained_models


def _assert_eq(left, right):
    assert left == right, f"{left} != {right}"


FOLDER = "rl-trained-agents/"
N_STEPS = 100
# Use local models
trained_models = get_trained_models(FOLDER)
# Use huggingface models too
trained_models.update(get_hf_trained_models())


@pytest.mark.parametrize("trained_model", trained_models.keys())
@pytest.mark.slow
def test_trained_agents(trained_model):
    algo, env_id = trained_models[trained_model]
    args = ["-n", str(N_STEPS), "-f", FOLDER, "--algo", algo, "--env", env_id, "--no-render"]

    # Since SB3 >= 1.1.0, HER is no more an algorithm but a replay buffer class
    if algo == "her":
        return

    # skip car racing
    if "CarRacing" in env_id:
        return

    # FIXME: skip Panda gym envs
    # need panda gym >= 3.0.1 and gymnasium
    if "Panda" in env_id:
        return

    # Skip mujoco envs
    if "Fetch" in trained_model or "-v3" in trained_model:
        return

    # FIXME: switch to MiniGrid package
    if "-MiniGrid-" in trained_model:
        # Skip for python 3.7, see https://github.com/DLR-RM/rl-baselines3-zoo/pull/372#issuecomment-1490562332
        if sys.version_info[:2] == (3, 7):
            pytest.skip("MiniGrid env does not work with Python 3.7")
        # FIXME: switch to Gymnsium
        return

    return_code = subprocess.call(["python", "enjoy.py", *args])
    _assert_eq(return_code, 0)


def test_benchmark(tmp_path):
    args = ["-n", str(N_STEPS), "--benchmark-dir", tmp_path, "--test-mode", "--no-hub"]

    return_code = subprocess.call(["python", "-m", "rl_zoo3.benchmark", *args])
    _assert_eq(return_code, 0)


def test_load(tmp_path):
    algo, env_id = "a2c", "CartPole-v1"
    args = [
        "-n",
        str(1000),
        "--algo",
        algo,
        "--env",
        env_id,
        "-params",
        "n_envs:1",
        "--log-folder",
        tmp_path,
        "--eval-freq",
        str(500),
        "--save-freq",
        str(500),
        "-P",  # Enable progress bar
    ]
    # Train and save checkpoints and best model
    return_code = subprocess.call(["python", "train.py", *args])
    _assert_eq(return_code, 0)

    # Load best model
    args = ["-n", str(N_STEPS), "-f", tmp_path, "--algo", algo, "--env", env_id, "--no-render"]
    # Test with progress bar
    return_code = subprocess.call(["python", "enjoy.py", *args, "--load-best", "-P"])
    _assert_eq(return_code, 0)

    # Load checkpoint
    return_code = subprocess.call(["python", "enjoy.py", *args, "--load-checkpoint", str(500)])
    _assert_eq(return_code, 0)

    # Load last checkpoint
    return_code = subprocess.call(["python", "enjoy.py", *args, "--load-last-checkpoint"])
    _assert_eq(return_code, 0)


def test_record_video(tmp_path):
    args = ["-n", "100", "--algo", "sac", "--env", "Pendulum-v1", "-o", str(tmp_path)]

    # Skip if no X-Server
    if not os.environ.get("DISPLAY"):
        pytest.skip("No X-Server")

    return_code = subprocess.call(["python", "-m", "rl_zoo3.record_video", *args])
    _assert_eq(return_code, 0)
    video_path = str(tmp_path / "final-model-sac-Pendulum-v1-step-0-to-step-100.mp4")
    # File is not empty
    assert os.stat(video_path).st_size != 0, "Recorded video is empty"


def test_record_training(tmp_path):
    videos_tmp_path = tmp_path / "videos"
    args_training = [
        "--algo",
        "ppo",
        "--env",
        "CartPole-v1",
        "--log-folder",
        str(tmp_path),
        "--save-freq",
        "4000",
        "-n",
        "10000",
    ]
    args_recording = [
        "--algo",
        "ppo",
        "--env",
        "CartPole-v1",
        "--gif",
        "-n",
        "100",
        "-f",
        str(tmp_path),
        "-o",
        str(videos_tmp_path),
    ]

    # Skip if no X-Server
    if not os.environ.get("DISPLAY"):
        pytest.skip("No X-Server")

    return_code = subprocess.call(["python", "train.py", *args_training])
    _assert_eq(return_code, 0)

    return_code = subprocess.call(["python", "-m", "rl_zoo3.record_training", *args_recording])
    _assert_eq(return_code, 0)
    mp4_path = str(videos_tmp_path / "training.mp4")
    gif_path = str(videos_tmp_path / "training.gif")
    # File is not empty
    assert os.stat(mp4_path).st_size != 0, "Recorded mp4 video is empty"
    assert os.stat(gif_path).st_size != 0, "Converted gif video is empty"
