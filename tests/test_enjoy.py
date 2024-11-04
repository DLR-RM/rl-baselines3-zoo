import os
import shlex
import subprocess
from importlib.metadata import version

import pytest

from rl_zoo3.utils import get_hf_trained_models, get_trained_models

# Test models from sb3 organization can be trusted
os.environ["TRUST_REMOTE_CODE"] = "True"


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

    # TODO: rename trained agents once we drop support for gymnasium v0.29
    if "Lander" in env_id and version("gymnasium") > "0.29.1":
        # LunarLander-v2 is now LunarLander-v3
        return

    # Skip mujoco envs
    if "Fetch" in trained_model or "-v3" in trained_model:
        return

    # FIXME: switch to MiniGrid package
    if "-MiniGrid-" in trained_model:
        # FIXME: switch to Gymnsium
        return

    cmd = f"python enjoy.py --algo {algo} --env {env_id} -n {N_STEPS} -f {FOLDER} --no-render"
    return_code = subprocess.call(shlex.split(cmd))
    _assert_eq(return_code, 0)


def test_benchmark(tmp_path):
    cmd = f"python -m rl_zoo3.benchmark -n {N_STEPS} --benchmark-dir {tmp_path} --test-mode --no-hub"
    return_code = subprocess.call(shlex.split(cmd))
    _assert_eq(return_code, 0)


def test_load(tmp_path):
    algo, env_id = "a2c", "CartPole-v1"
    # Train and save checkpoints and best model
    cmd = (
        f"python train.py --algo {algo} --env {env_id} -n 1000 -f {tmp_path} "
        # Enable progress bar
        f"-params n_envs:1 --eval-freq 500 --save-freq 500 -P"
    )
    return_code = subprocess.call(shlex.split(cmd))
    _assert_eq(return_code, 0)

    # Load best model
    base_cmd = f"python enjoy.py --algo {algo} --env {env_id} -n {N_STEPS} -f {tmp_path} --no-render "
    # Enable progress bar
    return_code = subprocess.call(shlex.split(base_cmd + "--load-best -P"))

    _assert_eq(return_code, 0)

    # Load checkpoint
    return_code = subprocess.call(shlex.split(base_cmd + "--load-checkpoint 500"))
    _assert_eq(return_code, 0)

    # Load last checkpoint
    return_code = subprocess.call(shlex.split(base_cmd + "--load-last-checkpoint"))
    _assert_eq(return_code, 0)


def test_record_video(tmp_path):
    # Skip if no X-Server
    if not os.environ.get("DISPLAY"):
        pytest.skip("No X-Server")

    cmd = f"python -m rl_zoo3.record_video -n 100 --algo sac --env Pendulum-v1 -o {tmp_path}"
    return_code = subprocess.call(shlex.split(cmd))

    _assert_eq(return_code, 0)
    video_path = str(tmp_path / "final-model-sac-Pendulum-v1-step-0-to-step-100.mp4")
    # File is not empty
    assert os.stat(video_path).st_size != 0, "Recorded video is empty"


def test_record_training(tmp_path):
    videos_tmp_path = tmp_path / "videos"
    algo, env_id = "ppo", "CartPole-v1"

    # Skip if no X-Server
    if not os.environ.get("DISPLAY"):
        pytest.skip("No X-Server")

    cmd = f"python train.py -n 10000 --algo {algo} --env {env_id} --log-folder {tmp_path} --save-freq 4000 "
    return_code = subprocess.call(shlex.split(cmd))
    _assert_eq(return_code, 0)

    cmd = (
        f"python -m rl_zoo3.record_training -n 100 --algo {algo} --env {env_id} "
        f"--f {tmp_path} "
        f"--gif -o {videos_tmp_path}"
    )
    return_code = subprocess.call(shlex.split(cmd))
    _assert_eq(return_code, 0)

    mp4_path = str(videos_tmp_path / "training.mp4")
    gif_path = str(videos_tmp_path / "training.gif")
    # File is not empty
    assert os.stat(mp4_path).st_size != 0, "Recorded mp4 video is empty"
    assert os.stat(gif_path).st_size != 0, "Converted gif video is empty"
