import argparse
import glob
import os
import shutil
import zipfile
from pathlib import Path
from pprint import pformat
from typing import Any, Dict, Optional, Tuple

import torch as th
import yaml
from huggingface_hub import HfApi, Repository
from huggingface_sb3.push_to_hub import _evaluate_agent, _generate_replay, _save_model_card, generate_metadata
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import VecEnv, unwrap_vec_normalize
from wasabi import Printer

import utils.import_envs  # noqa: F401 pylint: disable=unused-import
from utils import ALGOS, create_test_env, get_saved_hyperparams
from utils.exp_manager import ExperimentManager
from utils.utils import StoreDict, get_model_path

msg = Printer()


def generate_model_card(
    algo_name: str,
    model_architecture: str,
    organization: str,
    env_id: str,
    mean_reward: float,
    std_reward: float,
    hyperparams: Dict[str, Any],
    env_kwargs: Dict[str, Any],
) -> Tuple[str, Dict[str, Any]]:
    """
    Generate the model card for the Hub
    :param model_architecture: name of the model
    :env_id: name of the environment
    :mean_reward: mean reward of the agent
    :std_reward: standard deviation of the mean reward of the agent
    """
    # Step 1: Select the tags
    metadata = generate_metadata(model_architecture, env_id, mean_reward, std_reward)

    # Step 2: Generate the model card
    model_card = f"""
# **{model_architecture}** Agent playing **{env_id}**
This is a trained model of a **{model_architecture}** agent playing **{env_id}**
using the [stable-baselines3 library](https://github.com/DLR-RM/stable-baselines3)
and the [RL Zoo](https://github.com/DLR-RM/rl-baselines3-zoo).

The RL Zoo is a training framework for Stable Baselines3
reinforcement learning agents,
with hyperparameter optimization and pre-trained agents included.
"""

    model_card += f"""
## Usage (with SB3 RL Zoo)

RL Zoo: https://github.com/DLR-RM/rl-baselines3-zoo

```
# Download model and save it into the logs/ folder
python -m utils.load_from_hub --algo {algo_name} --env {env_id} -orga {organization} -f logs/
python enjoy --algo {algo_name} --env {env_id}  -f logs/
```

## Training (with the RL Zoo)
```
python train.py --algo {algo_name} --env {env_id} -f logs/
# Upload the model and generate video (when possible)
python -m utils.push_to_hub --algo {algo_name} --env {env_id} -f logs/ -orga {organization}
```

## Hyperparameters
```python
{pformat(hyperparams)}
```
"""
    if len(env_kwargs) > 0:
        model_card += f"""
# Environment Arguments
```python
{pformat(env_kwargs)}
```
"""

    return model_card, metadata


def package_to_hub(
    model: BaseAlgorithm,
    model_name: str,
    algo_name: str,
    model_architecture: str,
    log_path: Path,
    hyperparams: Dict[str, Any],
    env_kwargs: Dict[str, Any],
    env_id: str,
    eval_env: VecEnv,
    repo_id: str,
    commit_message: str,
    is_deterministic: bool = True,
    n_eval_episodes=10,
    token: Optional[str] = None,
    local_repo_path="hub",
    video_length=1000,
    generate_video: bool = False,
):
    """
    Evaluate, Generate a video and Upload a model to Hugging Face Hub.
    This method does the complete pipeline:
    - It evaluates the model
    - It generates the model card
    - It generates a replay video of the agent
    - It pushes everything to the hub

    This is a work in progress function, if it does not work,
    use `push_to_hub` method.

    :param model: trained model
    :param model_name: name of the model zip file
    :param model_architecture: name of the architecture of your model
        (DQN, PPO, A2C, SAC...)
    :param env_id: name of the environment
    :param eval_env: environment used to evaluate the agent
    :param repo_id: id of the model repository from the Hugging Face Hub
    :param commit_message: commit message
    :param is_deterministic: use deterministic or stochastic actions (by default: True)
    :param n_eval_episodes: number of evaluation episodes (by default: 10)
    :param local_repo_path: local repository path
    :param video_length: length of the video (in timesteps)
    """

    msg.info(
        "This function will save, evaluate, generate a video of your agent, "
        "create a model card and push everything to the hub. "
        "It might take up to some minutes if video generation is activated. "
        "This is a work in progress: if you encounter a bug, please open an issue."
    )

    organization, repo_name = repo_id.split("/")

    # Step 1: Clone or create the repo
    # Create the repo (or clone its content if it's nonempty)
    api = HfApi()

    repo_url = api.create_repo(
        name=repo_name,
        token=token,
        organization=organization,
        private=False,
        exist_ok=True,
    )

    # Git pull
    repo_local_path = Path(local_repo_path) / repo_name
    repo = Repository(repo_local_path, clone_from=repo_url, use_auth_token=True)
    repo.git_pull(rebase=True)

    repo.lfs_track(["*.mp4"])

    # Step 1: Save the model
    model.save(repo_local_path / model_name)

    # Retrieve VecNormalize wrapper if it exists
    # we need to save the statistics
    maybe_vec_normalize = unwrap_vec_normalize(eval_env)

    # Save the normalization
    if maybe_vec_normalize is not None:
        maybe_vec_normalize.save(repo_local_path / "vec_normalize.pkl")
        # Do not update the stats at test time
        maybe_vec_normalize.training = False
        # Reward normalization is not needed at test time
        maybe_vec_normalize.norm_reward = False

    # Unzip the model
    with zipfile.ZipFile(repo_local_path / f"{model_name}.zip", "r") as zip_ref:
        zip_ref.extractall(repo_local_path / model_name)

    # Step 2: Copy config files
    args_path = log_path / env_id / "args.yml"
    config_path = log_path / env_id / "config.yml"

    shutil.copy(args_path, repo_local_path / "args.yml")
    shutil.copy(config_path, repo_local_path / "config.yml")
    with open(repo_local_path / "env_kwargs.yml", "w") as outfile:
        yaml.dump(env_kwargs, outfile)

    # Copy train/eval metrics into zip
    with zipfile.ZipFile(repo_local_path / "train_eval_metrics.zip", "w") as archive:
        if os.path.isfile(log_path / "evaluations.npz"):
            archive.write(log_path / "evaluations.npz", arcname="evaluations.npz")
        for monitor_file in glob.glob(f"{log_path}/*.csv"):
            archive.write(monitor_file, arcname=monitor_file.split(os.sep)[-1])

    # Step 3: Evaluate the agent
    mean_reward, std_reward = _evaluate_agent(model, eval_env, n_eval_episodes, is_deterministic, repo_local_path)

    # Step 4: Generate a video
    if generate_video:
        # TODO: cleanup files after generation
        _generate_replay(model, eval_env, video_length, is_deterministic, repo_local_path)

    # Step 5: Generate the model card
    generated_model_card, metadata = generate_model_card(
        algo_name,
        model_architecture,
        organization,
        env_id,
        mean_reward,
        std_reward,
        hyperparams,
        env_kwargs,
    )

    _save_model_card(repo_local_path, generated_model_card, metadata)

    msg.info(f"Pushing repo {repo_name} to the Hugging Face Hub")
    repo.push_to_hub(commit_message=commit_message)

    msg.info(f"Your model is pushed to the hub. You can view your model here: {repo_url}")
    return repo_url


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", help="environment ID", type=str, required=True)
    parser.add_argument("-f", "--folder", help="Log folder", type=str, required=True)
    parser.add_argument("--algo", help="RL Algorithm", type=str, required=True, choices=list(ALGOS.keys()))
    parser.add_argument("-n", "--n-timesteps", help="number of timesteps", default=1000, type=int)
    parser.add_argument("--num-threads", help="Number of threads for PyTorch (-1 to use default)", default=-1, type=int)
    parser.add_argument("--n-envs", help="number of environments", default=1, type=int)
    parser.add_argument("--exp-id", help="Experiment ID (default: 0: latest, -1: no exp folder)", default=0, type=int)
    parser.add_argument("--verbose", help="Verbose mode (0: no output, 1: INFO)", default=1, type=int)
    parser.add_argument(
        "--no-render", action="store_true", default=False, help="Do not render the environment (useful for tests)"
    )
    parser.add_argument("--deterministic", action="store_true", default=False, help="Use deterministic actions")
    parser.add_argument("--device", help="PyTorch device to be use (ex: cpu, cuda...)", default="auto", type=str)
    parser.add_argument(
        "--load-best", action="store_true", default=False, help="Load best model instead of last model if available"
    )
    parser.add_argument(
        "--load-checkpoint",
        type=int,
        help="Load checkpoint instead of last model if available, "
        "you must pass the number of timesteps corresponding to it",
    )
    parser.add_argument(
        "--load-last-checkpoint",
        action="store_true",
        default=False,
        help="Load last checkpoint instead of last model if available",
    )
    parser.add_argument("--stochastic", action="store_true", default=False, help="Use stochastic actions")
    parser.add_argument("--seed", help="Random generator seed", type=int, default=0)
    parser.add_argument(
        "--env-kwargs", type=str, nargs="+", action=StoreDict, help="Optional keyword argument to pass to the env constructor"
    )
    parser.add_argument("-orga", "--organization", help="Huggingface hub organization", default="sb3")
    # TODO: allow to overwrite repo-id
    args = parser.parse_args()
    env_id = args.env
    algo = args.algo

    _, model_path, log_path = get_model_path(
        args.exp_id,
        args.folder,
        args.algo,
        args.env,
        args.load_best,
        args.load_checkpoint,
        args.load_last_checkpoint,
    )

    print(f"Loading {model_path}")

    # Off-policy algorithm only support one env for now
    off_policy_algos = ["qrdqn", "dqn", "ddpg", "sac", "her", "td3", "tqc"]

    if algo in off_policy_algos:
        args.n_envs = 1

    set_random_seed(args.seed)

    if args.num_threads > 0:
        if args.verbose > 1:
            print(f"Setting torch.num_threads to {args.num_threads}")
        th.set_num_threads(args.num_threads)

    is_atari = ExperimentManager.is_atari(env_id)

    stats_path = os.path.join(log_path, env_id)
    hyperparams, stats_path = get_saved_hyperparams(stats_path, test_mode=True)

    # load env_kwargs if existing
    env_kwargs = {}
    args_path = os.path.join(log_path, env_id, "args.yml")
    if os.path.isfile(args_path):
        with open(args_path) as f:
            loaded_args = yaml.load(f, Loader=yaml.UnsafeLoader)  # pytype: disable=module-attr
            if loaded_args["env_kwargs"] is not None:
                env_kwargs = loaded_args["env_kwargs"]
    # overwrite with command line arguments
    if args.env_kwargs is not None:
        env_kwargs.update(args.env_kwargs)

    eval_env = create_test_env(
        env_id,
        n_envs=args.n_envs,
        stats_path=stats_path,
        seed=args.seed,
        log_dir=None,
        should_render=not args.no_render,
        hyperparams=hyperparams,
        env_kwargs=env_kwargs,
    )

    kwargs = dict(seed=args.seed)
    if algo in off_policy_algos:
        # Dummy buffer size as we don't need memory to enjoy the trained agent
        kwargs.update(dict(buffer_size=1))

    # TODO: use try/except instead
    # Check if we are running python 3.8+
    # we need to patch saved model under python 3.6/3.7 to load them
    # newer_python_version = sys.version_info.major == 3 and sys.version_info.minor >= 8

    custom_objects = {}
    # if newer_python_version:
    #     custom_objects = {
    #         "learning_rate": 0.0,
    #         "lr_schedule": lambda _: 0.0,
    #         "clip_range": lambda _: 0.0,
    #     }
    model = ALGOS[algo].load(model_path, env=eval_env, custom_objects=custom_objects, device=args.device, **kwargs)

    # Deterministic by default except for atari games
    stochastic = args.stochastic or is_atari and not args.deterministic
    deterministic = not stochastic

    # TODO: allow overwrite
    model_name = f"{algo}-{env_id}"

    package_to_hub(
        model,
        model_name,
        algo,
        ALGOS[algo].__name__,
        Path(log_path),
        hyperparams,
        env_kwargs,
        env_id,
        eval_env,
        repo_id=f"{args.organization}/{model_name}",
        commit_message="Initial Commit",
        is_deterministic=deterministic,
        n_eval_episodes=10,
        token=None,
        local_repo_path="hub",
        video_length=1000,
        generate_video=not args.no_render,
    )
