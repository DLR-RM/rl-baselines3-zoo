import argparse
import os
import shutil
from pathlib import Path
import argparse
import glob
import importlib
import os
import sys

import numpy as np
import torch as th
import yaml
from stable_baselines3.common.utils import set_random_seed

from utils import ALGOS


def cache_hf_repo(organization, repo_name):
    repo_id = f"{organization}/{repo_name}"
    cached_hf_repo_path = snapshot_download(repo_id)
    return cached_hf_repo_path


def move_hf_repo(source, destination):
    if destination.is_dir():
        shutil.rmtree(str(destination))
    shutil.move(source, destination)
    print(f"{source} repo has been moved to {destination}")
    print(f"Now, you can continue to train your agent")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", help="RL Algorithm", default="ppo", type=str, required=False,
                        choices=list(ALGOS.keys()))
    parser.add_argument("--env", help="environment ID", type=str, default="CartPole-v1")
    parser.add_argument("-f", "--folder", help="Log folder", type=str, default="rl-trained-agents")
    parser.add_argument("--organization", help="", type=str, default="sb3")
    parser.add_argument("--repo-name", help="", type=str, default="PPO/CartPole-v1_1")

    parser.add_argument("--exp-id", help="Experiment ID (default: 0: latest, -1: no exp folder)", default=0, type=int)
    try:
        from huggingface_hub import hf_hub_url, snapshot_download
    except ImportError:
        raise ImportError(
            "You need to install huggingface_hub to use `load_from_hf_hub`. "
            "See https://pypi.org/project/huggingface-hub/ for installation."
        )
    # The goal is download the full folder of the repo => put in the correct place
    args = parser.parse_args()

    env_id = args.env
    algo = args.algo
    folder = args.folder

    # If it's a sb3 zoo model
    if args.organization == "sb3":
        ## Generate repo name
        repo_name = f"{algo}-{env_id}"

        # Step 1: Download the hf repo
        cached_hf_repo_path = cache_hf_repo(args.organization, repo_name)

        # Step 2: Place it correctly
        log_path = Path(os.path.join(folder, algo, f"{env_id}_1"))
        move_hf_repo(cached_hf_repo_path, log_path)

    ## Todo: add if it's not a sb3 organization model






