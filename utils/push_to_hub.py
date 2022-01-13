import argparse
import os
import shutil
from pathlib import Path
import argparse
import glob
import importlib
import os
import sys
from huggingface_hub import HfApi, HfFolder, Repository
import numpy as np
import torch as th
import yaml
from stable_baselines3.common.utils import set_random_seed

import logging
from utils import ALGOS

README_TEMPLATE = """---
tags:
- deep-reinforcement-learning
- reinforcement-learning
- stable-baselines3
---
# TODO: Fill this model card
"""


def _create_model_card(repo_dir: Path):
    """
    Creates a model card for the repository.
    :param repo_dir:
    """
    readme_path = repo_dir / "README.md"
    readme = ""
    if readme_path.exists():
        with readme_path.open("r", encoding="utf8") as f:
            readme = f.read()
    else:
        readme = README_TEMPLATE
    with readme_path.open("w", encoding="utf-8") as f:
        f.write(readme)


def _copy_file(filepath: Path, dst_directory: Path):
    if os.path.exists(dst_directory):
        shutil.rmtree(dst_directory)
        shutil.copytree(filepath, dst_directory)


def push_to_hub(repo_name: str,  # = repo_id
               model_dir: str,  # path where the model is logs/ppo/CartPole_v1_1
               organization: str,
               commit_message: str,
               use_auth_token=True,
               local_repo_path="hub"):
    """
      Upload a model to Hugging Face Hub.
      :param repo_name: name of the model repository from the Hugging Face Hub
      :param organization: name of the organization
      :param commit_message: commit message
      :use_auth_token
      :local_repo_path: local repository path
      """
    huggingface_token = HfFolder.get_token()

    working_dir = Path(os.path.join(model_dir))
    print("Working dir", working_dir)
    if not working_dir.exists() or not working_dir.is_dir():
        raise ValueError(
            f"Can't find path: {serialization_dir}, please point"
            "to a directory with the serialized model."
        )
    # Step 1: Clone or create the repo
    # Create the repo (or clone its content if it's nonempty)
    api = HfApi()
    repo_url = api.create_repo(
        name=repo_name,
        token=huggingface_token,
        organization=organization,
        private=False,
        exist_ok=True, )

    print("Repo url", repo_url)

    # Git pull
    repo_local_path = Path(local_repo_path) / repo_name
    print(repo_local_path)
    repo = Repository(repo_local_path, clone_from=repo_url, use_auth_token=use_auth_token)
    repo.git_pull(rebase=True)

    # Add the model
    # for filename in working_dir.iterdir():
    _copy_file(Path(model_dir), repo_local_path)
    _create_model_card(repo_local_path)

    logging.info(f"Pushing repo {repo_name} to the Hugging Face Hub")
    repo.push_to_hub(commit_message=commit_message)

    logging.info(f"View your model in {repo_url}")

    # Todo: I need to have a feedback like:
    # You can see your model here "https://huggingface.co/repo_url"
    # return repo_url


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--organization", help="", type=str, default="sb3")
    parser.add_argument("--repo-name", help="", type=str, default="PPO/CartPole-v1_1")
    parser.add_argument("--model-dir", type=str)
    parser.add_argument("--commit-message", type=str)

    args = parser.parse_args()
    push_to_hf(repo_name=args.repo_name,
               model_dir=args.model_dir,
               organization=args.organization,
               commit_message=args.commit_message,
               use_auth_token=True,
               local_repo_path="hub")
