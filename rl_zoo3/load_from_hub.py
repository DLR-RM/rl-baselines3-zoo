import argparse
import os
import shutil
import zipfile
from pathlib import Path
from typing import Optional

from huggingface_sb3 import EnvironmentName, ModelName, ModelRepoId, load_from_hub
from requests.exceptions import HTTPError

from rl_zoo3 import ALGOS, get_latest_run_id


def download_from_hub(
    algo: str,
    env_name: EnvironmentName,
    exp_id: int,
    folder: str,
    organization: str,
    repo_name: Optional[str] = None,
    force: bool = False,
) -> None:
    """
    Try to load a model from the Huggingface hub
    and save it following the RL Zoo structure.
    Default repo name is {organization}/{algo}-{env_id}
    where repo_name = {algo}-{env_id}

    :param algo: Algorithm
    :param env_name: Environment name
    :param exp_id: Experiment id
    :param folder: Log folder
    :param organization: Huggingface organization
    :param repo_name: Overwrite default repository name
    :param force: Allow overwritting the folder
        if it already exists.
    """

    model_name = ModelName(algo, env_name)

    if repo_name is None:
        repo_name = model_name  # Note: model name is {algo}-{env_name}

    # Note: repo id is {organization}/{repo_name}
    repo_id = ModelRepoId(organization, repo_name)
    print(f"Downloading from https://huggingface.co/{repo_id}")

    checkpoint = load_from_hub(repo_id, model_name.filename)
    config_path = load_from_hub(repo_id, "config.yml")

    # If VecNormalize, download
    try:
        vec_normalize_stats = load_from_hub(repo_id, "vec_normalize.pkl")
    except HTTPError:
        print("No normalization file")
        vec_normalize_stats = None

    saved_args = load_from_hub(repo_id, "args.yml")
    env_kwargs = load_from_hub(repo_id, "env_kwargs.yml")
    train_eval_metrics = load_from_hub(repo_id, "train_eval_metrics.zip")

    if exp_id == 0:
        exp_id = get_latest_run_id(os.path.join(folder, algo), env_name) + 1
    # Sanity checks
    if exp_id > 0:
        log_path = os.path.join(folder, algo, f"{env_name}_{exp_id}")
    else:
        log_path = os.path.join(folder, algo)

    # Check that the folder does not exist
    log_folder = Path(log_path)
    if log_folder.is_dir():
        if force:
            print(f"The folder {log_path} already exists, overwritting")
            # Delete the current one to avoid errors
            shutil.rmtree(log_path)
        else:
            raise ValueError(
                f"The folder {log_path} already exists, use --force to overwrite it, "
                "or choose '--exp-id 0' to create a new folder"
            )

    print(f"Saving to {log_path}")
    # Create folder structure
    os.makedirs(log_path, exist_ok=True)
    config_folder = os.path.join(log_path, env_name)
    os.makedirs(config_folder, exist_ok=True)

    # Copy config files and saved stats
    shutil.copy(checkpoint, os.path.join(log_path, f"{env_name}.zip"))
    shutil.copy(saved_args, os.path.join(config_folder, "args.yml"))
    shutil.copy(config_path, os.path.join(config_folder, "config.yml"))
    shutil.copy(env_kwargs, os.path.join(config_folder, "env_kwargs.yml"))
    if vec_normalize_stats is not None:
        shutil.copy(vec_normalize_stats, os.path.join(config_folder, "vecnormalize.pkl"))

    # Extract monitor file and evaluation file
    with zipfile.ZipFile(train_eval_metrics, "r") as zip_ref:
        zip_ref.extractall(log_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", help="environment ID", type=EnvironmentName, required=True)
    parser.add_argument("-f", "--folder", help="Log folder", type=str, required=True)
    parser.add_argument("-orga", "--organization", help="Huggingface hub organization", default="sb3")
    parser.add_argument("-name", "--repo-name", help="Huggingface hub repository name, by default 'algo-env_id'", type=str)
    parser.add_argument("--algo", help="RL Algorithm", type=str, required=True, choices=list(ALGOS.keys()))
    parser.add_argument("--exp-id", help="Experiment ID (default: 0: latest, -1: no exp folder)", default=0, type=int)
    parser.add_argument("--verbose", help="Verbose mode (0: no output, 1: INFO)", default=1, type=int)
    parser.add_argument(
        "--force", action="store_true", default=False, help="Allow overwritting exp folder if it already exist"
    )
    args = parser.parse_args()

    download_from_hub(
        algo=args.algo,
        env_name=args.env,
        exp_id=args.exp_id,
        folder=args.folder,
        organization=args.organization,
        repo_name=args.repo_name,
        force=args.force,
    )
