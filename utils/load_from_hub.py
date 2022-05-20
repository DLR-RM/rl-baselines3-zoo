import argparse
import os
import shutil
import zipfile

from huggingface_sb3 import load_from_hub
from requests.exceptions import HTTPError

from utils import ALGOS, get_latest_run_id

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", help="environment ID", type=str, required=True)
    parser.add_argument("-f", "--folder", help="Log folder", type=str, required=True)
    parser.add_argument("-orga", "--organization", help="Huggingface hub organization", default="sb3")
    parser.add_argument("--algo", help="RL Algorithm", type=str, required=True, choices=list(ALGOS.keys()))
    parser.add_argument("--exp-id", help="Experiment ID (default: 0: latest, -1: no exp folder)", default=0, type=int)
    parser.add_argument("--verbose", help="Verbose mode (0: no output, 1: INFO)", default=1, type=int)
    args = parser.parse_args()

    algo = args.algo
    env_id = args.env
    folder = args.folder
    exp_id = args.exp_id

    repo_id = f"{args.organization}/{args.algo}-{env_id}"
    model_name = f"{args.algo}-{env_id}"
    checkpoint = load_from_hub(repo_id, f"{model_name}.zip")
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

    # TODO: check if token is needed for download
    # otherwise just copy the repo

    if exp_id == 0:
        exp_id = get_latest_run_id(os.path.join(folder, algo), env_id) + 1
    # Sanity checks
    if exp_id > 0:
        log_path = os.path.join(folder, algo, f"{env_id}_{exp_id}")
    else:
        log_path = os.path.join(folder, algo)

    # Check that the folder does not exist
    # TODO: Allow use --force to overwrite
    assert not os.path.isdir(log_path), f"The {log_path} folder already exist"

    print(f"Saving to {log_path}")
    os.makedirs(log_path, exist_ok=True)
    config_folder = os.path.join(log_path, env_id)
    os.makedirs(config_folder, exist_ok=True)

    shutil.copy(checkpoint, os.path.join(log_path, f"{env_id}.zip"))
    shutil.copy(saved_args, os.path.join(config_folder, "args.yml"))
    shutil.copy(config_path, os.path.join(config_folder, "config.yml"))
    shutil.copy(env_kwargs, os.path.join(config_folder, "env_kwargs.yml"))
    if vec_normalize_stats is not None:
        shutil.copy(vec_normalize_stats, os.path.join(config_folder, "vecnormalize.pkl"))

    # Extract monitor file and evaluation file
    with zipfile.ZipFile(train_eval_metrics, "r") as zip_ref:
        zip_ref.extractall(log_path)
