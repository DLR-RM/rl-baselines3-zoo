import subprocess

from huggingface_sb3 import load_from_hub
from requests.exceptions import HTTPError

from utils import get_trained_models

folder = "rl-trained-agents"
trained_models = get_trained_models(folder)
orga = "sb3"

for trained_model in trained_models.keys():
    algo, env_id = trained_models[trained_model]
    args = ["-orga", orga, "-f", folder, "--algo", algo, "--env", env_id]

    # Since SB3 >= 1.1.0, HER is no more an algorithm but a replay buffer class
    if algo == "her":
        continue

    # if model doesn't exist already
    model_exists = False
    model_name = f"{algo}-{env_id}"
    repo_name = f"{algo}-{env_id}"
    repo_id = f"{orga}/{repo_name}"
    try:
        checkpoint = load_from_hub(repo_id, f"{model_name}.zip")
        model_exists = True
    except HTTPError:
        pass

    if model_exists:
        print(f"Skipping {repo_name}")
        continue

    return_code = subprocess.call(["python", "-m", "utils.push_to_hub"] + args)
