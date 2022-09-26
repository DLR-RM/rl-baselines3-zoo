import subprocess

from utils.utils import get_hf_trained_models, get_trained_models

folder = "rl-trained-agents"
orga = "sb3"
trained_models_local = get_trained_models(folder)
trained_models_hub = get_hf_trained_models(orga)
remaining_models = set(trained_models_local.keys()) - set(trained_models_hub.keys())

for trained_model in list(remaining_models):
    algo, env_id = trained_models_local[trained_model]
    args = ["-orga", orga, "-f", folder, "--algo", algo, "--env", env_id]

    # Since SB3 >= 1.1.0, HER is no more an algorithm but a replay buffer class
    if algo == "her":
        continue

    return_code = subprocess.call(["python", "-m", "utils.push_to_hub"] + args)
