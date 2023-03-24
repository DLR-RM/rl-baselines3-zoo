import os
import subprocess
import time

import numpy as np

ALGOS = ["sac", "td3", "tqc"]
# "Humanoid-v3",
ENVS = ["HalfCheetah-v3", "Ant-v3", "Hopper-v3", "Walker2d-v3", "Swimmer-v3"]
N_SEEDS = 1
EVAL_FREQ = 25000
N_EVAL_EPISODES = 20
N_EVAL_ENVS = 5
np.random.seed(8)
SEEDS = np.random.randint(2**20, size=(N_SEEDS,))
# N_TIMESTEPS = int(1e6)

os.makedirs(os.path.join("logs", "slurm"), exist_ok=True)
log_folder = "logs/"


for algo in ALGOS:
    for env_id in ENVS:
        for seed in SEEDS:
            args = [
                "--algo",
                algo,
                "--env",
                env_id,
                # "--hyperparams",
                # "use_sde:False",
                "--eval-episodes",
                N_EVAL_EPISODES,
                "--eval-freq",
                EVAL_FREQ,
                "--n-eval-envs",
                N_EVAL_ENVS,
                "-f",
                log_folder,
                "--seed",
                seed,
                "--log-interval",
                10,
                "--num-threads",
                2,
                # "-n",
                # N_TIMESTEPS,
                "-uuid",
            ]
            args = list(map(str, args))

            command = " ".join(["python", "-u", "train.py", *args])

            ok = subprocess.call(["sbatch", "cluster_torchy.sh", algo, env_id, "ablation", command])
            time.sleep(0.05)
