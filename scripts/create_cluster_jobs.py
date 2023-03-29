"""
Send multiple jobs to the cluster.
"""
import os
import subprocess
import time
from typing import List

import numpy as np

ALGOS = ["sac"]
ENVS = ["HalfCheetahBulletEnv-v0"]
N_SEEDS = 5
N_EVAL_EPISODES = 10
LOG_STD_INIT = [-6, -5, -4, -3, -2, -1, 0, 1]

os.makedirs(os.path.join("logs", "slurm"), exist_ok=True)

for algo in ALGOS:
    for env_id in ENVS:
        for log_std_init in LOG_STD_INIT:
            log_folder = f"logs_std_{np.exp(log_std_init):.4f}"
            for _ in range(N_SEEDS):
                args = [
                    "--algo",
                    algo,
                    "--env",
                    env_id,
                    "--hyperparams",
                    f'policy_kwargs:"dict(log_std_init={log_std_init}, net_arch=[400, 300])"',
                    "--eval-episodes",
                    N_EVAL_EPISODES,
                    "-f",
                    log_folder,
                    "-uuid",
                ]
                arg_str_list: List[str] = list(map(str, args))

                command = " ".join(["python", "-u", "train.py", *arg_str_list])

                ok = subprocess.call(["sbatch", "cluster_torchy.sh", algo, env_id, "ablation", command])
                time.sleep(0.05)
