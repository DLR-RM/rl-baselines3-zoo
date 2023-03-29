"""
Run multiple experiments on a single machine.
"""
import subprocess
from typing import List

import numpy as np

ALGOS = ["sac"]
ENVS = ["MountainCarContinuous-v0"]
N_SEEDS = 10
EVAL_FREQ = 5000
N_EVAL_EPISODES = 10
LOG_STD_INIT = [-6, -5, -4, -3, -2, -1, 0, 1]

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
                    f"policy_kwargs:dict(log_std_init={log_std_init}, net_arch=[64, 64])",
                    "--eval-episodes",
                    N_EVAL_EPISODES,
                    "--eval-freq",
                    EVAL_FREQ,
                    "-f",
                    log_folder,
                ]
                arg_str_list: List[str] = list(map(str, args))

                ok = subprocess.call(["python", "train.py", *arg_str_list])
