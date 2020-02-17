import os
import warnings
import argparse

# For tensorflow imported with tensorboard
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import matplotlib.pyplot as plt

from torchy_baselines.common.results_plotter import *


parser = argparse.ArgumentParser()
parser.add_argument('-a', '--algos', nargs='+', type=str)
parser.add_argument('-e', '--env', type=str)
parser.add_argument('-f', '--exp_folder', nargs='+', type=str)
args = parser.parse_args()


env = args.env

for exp_folder in args.exp_folder:
    for algo in args.algos:
        log_path = os.path.join(exp_folder, algo)

        dirs = [os.path.join(log_path, d) for d in os.listdir(log_path) if env in d]

        max_len = 0
        merged_mean = []
        merged_std = []
        timesteps = None
        plt.figure('results')
        for idx, dir_ in enumerate(dirs):
            try:
                log = np.load(os.path.join(dir_, 'evaluations.npz'))
            except FileNotFoundError:
                print("Eval not found for", dir_)
                continue

            mean_ = np.squeeze(log['results'].mean(axis=1))
            std_ = np.squeeze(log['results'].std(axis=1))

            max_len = max(max_len, len(mean_))
            merged_mean.append(mean_)
            merged_std.append(std_)
            if len(log['timesteps']) == max_len:
                timesteps = log['timesteps']

        # Remove incomplete runs
        mean_tmp, std_tmp = [], []
        for idx in range(len(merged_mean)):
            if len(merged_mean[idx]) == max_len:
                mean_tmp.append(merged_mean[idx])
                std_tmp.append(merged_std[idx])
        merged_mean = mean_tmp
        merged_std = std_tmp

        if len(merged_mean) > 0:
            mean_ = np.mean(merged_mean, axis=0)
            std_ = np.mean(merged_std, axis=0)

            plt.plot(timesteps, mean_, label=f'{algo}-{exp_folder}')
            plt.fill_between(timesteps, mean_ + std_, mean_ - std_, alpha=0.5)

plt.legend()
plt.show()
