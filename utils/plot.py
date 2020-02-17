import os
import warnings
import argparse

# For tensorflow imported with tensorboard
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import matplotlib.pyplot as plt

from torchy_baselines.common.results_plotter import *


parser = argparse.ArgumentParser()
parser.add_argument('algo', type=str)
parser.add_argument('env', type=str)
parser.add_argument('exp_folder', type=str)
args = parser.parse_args()


algo = args.algo
env = args.env
log_path = os.path.join(args.exp_folder, algo)

dirs = [os.path.join(log_path, d) for d in os.listdir(log_path) if env in d]

try:
    plot_results(dirs, 2e6, X_TIMESTEPS, env)
except Exception as e:
    print(e)


merged_mean = []
merged_std = []
plt.figure('results')
for idx, dir_ in enumerate(dirs):
    try:
        log = np.load(os.path.join(dir_, 'evaluations.npz'))
    except FileNotFoundError:
        continue
    # log['results'].shape
    # log['results'].mean(axis=1).shape
    # log['results'].mean(axis=1)
    # log['results'].std(axis=1)
    # print(log['results'].shape)
    mean_ = np.squeeze(log['results'].mean(axis=1))
    std_ = np.squeeze(log['results'].std(axis=1))

    merged_mean.append(mean_)
    merged_std.append(std_)

    plt.plot(log['timesteps'], log['results'].mean(axis=1), label=f'{algo}_{idx + 1}')
    plt.fill_between(log['timesteps'], mean_ + std_, mean_ - std_, alpha=0.5)


plt.figure("Merged results")


mean_ = np.mean(merged_mean, axis=0)
std_ = np.mean(merged_std, axis=0)

plt.plot(log['timesteps'], mean_)
plt.fill_between(log['timesteps'], mean_ + std_, mean_ - std_, alpha=0.5)

plt.legend()
plt.show()
