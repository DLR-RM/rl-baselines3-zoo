import os
import warnings
import argparse

# For tensorflow imported with tensorboard
warnings.filterwarnings("ignore", category=FutureWarning)

import pytablewriter
import numpy as np
import seaborn
import matplotlib.pyplot as plt

from torchy_baselines.common.results_plotter import *


parser = argparse.ArgumentParser('Gather results, plot them and create table')
parser.add_argument('-a', '--algos', help='Algorithms to include', nargs='+', type=str)
parser.add_argument('-e', '--env', help='Environments to include', nargs='+', type=str)
parser.add_argument('-f', '--exp_folder', help='Folders to include', nargs='+', type=str)
parser.add_argument('-l', '--labels', help='Label for each folder', nargs='+', default=['sde', 'gaussian'], type=str)
parser.add_argument('-median', '--median', action='store_true', default=False,
                    help='Display median instead of mean in the table')
args = parser.parse_args()

# Activate seaborn
seaborn.set()
results = {}

args.algos = [algo.upper() for algo in args.algos]

for env in args.env:
    plt.figure(f'Results {env}')
    plt.title(f'{env}BulletEnv-v0', fontsize=14)
    plt.xlabel('Timesteps (in Million)', fontsize=14)
    plt.ylabel('Score', fontsize=14)
    results[env] = {}
    for algo in args.algos:
        for folder_idx, exp_folder in enumerate(args.exp_folder):

            results[env][f'{args.labels[folder_idx]}-{algo}'] = 0.0
            log_path = os.path.join(exp_folder, algo.lower())

            dirs = [os.path.join(log_path, d) for d in os.listdir(log_path) if env in d]

            max_len = 0
            merged_mean, merged_std = [], []
            last_eval = []
            timesteps = None
            for idx, dir_ in enumerate(dirs):
                try:
                    log = np.load(os.path.join(dir_, 'evaluations.npz'))
                except FileNotFoundError:
                    print("Eval not found for", dir_)
                    continue

                mean_ = np.squeeze(log['results'].mean(axis=1))
                #
                # TODO: Compute standard error for all the runs
                # std_ = np.squeeze(log['results'].std(axis=1)) / np.sqrt(log['results'].shape[1])
                std_ = np.squeeze(log['results'].std(axis=1))

                merged_mean.append(mean_)
                merged_std.append(std_)
                last_eval.append(log['results'][-1])

                max_len = max(max_len, len(mean_))
                if len(log['timesteps']) == max_len:
                    timesteps = log['timesteps']

            # Remove incomplete runs
            mean_tmp, std_tmp, last_eval_tmp = [], [], []
            for idx in range(len(merged_mean)):
                if len(merged_mean[idx]) == max_len:
                    mean_tmp.append(merged_mean[idx])
                    std_tmp.append(merged_std[idx])
                    last_eval_tmp.append(last_eval[idx])
            merged_mean = mean_tmp
            merged_std = std_tmp
            last_eval = last_eval_tmp

            if len(merged_mean) > 0:
                mean_ = np.mean(merged_mean, axis=0)
                # TODO: fix standard error computation
                std_ = np.mean(merged_std, axis=0) / np.sqrt(len(merged_mean))
                # Take last evaluation
                # Compute standard error: std / sqrt(n_runs)
                std_error = np.std(last_eval) / np.sqrt(len(last_eval))

                if args.median:
                    results[env][f'{algo}-{args.labels[folder_idx]}'] = f'{np.median(last_eval):.0f}'
                else:
                    results[env][f'{algo}-{args.labels[folder_idx]}'] = f'{np.mean(last_eval):.0f} +/- {std_error:.0f}'


                # x axis in Millions of timesteps
                plt.plot(timesteps / 1e6, mean_, label=f'{algo}-{args.labels[folder_idx]}')
                plt.fill_between(timesteps / 1e6, mean_ + std_, mean_ - std_, alpha=0.5)

    plt.legend()


writer = pytablewriter.MarkdownTableWriter()
writer.table_name = "results_table"

headers = ["Environments"]

# One additional row for the subheader
value_matrix = [[] for i in range(len(args.env) + 1)]

headers = ["Environments"]
# Header and sub-header
value_matrix[0].append('')
for algo in args.algos:
    for label in args.labels:
        value_matrix[0].append(label)
        headers.append(algo)

writer.headers = headers


for i, env in enumerate(args.env, start=1):
    value_matrix[i].append(env)
    for algo in args.algos:
        for label in args.labels:
            key = f'{algo}-{label}'
            value_matrix[i].append(f'{results[env].get(key, "0.0 +/- 0.0")}')

writer.value_matrix = value_matrix
writer.write_table()


plt.show()
