import os
import warnings
import argparse

# For tensorflow imported with tensorboard
warnings.filterwarnings("ignore", category=FutureWarning)

import pytablewriter
import numpy as np
import matplotlib.pyplot as plt

from torchy_baselines.common.results_plotter import *


parser = argparse.ArgumentParser()
parser.add_argument('-a', '--algos', nargs='+', type=str)
parser.add_argument('-e', '--env', nargs='+', type=str)
parser.add_argument('-f', '--exp_folder', nargs='+', type=str)
parser.add_argument('-l', '--labels', nargs='+', default=['sde', 'gaussian'], type=str)
args = parser.parse_args()

results = {}

for env in args.env:
    plt.figure(f'Results {env}')
    plt.title(f'{env}')
    results[env] = {}
    for folder_idx, exp_folder in enumerate(args.exp_folder):
        for algo in args.algos:

            results[env][f'{args.labels[folder_idx]}-{algo}'] = 0.0
            log_path = os.path.join(exp_folder, algo)

            dirs = [os.path.join(log_path, d) for d in os.listdir(log_path) if env in d]

            max_len = 0
            merged_mean = []
            merged_std = []
            timesteps = None
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
                # Take last evaluationqq
                results[env][f'{algo}-{args.labels[folder_idx]}'] = mean_[-1]

                plt.plot(timesteps, mean_, label=f'{algo}-{args.labels[folder_idx]}')
                plt.fill_between(timesteps, mean_ + std_, mean_ - std_, alpha=0.5)

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
            value_matrix[i].append(f'{results[env].get(key, 0.0):.2f}')

writer.value_matrix = value_matrix
writer.write_table()


plt.show()
