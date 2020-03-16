import os
import warnings
import argparse
import pickle

import pytablewriter
import seaborn
import numpy as np
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser('Gather results, plot them and create table')
parser.add_argument('-i', '--input', help='Input filename (numpy archive)', type=str)
parser.add_argument('-skip', '--skip-envs', help='Environments to skip', nargs='+', default=[], type=str)
parser.add_argument('--no-million', action='store_true', default=False,
                    help='Do not convert x-axis to million')
parser.add_argument('--skip-timesteps', action='store_true', default=False,
                    help='Do not display learning curves')
parser.add_argument('-o', '--output', help='Output filename (image)', type=str)
parser.add_argument('--format', help='Output format', type=str, default='svg')
parser.add_argument('-loc', '--legend-loc', help='The location of the legend.', type=str, default='best')
parser.add_argument('--figsize', help='Figure size, width, height in inches.', nargs=2, type=int, default=[6.4, 4.8])
parser.add_argument('--fontsize', help='Font size', type=int, default=14)
parser.add_argument('-l', '--labels', help='Custom labels', type=str, nargs='+')
parser.add_argument('-b', '--boxplot', help='Enable boxplot', action='store_true', default=False)
parser.add_argument('-latex', '--latex', help='Enable latex support', action='store_true', default=False)

args = parser.parse_args()

# Activate seaborn
seaborn.set()
# Enable LaTeX support
if args.latex:
    plt.rc('text', usetex=True)

filename = args.input

if not filename.endswith('.pkl'):
    filename += '.pkl'

with open(filename, 'rb') as file_handler:
    results = pickle.load(file_handler)

# Plot table
writer = pytablewriter.MarkdownTableWriter()
writer.table_name = "results_table"
writer.headers = results['results_table']['headers']
writer.value_matrix = results['results_table']['value_matrix']
writer.write_table()

del results['results_table']

keys = [key for key in results[list(results.keys())[0]].keys()]
envs = [env for env in results.keys() if env not in args.skip_envs]
labels = {key:key for key in keys}
if args.labels is not None:
    for key, label in zip(keys, args.labels):
        labels[key] = label

if not args.skip_timesteps:
    # Plot learning curves per env
    for env in envs:

        plt.figure(f'Results {env}')
        title = f'{env}BulletEnv-v0'
        if 'Mountain' in env:
            title = 'MountainCarContinuous-v0'

        plt.title(title, fontsize=args.fontsize)

        x_label_suffix = '' if args.no_million else '(1e6)'
        plt.xlabel(f'Timesteps {x_label_suffix}', fontsize=args.fontsize)
        plt.ylabel('Score', fontsize=args.fontsize)

        for key in keys:
            # x axis in Millions of timesteps
            divider = 1e6
            if args.no_million:
                divider = 1.0

            timesteps = results[env][key]['timesteps']
            mean_ = results[env][key]['mean']
            std_error = results[env][key]['std_error']

            plt.xticks(fontsize=13)
            plt.plot(timesteps / divider, mean_, label=labels[key], linewidth=3)
            plt.fill_between(timesteps / divider, mean_ + std_error, mean_ - std_error, alpha=0.5)

        plt.legend(fontsize=args.fontsize)
        plt.tight_layout()

# Plot final results with env as x axis
plt.figure('Sensitivity plot', figsize=args.figsize)
plt.title('Sensitivity plot', fontsize=args.fontsize)
# plt.title('Influence of the time feature', fontsize=args.fontsize)
# plt.title('Influence of the network architecture', fontsize=args.fontsize)
# plt.title('Influence of the exploration variance $log \sigma$', fontsize=args.fontsize)
# plt.title('Influence of the sampling frequency', fontsize=args.fontsize)
# plt.title('Parallel vs No Parallel Sampling', fontsize=args.fontsize)
# plt.title('Influence of the exploration function input', fontsize=args.fontsize)
plt.xticks(fontsize=13)
plt.xlabel('Environment', fontsize=args.fontsize)
plt.ylabel('Score', fontsize=args.fontsize)

for key in keys:
    values = [np.mean(results[env][key]['last_evals']) for env in envs]
    # Overwrite the labels
    # labels = {key:i for i, key in enumerate(keys, start=-6)}
    plt.errorbar(envs, values, yerr=results[env][key]['std_error'][-1],
                 linewidth=3, fmt='-o', label=labels[key], capsize=5, capthick=2, elinewidth=2)

plt.legend(fontsize=13, loc=args.legend_loc)
plt.tight_layout()
if args.output is not None:
    plt.savefig(args.output, format=args.format)

# Plot final results with env as labels and method as x axis
# plt.figure('Sensitivity plot inverted', figsize=args.figsize)
# plt.title('Sensitivity plot', fontsize=args.fontsize)
# plt.xticks(fontsize=13)
# # plt.xlabel('Method', fontsize=args.fontsize)
# plt.ylabel('Score', fontsize=args.fontsize)
#
# for env in envs:
#     values = [np.mean(results[env][key]['last_evals']) for key in keys]
#     # Overwrite the labels
#     # labels = {key:i for i, key in enumerate(keys, start=-6)}
#     plt.errorbar(labels.values(), values, yerr=results[env][key]['std_error'][-1],
#                  linewidth=3, fmt='-o', label=env, capsize=5, capthick=2, elinewidth=2)
#
# plt.legend(fontsize=13, loc=args.legend_loc)
# plt.tight_layout()

if args.boxplot:
    # Box plot
    plt.figure('Sensitivity box plot', figsize=args.figsize)
    plt.title('Sensitivity box plot', fontsize=args.fontsize)
    # plt.title('Influence of the exploration variance $log \sigma$ on Hopper', fontsize=args.fontsize)
    # plt.title('Influence of the sampling frequency on Walker2D', fontsize=args.fontsize)
    # plt.title('Influence of the exploration function input on Hopper', fontsize=args.fontsize)
    plt.xticks(fontsize=13)
    # plt.xlabel('Exploration variance $log \sigma$', fontsize=args.fontsize)
    plt.xlabel('Sampling frequency', fontsize=args.fontsize)
    # plt.xlabel('Method', fontsize=args.fontsize)
    plt.ylabel('Score', fontsize=args.fontsize)

    data, labels_ = [], []
    for env in envs:
        for key in keys:
            data.append(results[env][key]['last_evals'])
            text = f'{env}-{labels[key]}' if len(envs) > 1 else labels[key]
            labels_.append(text)
    plt.boxplot(data)
    plt.xticks(np.arange(1, len(data) + 1), labels_, rotation=0)
    plt.tight_layout()

plt.show()
