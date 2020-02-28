import os
import warnings
import argparse
import pickle

import pytablewriter
import numpy as np
import seaborn
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser('Gather results, plot them and create table')
parser.add_argument('-i', '--input', help='Input filename (numpy archive)', type=str)
parser.add_argument('-skip', '--skip-envs', help='Environments to skip', nargs='+', default=[], type=str)
parser.add_argument('--no-million', action='store_true', default=False,
                    help='Do not convert x-axis to million')
parser.add_argument('--skip-timesteps', action='store_true', default=False,
                    help='Do not display learning curves')
args = parser.parse_args()

# Activate seaborn
seaborn.set()

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

labels = [key for key in results[list(results.keys())[0]].keys()]
envs = [env for env in results.keys() if env not in args.skip_envs]

if not args.skip_timesteps:
    # Plot learning curves per env
    for env in envs:

        plt.figure(f'Results {env}')
        plt.title(f'{env}BulletEnv-v0', fontsize=14)

        x_label_suffix = '' if args.no_million else '(in Million)'
        plt.xlabel(f'Timesteps {x_label_suffix}', fontsize=14)
        plt.ylabel('Score', fontsize=14)

        for key in results[env].keys():
            # x axis in Millions of timesteps
            divider = 1e6
            if args.no_million:
                divider = 1.0

            timesteps = results[env][key]['timesteps']
            mean_ = results[env][key]['mean']
            std_error = results[env][key]['std_error']


            plt.plot(timesteps / divider, mean_, label=key)
            plt.fill_between(timesteps / divider, mean_ + std_error, mean_ - std_error, alpha=0.5)

        plt.legend()

# Plot final results with env as x axis
plt.figure('Sensitivity plot')
plt.title('Sensitivity plot', fontsize=14)
plt.xlabel('Environment', fontsize=14)
plt.ylabel('Score', fontsize=14)

for label in labels:
    values = [np.mean(results[env][label]['last_evals']) for env in envs]
    plt.plot(envs, values, label=label)

plt.legend()
plt.show()
# TODO: export, we need to fix the figure size and axis first
# we also may have to change backend
# plt.savefig("exported.svg", format="svg")
