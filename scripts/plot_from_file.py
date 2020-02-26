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
parser.add_argument('--no-million', action='store_true', default=False,
                    help='Do not convert x-axis to million')
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

# Plot learning curves
for env in results.keys():

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

plt.show()
