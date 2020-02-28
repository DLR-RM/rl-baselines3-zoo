"""
Plot training reward
"""
import os
import warnings
import argparse

# For tensorflow imported with tensorboard
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import matplotlib.pyplot as plt

from torchy_baselines.common.results_plotter import X_TIMESTEPS, plot_results


parser = argparse.ArgumentParser()
parser.add_argument('algo', type=str)
parser.add_argument('env', type=str)
parser.add_argument('exp_folder', type=str)
args = parser.parse_args()


algo = args.algo
env = args.env
log_path = os.path.join(args.exp_folder, algo)

dirs = [os.path.join(log_path, folder) for folder in os.listdir(log_path) if env in folder]

try:
    plot_results(dirs, 2e6, X_TIMESTEPS, env)
except Exception as e:
    print(e)

plt.legend()
plt.show()
