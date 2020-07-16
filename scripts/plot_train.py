"""
Plot training reward
"""
import argparse
import os

from matplotlib import pyplot as plt
from stable_baselines3.common.results_plotter import X_TIMESTEPS, plot_results

# For tensorflow imported with tensorboard
# import warnings
# warnings.filterwarnings("ignore", category=FutureWarning)


parser = argparse.ArgumentParser()
parser.add_argument("algo", type=str)
parser.add_argument("env", type=str)
parser.add_argument("exp_folder", type=str)
args = parser.parse_args()


algo = args.algo
env = args.env
log_path = os.path.join(args.exp_folder, algo)

dirs = [
    os.path.join(log_path, folder)
    for folder in os.listdir(log_path)
    if (env in folder and os.path.isdir(os.path.join(log_path, folder)))
]

try:
    plot_results(dirs, 2e6, X_TIMESTEPS, env)
except Exception as e:
    print(e)

plt.show()
