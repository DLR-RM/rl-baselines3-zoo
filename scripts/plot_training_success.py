"""
Plot training reward
"""
import argparse
import os

import numpy as np
import seaborn
from matplotlib import pyplot as plt
from stable_baselines3.common.monitor import get_monitor_files, load_results
from stable_baselines3.common.results_plotter import X_EPISODES, X_TIMESTEPS, X_WALLTIME, ts2xy, window_func

# For tensorflow imported with tensorboard
# import warnings
# warnings.filterwarnings("ignore", category=FutureWarning)


# Activate seaborn
seaborn.set()

parser = argparse.ArgumentParser("Gather results, plot training success")
parser.add_argument("-a", "--algo", help="Algorithm to include", type=str, required=True)
parser.add_argument("-e", "--env", help="Environment to include", type=str, required=True)
parser.add_argument("-f", "--exp-folder", help="Folders to include", type=str, required=True)
parser.add_argument("--figsize", help="Figure size, width, height in inches.", nargs=2, type=int, default=[6.4, 4.8])
parser.add_argument("--fontsize", help="Font size", type=int, default=14)
parser.add_argument("-max", "--max-timesteps", help="Max number of timesteps to display", type=int)
parser.add_argument("-x", "--x-axis", help="X-axis", choices=["steps", "episodes", "time"], type=str, default="steps")
parser.add_argument("-w", "--episode-window", help="Rolling window size", type=int, default=100)

args = parser.parse_args()


algo = args.algo
env = args.env
log_path = os.path.join(args.exp_folder, algo)

x_axis = {"steps": X_TIMESTEPS, "episodes": X_EPISODES, "time": X_WALLTIME}[args.x_axis]

x_label = {"steps": "Timesteps", "episodes": "Episodes", "time": "Walltime (in hours)"}[args.x_axis]

dirs = []

for folder in os.listdir(log_path):
    path = os.path.join(log_path, folder)
    if env in folder and os.path.isdir(path):
        monitor_files = get_monitor_files(path)
        if len(monitor_files) > 0:
            dirs.append(path)

plt.figure("Training Success Rate", figsize=args.figsize)
plt.title("Training Success Rate", fontsize=args.fontsize)
plt.xlabel(f"{x_label}", fontsize=args.fontsize)
plt.ylabel("Success Rate", fontsize=args.fontsize)
for folder in dirs:
    data_frame = load_results(folder)
    if args.max_timesteps is not None:
        data_frame = data_frame[data_frame.l.cumsum() <= args.max_timesteps]
    success = np.array(data_frame["is_success"])
    x, _ = ts2xy(data_frame, x_axis)

    # Do not plot the smoothed curve at all if the timeseries is shorter than window size.
    if x.shape[0] >= args.episode_window:
        # Compute and plot rolling mean with window of size args.episode_window
        x, y_mean = window_func(x, success, args.episode_window, np.mean)
        plt.plot(x, y_mean, linewidth=2)
plt.tight_layout()
plt.show()
