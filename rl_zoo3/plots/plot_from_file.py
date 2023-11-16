import argparse
import itertools
import pickle
import warnings

import numpy as np
import pandas as pd
import pytablewriter
import seaborn
from matplotlib import pyplot as plt

try:
    from rliable import library as rly
    from rliable import metrics, plot_utils
except ImportError:
    rly = None

from rl_zoo3.plots.score_normalization import normalize_score


# From https://github.com/mwaskom/seaborn/blob/master/seaborn/categorical.py
def restyle_boxplot(artist_dict, color, gray="#222222", linewidth=1, fliersize=5):
    """Take a drawn matplotlib boxplot and make it look nice."""
    for box in artist_dict["boxes"]:
        box.update(dict(facecolor=color, zorder=0.9, edgecolor=gray, linewidth=linewidth))

    for whisk in artist_dict["whiskers"]:
        whisk.update(dict(color=gray, linewidth=linewidth, linestyle="-"))

    for cap in artist_dict["caps"]:
        cap.update(dict(color=gray, linewidth=linewidth))

    for med in artist_dict["medians"]:
        med.update(dict(color=gray, linewidth=linewidth))

    for fly in artist_dict["fliers"]:
        fly.update(dict(markerfacecolor=gray, marker="d", markeredgecolor=gray, markersize=fliersize))


def plot_from_file():  # noqa: C901
    parser = argparse.ArgumentParser("Gather results, plot them and create table")
    parser.add_argument("-i", "--input", help="Input filename (numpy archive)", type=str)
    parser.add_argument("-skip", "--skip-envs", help="Environments to skip", nargs="+", default=[], type=str)
    parser.add_argument("--keep-envs", help="Envs to keep", nargs="+", default=[], type=str)
    parser.add_argument("--skip-keys", help="Keys to skip", nargs="+", default=[], type=str)
    parser.add_argument("--keep-keys", help="Keys to keep", nargs="+", default=[], type=str)
    parser.add_argument("--no-million", action="store_true", default=False, help="Do not convert x-axis to million")
    parser.add_argument("--skip-timesteps", action="store_true", default=False, help="Do not display learning curves")
    parser.add_argument("-o", "--output", help="Output filename (image)", type=str)
    parser.add_argument("--format", help="Output format", type=str, default="svg")
    parser.add_argument("-loc", "--legend-loc", help="The location of the legend.", type=str, default="best")
    parser.add_argument("--figsize", help="Figure size, width, height in inches.", nargs=2, type=int, default=[6.4, 4.8])
    parser.add_argument("--fontsize", help="Font size", type=int, default=14)
    parser.add_argument("-l", "--labels", help="Custom labels", type=str, nargs="+")
    parser.add_argument("-b", "--boxplot", help="Enable boxplot", action="store_true", default=False)
    parser.add_argument("-r", "--rliable", help="Enable rliable plots", action="store_true", default=False)
    parser.add_argument("-vs", "--versus", help="Enable probability of improvement plot", action="store_true", default=False)
    parser.add_argument("-iqm", "--iqm", help="Enable IQM sample efficiency plot", action="store_true", default=False)
    parser.add_argument("-ci", "--ci-size", help="Confidence interval size (for rliable)", type=float, default=0.95)
    parser.add_argument("-latex", "--latex", help="Enable latex support", action="store_true", default=False)
    parser.add_argument("--merge", help="Merge with other results files", nargs="+", default=[], type=str)

    args = parser.parse_args()

    # Activate seaborn
    seaborn.set()
    # Seaborn style
    seaborn.set(style="whitegrid")

    # Enable LaTeX support
    if args.latex:
        plt.rc("text", usetex=True)

    filename = args.input

    if not filename.endswith(".pkl"):
        filename += ".pkl"

    with open(filename, "rb") as file_handler:
        results = pickle.load(file_handler)

    # Plot table
    writer = pytablewriter.MarkdownTableWriter(max_precision=3)
    writer.table_name = "results_table"
    writer.headers = results["results_table"]["headers"]
    writer.value_matrix = results["results_table"]["value_matrix"]
    writer.write_table()

    del results["results_table"]

    for filename in args.merge:
        # Merge other files
        with open(filename, "rb") as file_handler:
            results_2 = pickle.load(file_handler)
            del results_2["results_table"]
            for key in results.keys():
                if key in results_2:
                    for new_key in results_2[key].keys():
                        results[key][new_key] = results_2[key][new_key]

    keys = [key for key in results[next(iter(results.keys()))].keys() if key not in args.skip_keys]
    print(f"keys: {keys}")
    if len(args.keep_keys) > 0:
        keys = [key for key in keys if key in args.keep_keys]
    envs = [env for env in results.keys() if env not in args.skip_envs]

    if len(args.keep_envs) > 0:
        envs = [env for env in envs if env in args.keep_envs]

    labels = {key: key for key in keys}
    if args.labels is not None:
        for key, label in zip(keys, args.labels):
            labels[key] = label

    if not args.skip_timesteps:
        # Plot learning curves per env
        for env in envs:
            plt.figure(f"Results {env}")
            title = f"{env}"  # BulletEnv-v0
            if "Mountain" in env:
                title = "MountainCarContinuous-v0"

            plt.title(title, fontsize=args.fontsize)

            x_label_suffix = "" if args.no_million else "(1e6)"
            plt.xlabel(f"Timesteps {x_label_suffix}", fontsize=args.fontsize)
            plt.ylabel("Score", fontsize=args.fontsize)

            for key in keys:
                # x axis in Millions of timesteps
                divider = 1e6
                if args.no_million:
                    divider = 1.0

                timesteps = results[env][key]["timesteps"]
                mean_ = results[env][key]["mean"]
                std_error = results[env][key]["std_error"]

                plt.xticks(fontsize=13)
                plt.plot(timesteps / divider, mean_, label=labels[key], linewidth=3)
                plt.fill_between(timesteps / divider, mean_ + std_error, mean_ - std_error, alpha=0.5)

            plt.legend(fontsize=args.fontsize)
            plt.tight_layout()

    # Convert to pandas dataframe, in order to use seaborn
    labels_df, envs_df, scores = [], [], []
    # Post-process to use it with rliable
    # algo: (n_runs, n_envs)
    normalized_score_dict = {}
    # algo: (n_runs, n_envs, n_eval)
    all_eval_normalized_scores_dict = {}
    # Convert env key to env id for normalization
    env_key_to_env_id = {
        "Half": "HalfCheetahBulletEnv-v0",
        "Ant": "AntBulletEnv-v0",
        "Hopper": "HopperBulletEnv-v0",
        "Walker": "Walker2DBulletEnv-v0",
    }
    # Backward compat
    skip_all_algos_dict = False

    for key in keys:
        algo_scores, all_algo_scores = [], []
        for env in envs:
            if isinstance(results[env][key]["last_evals"], (np.float32, np.float64)):
                # No enough timesteps
                print(f"Skipping {env}-{key}")
                continue

            for score in results[env][key]["last_evals"]:
                labels_df.append(labels[key])
                # convert to int if needed
                # labels_df.append(int(labels[key]))
                envs_df.append(env)
                scores.append(score)

            algo_scores.append(results[env][key]["last_evals"])

            # Backward compat: mean_per_eval key may not be present
            if "mean_per_eval" in results[env][key]:
                all_algo_scores.append(results[env][key]["mean_per_eval"])
            else:
                skip_all_algos_dict = True

            # Normalize score, env key must match env_id
            if env in env_key_to_env_id:
                algo_scores[-1] = normalize_score(algo_scores[-1], env_key_to_env_id[env])
                if not skip_all_algos_dict:
                    all_algo_scores[-1] = normalize_score(all_algo_scores[-1], env_key_to_env_id[env])
            elif env not in env_key_to_env_id and args.rliable:
                warnings.warn(f"{env} not found for normalizing scores, you should update `env_key_to_env_id`")

        # Truncate to convert to matrix
        min_runs = min(len(algo_score) for algo_score in algo_scores)
        if min_runs > 0:
            algo_scores = [algo_score[:min_runs] for algo_score in algo_scores]
            # shape: (n_envs, n_runs) -> (n_runs, n_envs)
            normalized_score_dict[labels[key]] = np.array(algo_scores).T
            if not skip_all_algos_dict:
                all_algo_scores = [all_algo_score[:, :min_runs] for all_algo_score in all_algo_scores]
                # (n_envs, n_eval, n_runs) -> (n_runs, n_envs, n_eval)
                all_eval_normalized_scores_dict[labels[key]] = np.array(all_algo_scores).transpose((2, 0, 1))

    data_frame = pd.DataFrame(data=dict(Method=labels_df, Environment=envs_df, Score=scores))

    # Rliable plots, see https://github.com/google-research/rliable
    if args.rliable:
        if rly is None:
            raise ImportError(
                "You must install rliable package to use this feature. Note: Python 3.7+ is required in that case."
            )

        print("Computing bootstrap CI ...")
        algorithms = list(labels.values())
        # Scores as a dictionary mapping algorithms to their normalized
        # score matrices, each of which is of size `(num_runs x num_envs)`.

        aggregate_func = lambda x: np.array(  # noqa: E731
            [
                metrics.aggregate_median(x),
                metrics.aggregate_iqm(x),
                metrics.aggregate_mean(x),
                metrics.aggregate_optimality_gap(x),
            ]
        )
        aggregate_scores, aggregate_interval_estimates = rly.get_interval_estimates(
            normalized_score_dict,
            aggregate_func,
            # Default was 50000
            reps=2000,  # Number of bootstrap replications.
            confidence_interval_size=args.ci_size,  # Coverage of confidence interval. Defaults to 95%.
        )

        fig, axes = plot_utils.plot_interval_estimates(
            aggregate_scores,
            aggregate_interval_estimates,
            metric_names=["Median", "IQM", "Mean", "Optimality Gap"],
            algorithms=algorithms,
            xlabel="Normalized Score",
            xlabel_y_coordinate=0.02,
            subfigure_width=5,
            row_height=1,
            max_ticks=4,
            interval_height=0.6,
        )
        fig.canvas.manager.set_window_title("Rliable metrics")
        # Adjust margin to see the x label
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.2)

        # Performance profiles
        # Normalized score thresholds
        normalized_score_thresholds = np.linspace(0.0, 1.5, 50)
        score_distributions, score_distributions_cis = rly.create_performance_profile(
            normalized_score_dict,
            normalized_score_thresholds,
            reps=2000,
            confidence_interval_size=args.ci_size,
        )
        # Plot score distributions
        fig, ax = plt.subplots(ncols=1, figsize=(7, 5))
        plot_utils.plot_performance_profiles(
            score_distributions,
            normalized_score_thresholds,
            performance_profile_cis=score_distributions_cis,
            colors=dict(zip(algorithms, seaborn.color_palette("colorblind"))),
            xlabel=r"Normalized Score $(\tau)$",
            ax=ax,
        )
        fig.canvas.manager.set_window_title("Performance profiles")
        plt.legend()

        # Probability of improvement
        # Scores as a dictionary containing pairs of normalized score
        # matrices for pairs of algorithms we want to compare
        algorithm_pairs_keys = itertools.combinations(algorithms, 2)
        # algorithm_pairs = {.. , 'x,y': (score_x, score_y), ..}
        algorithm_pairs = {}
        for algo1, algo2 in algorithm_pairs_keys:
            algorithm_pairs[f"{algo1}, {algo2}"] = (normalized_score_dict[algo1], normalized_score_dict[algo2])

        if args.versus:
            average_probabilities, average_prob_cis = rly.get_interval_estimates(
                algorithm_pairs,
                metrics.probability_of_improvement,
                reps=1000,  # Default was 50000
                confidence_interval_size=args.ci_size,
            )
            plot_utils.plot_probability_of_improvement(
                average_probabilities,
                average_prob_cis,
                figsize=(10, 8),
                interval_height=0.6,
            )
            plt.gcf().canvas.manager.set_window_title("Probability of Improvement")
            plt.tight_layout()

        if args.iqm:
            # Load scores as a dictionary mapping algorithms to their normalized
            # score matrices across all evaluations, each of which is of size
            # `(n_runs, n_envs, n_eval)` where scores are recorded every n steps.
            # Only compute CI for 1/4 of the evaluations and keep the first and last eval
            downsample_factor = 4
            n_evals = all_eval_normalized_scores_dict[algorithms[0]].shape[-1]
            eval_indices = np.arange(n_evals - 1)[::downsample_factor]
            eval_indices = np.concatenate((eval_indices, [n_evals - 1]))
            eval_indices_scores_dict = {
                algorithm: score[:, :, eval_indices] for algorithm, score in all_eval_normalized_scores_dict.items()
            }
            iqm = lambda scores: np.array(  # noqa: E731
                [metrics.aggregate_iqm(scores[..., eval_idx]) for eval_idx in range(scores.shape[-1])]
            )
            iqm_scores, iqm_cis = rly.get_interval_estimates(
                eval_indices_scores_dict,
                iqm,
                reps=2000,
                confidence_interval_size=args.ci_size,
            )
            plot_utils.plot_sample_efficiency_curve(
                eval_indices + 1,
                iqm_scores,
                iqm_cis,
                algorithms=algorithms,
                # TODO: convert to timesteps using the timesteps
                xlabel=r"Number of Evaluations",
                ylabel="IQM Normalized Score",
            )
            plt.gcf().canvas.manager.set_window_title("IQM Normalized Score - Sample Efficiency Curve")
            plt.legend()
            plt.tight_layout()

        plt.show()

    # Plot final results with env as x axis
    plt.figure("Sensitivity plot", figsize=args.figsize)
    plt.title("Sensitivity plot", fontsize=args.fontsize)
    # plt.title('Influence of the time feature', fontsize=args.fontsize)
    # plt.title('Influence of the network architecture', fontsize=args.fontsize)
    # plt.title('Influence of the exploration variance $log \sigma$', fontsize=args.fontsize)
    # plt.title("Influence of the sampling frequency", fontsize=args.fontsize)
    # plt.title('Parallel vs No Parallel Sampling', fontsize=args.fontsize)
    # plt.title('Influence of the exploration function input', fontsize=args.fontsize)
    plt.title("PyBullet envs", fontsize=args.fontsize)
    plt.xticks(fontsize=13)
    plt.xlabel("Environment", fontsize=args.fontsize)
    plt.ylabel("Score", fontsize=args.fontsize)

    ax = seaborn.barplot(x="Environment", y="Score", hue="Method", data=data_frame)
    # Custom legend title
    handles, labels_legend = ax.get_legend_handles_labels()
    # ax.legend(handles=handles, labels=labels_legend, title=r"$log \sigma$", loc=args.legend_loc)
    # ax.legend(handles=handles, labels=labels_legend, title="Network Architecture", loc=args.legend_loc)
    # ax.legend(handles=handles, labels=labels_legend, title="Interval", loc=args.legend_loc)
    # Old error plot
    # for key in keys:
    #     values = [np.mean(results[env][key]["last_evals"]) for env in envs]
    #     # Overwrite the labels
    #     # labels = {key:i for i, key in enumerate(keys, start=-6)}
    #     plt.errorbar(
    #         envs,
    #         values,
    #         yerr=results[env][key]["std_error"][-1],
    #         linewidth=3,
    #         fmt="-o",
    #         label=labels[key],
    #         capsize=5,
    #         capthick=2,
    #         elinewidth=2,
    #     )
    # plt.legend(fontsize=13, loc=args.legend_loc)
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
        plt.figure("Sensitivity box plot", figsize=args.figsize)
        plt.title("Sensitivity box plot", fontsize=args.fontsize)
        # plt.title('Influence of the exploration variance $log \sigma$ on Hopper', fontsize=args.fontsize)
        # plt.title('Influence of the sampling frequency on Walker2D', fontsize=args.fontsize)
        # plt.title('Influence of the exploration function input on Hopper', fontsize=args.fontsize)
        plt.xticks(fontsize=13)
        # plt.xlabel('Exploration variance $log \sigma$', fontsize=args.fontsize)
        # plt.xlabel("Sampling frequency", fontsize=args.fontsize)
        # plt.xlabel('Method', fontsize=args.fontsize)
        plt.ylabel("Score", fontsize=args.fontsize)

        data, labels_ = [], []
        for env in envs:
            for key in keys:
                data.append(results[env][key]["last_evals"])
                text = f"{env}-{labels[key]}" if len(envs) > 1 else labels[key]
                labels_.append(text)
        artist_dict = plt.boxplot(data, patch_artist=True)
        # Make the boxplot looks nice
        # see https://github.com/mwaskom/seaborn/blob/master/seaborn/categorical.py
        color_palette = seaborn.color_palette()
        # orange
        boxplot_color = color_palette[1]
        restyle_boxplot(artist_dict, color=boxplot_color)
        plt.xticks(np.arange(1, len(data) + 1), labels_, rotation=0)
        plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    plot_from_file()
