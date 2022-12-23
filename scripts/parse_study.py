import argparse
import json
import os
import pickle
from pprint import pprint

import optuna
from optuna.trial import FrozenTrial


def value_key(trial: FrozenTrial) -> float:
    # Returns value of trial object for sorting
    if trial.value is None:
        return float("-inf")
    else:
        return trial.value


parser = argparse.ArgumentParser()
parser.add_argument("-i", "--study-file", help="Path to a pickle file contained a saved study", type=str)
parser.add_argument(
    "-f",
    "--folder",
    help="Folder where the best hyperparameter json files will be written",
    type=str,
    default="logs/hyperparameter_jsons",
)
parser.add_argument("--study-name", help="Study name used during hyperparameter optimization", type=str)
parser.add_argument("--storage", help="Database storage path used during hyperparameter optimization", type=str)
parser.add_argument("--print-n-best-trials", help="Show final return values for n best trials", type=int, default=0)
parser.add_argument(
    "--save-n-best-hyperparameters",
    help="Save the hyperparameters for the n best trials that resulted in the best returns",
    type=int,
    default=0,
)
args = parser.parse_args()

if args.study_name is None:
    assert args.study_file is not None, "No --study-file, nor --study-name were provided."
    with open(args.study_file, "rb") as f:
        study = pickle.load(f)

else:
    assert args.storage is not None, "No storage was specified."

    study = optuna.create_study(
        study_name=args.study_name,
        storage=args.storage,
        load_if_exists=True,
        direction="maximize",
    )

trials = study.trials
trials.sort(key=value_key, reverse=True)

for idx, trial in enumerate(trials):
    if idx < args.print_n_best_trials:
        print(f"# Top {idx + 1} - value: {trial.value:.2f}")
        print()
        pprint(trial.params)
        print()

if args.save_n_best_hyperparameters > 0:
    os.makedirs(f"{args.folder}", exist_ok=True)
    for i in range(min(args.save_n_best_hyperparameters, len(trials))):
        params = trials[i].params
        with open(f"{args.folder}/hyperparameters_{i + 1}.json", "w+") as json_file:
            json_file.write(json.dumps(trials[i].params, indent=4))
    print(f"Saved best hyperparameters to {args.folder}")
