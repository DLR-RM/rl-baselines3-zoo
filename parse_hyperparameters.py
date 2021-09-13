import argparse
import json
import pickle

import optuna


def value_key(trial: optuna.trial.Trial) -> float:
    # Returns value of trial object for sorting
    if trial.value is None:
        return float("-inf")
    else:
        return trial.value


parser = argparse.ArgumentParser()
parser.add_argument("-i", "--study-file", help="Path to a pickle file contained a saved study", type=str)
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

values = []
trials = study.trials
trials.sort(key=value_key, reverse=True)

for trial in trials:
    if len(values) < args.print_n_best_trials:
        print(trial.value)
    values.append(trial.value)


for i in range(min(args.save_n_best_hyperparameters, len(trials))):
    params = trials[i].params
    text = json.dumps(params)
    with open(f"hyperparameter_jsons/hyperparameters_{i}.json", "w+") as json_file:
        json_file.write(text)
