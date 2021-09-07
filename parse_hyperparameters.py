import argparse
import json

import optuna


def value_key(trial: optuna.trial.Trial):
    # Returns value of trial object for sorting
    if trial.value is None:
        return float("-inf")
    else:
        return trial.value


parser = argparse.ArgumentParser()
parser.add_argument(
    "--study-name",
    help="Study name used during hyperparameter optimization",
    type=str,
    default=None,
)
parser.add_argument(
    "--storage",
    help="Database storage path used during hyperparameter optimization",
    type=str,
    default=None,
)
parser.add_argument(
    "--print-n-best-trials",
    help="Show final return values for n best trials",
    type=int,
    default=0,
)
parser.add_argument(
    "--save-n-best-hyperparameters",
    help="Save the hyperparameters for the n best trials that resulted in the best returns",
    type=int,
    default=0,
)
args = parser.parse_args()

study = optuna.create_study(
    study_name=args.study_name,
    storage=args.storage,
    load_if_exists=True,
    direction="maximize",
)
values = []
trials = study.trials
trials.sort(key=value_key, reverse=True)

for i in trials:
    if len(values) < args.print_n_best_trials:
        print(i.value)
    values.append(i.value)

for i in range(args.save_n_best_hyperparameters):
    params = trials[i].params
    text = json.dumps(params)
    json_file = open(f"hyperparameter_jsons/hyperparameters_{i}.json" + ".json", "w+")
    json_file.write(text)
    json_file.close()
