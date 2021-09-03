import optuna
import json
import argparse


def value_key(a):
    if a.value is None:
        return float('-inf')
    else:
        return a.value


parser = argparse.ArgumentParser()
parser.add_argument("--study-name", help="Study name used during hyperparameter optimization", type=str, default=None)
parser.add_argument("--storage", help="Database storage path used during hyperparameter optimization", type=str, default=None)
parser.add_argument("--print-n-best-trials", help="Show final return values for n best trials", type=int, default=0)
parser.add_argument("--save-n-best-hyperparameters", help="Save the hyperparameters for the n best trials that resulted in the best returns", type=int, default=0)
args = parser.parse_args()

study = optuna.create_study(study_name=args.study_name, storage=args.storage, load_if_exists=True, direction="maximize")
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
    jsonFile = open('hyperparameter_jsons/' + 'hyperparameters_' + str(i) + ".json", "w+")
    jsonFile.write(text)
    jsonFile.close()
