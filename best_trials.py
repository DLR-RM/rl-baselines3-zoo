import optuna
import json
import numpy as np
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--study-name", help="Study name used during hyperparameter optimization", type=str, default=None)
parser.add_argument("--storage", help="Database storage path used during hyperparameter optimization", type=str, default=None)
parser.add_argument("--print-n-best-trials", help="Show final return values for n best trials", type=int, default=0)
parser.add_argument("--save-n-best-hyperparameters", help="Save the hyperparameters for the n best trials that resulted in the best returns", type=int, default=0)
args = parser.parse_args()

study = optuna.create_study(study_name=args.study_name, storage=args.storage, load_if_exists=True, direction="maximize")

values = []
for i in study.trials:
    if i < args.print_n_best_trials:
        print(i.value)
    values.append(i.value)

scratch_values = [-np.inf if i is None else i for i in values]
ordered_indices = np.argsort(scratch_values)[::-1]

for i in range(args.save_n_best_hyperparameters):
    params = study.trials[ordered_indices[i]].params
    text = json.dumps(params)
    jsonFile = open(str(i) + ".json", "w+")
    jsonFile.write(text)
    jsonFile.close()

# print([values[i] for i in ordered_indices])

# for i in ordered_indices:
#     print(values[i])

# pick max element, put its index in new list, set its value to -200, repeat for length of list

# get max indices

# print(study.trials)


# trial = study.best_trial

# optuna.study.StudySummary(study)
# summary = optuna.study.get_all_study_summaries('mysql://root:dummy@10.128.0.28/pistonball18')

# for i in study.trials:
#     print(i.value)
