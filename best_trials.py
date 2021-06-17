import optuna
study = optuna.create_study(study_name='pistonball21', storage='mysql://root:dummy@10.128.0.28/pistonball21', load_if_exists=True, direction="maximize")
# print(study.best_trial)
values = []
for i in study.trials:
    # print(i.value)
    values.append(i.value)

ordered_indices = []
scratch_values = values.copy()

scratch_values = list(filter(None.__ne__,scratch_values))

for i in range(len(values)):
    index = scratch_values.index(max(values))
    ordered_indices.append(index)
    scratch_values[index] = -200

for i in ordered_indices:
    print(values[i])

# pick max element, put its index in new list, set its value to -200, repeat for length of list



# get max indices

# print(study.trials)


# trial = study.best_trial

# optuna.study.StudySummary(study)
# summary = optuna.study.get_all_study_summaries('mysql://root:dummy@10.128.0.28/pistonball18')

# for i in study.trials:
#     print(i.value)
