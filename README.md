


# Исходный код для статьи Применение методов Deep RL для отбора признаков при обнаружении компьютерных атак в IDS серевого типа



Форк RL pytorch-based фреймворка 
RL Baselines3 Zoo (https://github.com/DLR-RM/stable-baselines3).

## Предобработка наборов данных
```
creating_dataset.ipynb
```

## Обучение агента 

Команда для запуска
```
python train.py --algo ppo
--env cwcf-v0
--n-timesteps 1024000
--optimization-log-path logs/experiment_lambda_1e-2_with_val_set_correct
--eval-episodes 990
--n-eval-envs 1
-optimize --n-trials 500
--sampler tpe
--pruner median
--n-startup-trials 10
--n-evaluations 5
--env-kwargs lambda_coefficient:0.01 mode:'TRAIN' terminal_reward:[[0,-0.3],[-0.7,0]]
--eval-env-kwargs lambda_coefficient:0.01 random_mode:False mode:'VAL' terminal_reward:[[0,-0.3],[-0.7,0]] --tensorboard-log /tmp/stable-baselines/
```
## тестирования агента 

Найденные гиперпараметры определены в  `hyperparameters/algo_name.yml`.
```
sh run.sh
```

##  Для отображения результатов:
```
approb_model_real_data.ipynb
```

```
plot_results.ipynb
```
## Другие методы
```
RandomForestModel_Web_attacks.ipynb
```

```
neural_net.ipynb
```

