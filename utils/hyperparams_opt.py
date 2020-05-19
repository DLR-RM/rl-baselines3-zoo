import torch.nn as nn
import numpy as np
import optuna
from optuna.pruners import SuccessiveHalvingPruner, MedianPruner
from optuna.samplers import RandomSampler, TPESampler
from optuna.integration.skopt import SkoptSampler
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
# from stable_baselines3.her import HERGoalEnvWrapper

from .callbacks import TrialEvalCallback
from utils import linear_schedule


def hyperparam_optimization(algo, model_fn, env_fn, n_trials=10, n_timesteps=5000, hyperparams=None,
                            n_jobs=1, sampler_method='random', pruner_method='halving',
                            n_startup_trials=10, n_evaluations=20, n_eval_episodes=5, seed=0, verbose=1):
    """
    :param algo: (str)
    :param model_fn: (func) function that is used to instantiate the model
    :param env_fn: (func) function that is used to instantiate the env
    :param n_trials: (int) maximum number of trials for finding the best hyperparams
    :param n_timesteps: (int) maximum number of timesteps per trial
    :param hyperparams: (dict)
    :param n_jobs: (int) number of parallel jobs
    :param sampler_method: (str)
    :param pruner_method: (str)
    :param n_startup_trials: (int)
    :param n_evaluations: (int) Evaluate every 20th of the maximum budget per iteration
    :param n_eval_episodes: (int) Evaluate the model during 5 episodes
    :param seed: (int)
    :param verbose: (int)
    :return: (pd.Dataframe) detailed result of the optimization
    """
    # TODO: eval each hyperparams several times to account for noisy evaluation
    if hyperparams is None:
        hyperparams = {}

    eval_freq = int(n_timesteps / n_evaluations)

    # n_warmup_steps: Disable pruner until the trial reaches the given number of step.
    if sampler_method == 'random':
        sampler = RandomSampler(seed=seed)
    elif sampler_method == 'tpe':
        sampler = TPESampler(n_startup_trials=n_startup_trials, seed=seed)
    elif sampler_method == 'skopt':
        # cf https://scikit-optimize.github.io/#skopt.Optimizer
        # GP: gaussian process
        # Gradient boosted regression: GBRT
        sampler = SkoptSampler(skopt_kwargs={'base_estimator': "GP", 'acq_func': 'gp_hedge'})
    else:
        raise ValueError(f'Unknown sampler: {sampler_method}')

    if pruner_method == 'halving':
        pruner = SuccessiveHalvingPruner(min_resource=1, reduction_factor=4, min_early_stopping_rate=0)
    elif pruner_method == 'median':
        pruner = MedianPruner(n_startup_trials=n_startup_trials, n_warmup_steps=n_evaluations // 3)
    elif pruner_method == 'none':
        # Do not prune
        pruner = MedianPruner(n_startup_trials=n_trials, n_warmup_steps=n_evaluations)
    else:
        raise ValueError(f'Unknown pruner: {pruner_method}')

    if verbose > 0:
        print(f"Sampler: {sampler_method} - Pruner: {pruner_method}")

    study = optuna.create_study(sampler=sampler, pruner=pruner)
    algo_sampler = HYPERPARAMS_SAMPLER[algo]

    def objective(trial):

        kwargs = hyperparams.copy()

        trial.model_class = None
        if algo == 'her':
            trial.model_class = hyperparams['model_class']

        # Hack to use DDPG/TD3 noise sampler
        if algo in ['ddpg', 'td3'] or trial.model_class in ['ddpg', 'td3']:
            trial.n_actions = env_fn(n_envs=1).action_space.shape[0]
        kwargs.update(algo_sampler(trial))

        model = model_fn(**kwargs)
        model.trial = trial

        eval_env = env_fn(n_envs=1, eval_env=True)
        # Account for parallel envs
        eval_freq_ = max(eval_freq // model.get_env().num_envs, 1)
        # TODO: use non-deterministic eval for Atari?
        eval_callback = TrialEvalCallback(eval_env, trial, n_eval_episodes=n_eval_episodes,
                                          eval_freq=eval_freq_, deterministic=True)

        try:
            model.learn(n_timesteps, callback=eval_callback)
            # Free memory
            model.env.close()
            eval_env.close()
        except AssertionError as e:
            # Sometimes, random hyperparams can generate NaN
            # Free memory
            model.env.close()
            eval_env.close()
            # Prune hyperparams that generate NaNs
            print(e)
            raise optuna.exceptions.TrialPruned()
        is_pruned = eval_callback.is_pruned
        cost = -1 * eval_callback.last_mean_reward

        del model.env, eval_env
        del model

        if is_pruned:
            raise optuna.exceptions.TrialPruned()

        return cost

    try:
        study.optimize(objective, n_trials=n_trials, n_jobs=n_jobs)
    except KeyboardInterrupt:
        pass

    print('Number of finished trials: ', len(study.trials))

    print('Best trial:')
    trial = study.best_trial

    print('Value: ', trial.value)

    print('Params: ')
    for key, value in trial.params.items():
        print(f'    {key}: {value}')

    return study.trials_dataframe()


def sample_ppo_params(trial):
    """
    Sampler for PPO2 hyperparams.

    :param trial: (optuna.trial)
    :return: (dict)
    """
    batch_size = trial.suggest_categorical('batch_size', [8, 16, 32, 64, 128, 256, 512])
    n_steps = trial.suggest_categorical('n_steps', [8, 16, 32, 64, 128, 256, 512, 1024, 2048])
    gamma = trial.suggest_categorical('gamma', [0.9, 0.95, 0.98, 0.99, 0.995, 0.999, 0.9999])
    learning_rate = trial.suggest_loguniform('lr', 1e-5, 1)
    lr_schedule = 'constant'
    # lr_schedule = trial.suggest_categorical('lr_schedule', ['linear', 'constant'])
    ent_coef = trial.suggest_loguniform('ent_coef', 0.00000001, 0.1)
    clip_range = trial.suggest_categorical('clip_range', [0.1, 0.2, 0.3, 0.4])
    n_epochs = trial.suggest_categorical('n_epochs', [1, 5, 10, 20])
    gae_lambda = trial.suggest_categorical('gae_lambda', [0.8, 0.9, 0.92, 0.95, 0.98, 0.99, 1.0])
    max_grad_norm = trial.suggest_categorical('max_grad_norm', [0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 2, 5])
    vf_coef = trial.suggest_uniform('vf_coef', 0, 1)
    net_arch = trial.suggest_categorical('net_arch', ['small', 'medium'])
    log_std_init = trial.suggest_uniform('log_std_init', -4, 1)
    sde_sample_freq = trial.suggest_categorical('sde_sample_freq', [-1, 8, 16, 32, 64, 128, 256])
    ortho_init = False
    # ortho_init = trial.suggest_categorical('ortho_init', [False, True])
    # activation_fn = trial.suggest_categorical('activation_fn', ['tanh', 'relu', 'elu', 'leaky_relu'])
    activation_fn = trial.suggest_categorical('activation_fn', ['tanh', 'relu'])

    # TODO: account when using multiple envs
    if batch_size > n_steps:
        batch_size = n_steps

    if lr_schedule == 'linear':
        learning_rate = linear_schedule(learning_rate)

    net_arch = {
        'small': [dict(pi=[64, 64], vf=[64, 64])],
        'medium': [dict(pi=[256, 256], vf=[256, 256])],
    }[net_arch]

    activation_fn = {
        'tanh': nn.Tanh,
        'relu': nn.ReLU,
        'elu': nn.ELU,
        'leaky_relu': nn.LeakyReLU
    }[activation_fn]

    return {
        'n_steps': n_steps,
        'batch_size': batch_size,
        'gamma': gamma,
        'learning_rate': learning_rate,
        'ent_coef': ent_coef,
        'clip_range': clip_range,
        'n_epochs': n_epochs,
        'gae_lambda': gae_lambda,
        'max_grad_norm': max_grad_norm,
        'vf_coef': vf_coef,
        'sde_sample_freq': sde_sample_freq,
        'policy_kwargs': dict(log_std_init=log_std_init, net_arch=net_arch, activation_fn=activation_fn)
    }


def sample_a2c_params(trial):
    """
    Sampler for A2C hyperparams.

    :param trial: (optuna.trial)
    :return: (dict)
    """
    gamma = trial.suggest_categorical('gamma', [0.9, 0.95, 0.98, 0.99, 0.995, 0.999, 0.9999])
    normalize_advantage = trial.suggest_categorical('normalize_advantage', [False, True])
    max_grad_norm = trial.suggest_categorical('max_grad_norm', [0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 2, 5])
    use_rms_prop = trial.suggest_categorical('use_rms_prop', [False, True])
    gae_lambda = trial.suggest_categorical('gae_lambda', [0.8, 0.9, 0.92, 0.95, 0.98, 0.99, 1.0])
    n_steps = trial.suggest_categorical('n_steps', [8, 16, 32, 64, 128, 256, 512, 1024, 2048])
    lr_schedule = trial.suggest_categorical('lr_schedule', ['linear', 'constant'])
    learning_rate = trial.suggest_loguniform('lr', 1e-5, 1)
    ent_coef = trial.suggest_loguniform('ent_coef', 0.00000001, 0.1)
    vf_coef = trial.suggest_uniform('vf_coef', 0, 1)
    log_std_init = trial.suggest_uniform('log_std_init', -4, 1)
    ortho_init = trial.suggest_categorical('ortho_init', [False, True])
    net_arch = trial.suggest_categorical('net_arch', ['small', 'medium'])
    sde_net_arch = trial.suggest_categorical('sde_net_arch', [None, 'tiny', 'small'])
    full_std = trial.suggest_categorical('full_std', [False, True])
    # activation_fn = trial.suggest_categorical('activation_fn', ['tanh', 'relu', 'elu', 'leaky_relu'])
    activation_fn = trial.suggest_categorical('activation_fn', ['tanh', 'relu'])

    if lr_schedule == 'linear':
        learning_rate = linear_schedule(learning_rate)

    net_arch = {
        'small': [dict(pi=[64, 64], vf=[64, 64])],
        'medium': [dict(pi=[256, 256], vf=[256, 256])],
    }[net_arch]

    sde_net_arch = {
        None: None,
        'tiny': [64],
        'small': [64, 64],
    }[sde_net_arch]

    activation_fn = {
        'tanh': nn.Tanh,
        'relu': nn.ReLU,
        'elu': nn.ELU,
        'leaky_relu': nn.LeakyReLU
    }[activation_fn]

    return {
        'n_steps': n_steps,
        'gamma': gamma,
        'gae_lambda': gae_lambda,
        'learning_rate': learning_rate,
        'ent_coef': ent_coef,
        'normalize_advantage': normalize_advantage,
        'max_grad_norm': max_grad_norm,
        'use_rms_prop': use_rms_prop,
        'vf_coef': vf_coef,
        'policy_kwargs': dict(log_std_init=log_std_init, net_arch=net_arch, full_std=full_std,
                              activation_fn=activation_fn, sde_net_arch=sde_net_arch)
    }


def sample_sac_params(trial):
    """
    Sampler for SAC hyperparams.

    :param trial: (optuna.trial)
    :return: (dict)
    """
    gamma = trial.suggest_categorical('gamma', [0.9, 0.95, 0.98, 0.99, 0.995, 0.999, 0.9999])
    learning_rate = trial.suggest_loguniform('lr', 1e-5, 1)
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 128, 256, 512])
    buffer_size = trial.suggest_categorical('buffer_size', [int(1e4), int(1e5), int(1e6)])
    learning_starts = trial.suggest_categorical('learning_starts', [0, 1000, 10000, 20000])
    # train_freq = trial.suggest_categorical('train_freq', [1, 10, 100, 300])
    train_freq = trial.suggest_categorical('train_freq', [8, 16, 32, 64, 128, 256, 512])
    # Polyak coeff
    tau = trial.suggest_categorical('tau', [0.001, 0.005, 0.01, 0.02])
    # gradient_steps takes too much time
    # gradient_steps = trial.suggest_categorical('gradient_steps', [1, 100, 300])
    gradient_steps = train_freq
    # ent_coef = trial.suggest_categorical('ent_coef', ['auto', 0.5, 0.1, 0.05, 0.01, 0.0001])
    ent_coef = 'auto'
    log_std_init = trial.suggest_uniform('log_std_init', -4, 1)
    net_arch = trial.suggest_categorical('net_arch', ["small", "medium", "big"])
    # activation_fn = trial.suggest_categorical('activation_fn', [nn.Tanh, nn.ReLU, nn.ELU, nn.LeakyReLU])

    net_arch = {
        'small': [64, 64],
        'medium': [256, 256],
        'big': [400, 300],
    }[net_arch]

    target_entropy = 'auto'
    # if ent_coef == 'auto':
    #     # target_entropy = trial.suggest_categorical('target_entropy', ['auto', 5, 1, 0, -1, -5, -10, -20, -50])
    #     target_entropy = trial.suggest_uniform('target_entropy', -10, 10)

    return {
        'gamma': gamma,
        'learning_rate': learning_rate,
        'batch_size': batch_size,
        'buffer_size': buffer_size,
        'learning_starts': learning_starts,
        'train_freq': train_freq,
        'gradient_steps': gradient_steps,
        'ent_coef': ent_coef,
        'tau': tau,
        'target_entropy': target_entropy,
        'policy_kwargs': dict(log_std_init=log_std_init, net_arch=net_arch)
    }


def sample_td3_params(trial):
    """
    Sampler for TD3 hyperparams.

    :param trial: (optuna.trial)
    :return: (dict)
    """
    gamma = trial.suggest_categorical('gamma', [0.9, 0.95, 0.98, 0.99, 0.995, 0.999, 0.9999])
    learning_rate = trial.suggest_loguniform('lr', 1e-5, 1)
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 100, 128, 256, 512])
    buffer_size = trial.suggest_categorical('buffer_size', [int(1e4), int(1e5), int(1e6)])

    episodic = trial.suggest_categorical('episodic', [True, False])

    if episodic:
        n_episodes_rollout = 1
        train_freq, gradient_steps = -1, -1
    else:
        train_freq = trial.suggest_categorical('train_freq', [1, 16, 128, 256, 1000, 2000])
        gradient_steps = train_freq
        n_episodes_rollout = -1

    noise_type = trial.suggest_categorical('noise_type', ['ornstein-uhlenbeck', 'normal', None])
    noise_std = trial.suggest_uniform('noise_std', 0, 1)

    net_arch = trial.suggest_categorical('net_arch', ["small", "medium", "big"])
    # activation_fn = trial.suggest_categorical('activation_fn', [nn.Tanh, nn.ReLU, nn.ELU, nn.LeakyReLU])

    net_arch = {
        'small': [64, 64],
        'medium': [256, 256],
        'big': [400, 300],
    }[net_arch]

    hyperparams = {
        'gamma': gamma,
        'learning_rate': learning_rate,
        'batch_size': batch_size,
        'buffer_size': buffer_size,
        'train_freq': train_freq,
        'gradient_steps': gradient_steps,
        'n_episodes_rollout': n_episodes_rollout,
        'policy_kwargs': dict(net_arch=net_arch),
    }

    if noise_type == 'normal':
        hyperparams['action_noise'] = NormalActionNoise(mean=np.zeros(trial.n_actions),
                                                        sigma=noise_std * np.ones(trial.n_actions))
    elif noise_type == 'ornstein-uhlenbeck':
        hyperparams['action_noise'] = OrnsteinUhlenbeckActionNoise(mean=np.zeros(trial.n_actions),
                                                                   sigma=noise_std * np.ones(trial.n_actions))

    return hyperparams


HYPERPARAMS_SAMPLER = {
    'ppo': sample_ppo_params,
    'sac': sample_sac_params,
    'a2c': sample_a2c_params,
    'td3': sample_td3_params
}
