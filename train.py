import argparse
import difflib
import os
import importlib
import time
import uuid
from collections import OrderedDict
from pprint import pprint

import yaml
import gym
import seaborn
import numpy as np
import torch as th
# For custom activation fn
import torch.nn as nn  # noqa: F401 pytype: disable=unused-import

from stable_baselines3.common.utils import set_random_seed
# from stable_baselines3.common.cmd_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack, VecNormalize, DummyVecEnv, VecTransposeImage
from stable_baselines3.common.preprocessing import is_image_space
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.utils import constant_fn
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback

# Register custom envs
import utils.import_envs  # noqa: F401 pytype: disable=import-error
from utils import make_env, ALGOS, linear_schedule, get_latest_run_id, get_wrapper_class
from utils.hyperparams_opt import hyperparam_optimization
from utils.callbacks import SaveVecNormalizeCallback
from utils.noise import LinearNormalActionNoise
from utils.utils import StoreDict, get_callback_class

seaborn.set()

if __name__ == '__main__':  # noqa: C901
    parser = argparse.ArgumentParser()
    parser.add_argument('--algo', help='RL Algorithm', default='ppo',
                        type=str, required=False, choices=list(ALGOS.keys()))
    parser.add_argument('--env', type=str, default="CartPole-v1", help='environment ID')
    parser.add_argument('-tb', '--tensorboard-log', help='Tensorboard log dir', default='', type=str)
    parser.add_argument('-i', '--trained-agent', help='Path to a pretrained agent to continue training',
                        default='', type=str)
    parser.add_argument('-n', '--n-timesteps', help='Overwrite the number of timesteps', default=-1,
                        type=int)
    parser.add_argument('--num-threads', help='Number of threads for PyTorch (-1 to use default)', default=-1,
                        type=int)
    parser.add_argument('--log-interval', help='Override log interval (default: -1, no change)', default=-1,
                        type=int)
    parser.add_argument('--eval-freq', help='Evaluate the agent every n steps (if negative, no evaluation)',
                        default=10000, type=int)
    parser.add_argument('--eval-episodes', help='Number of episodes to use for evaluation',
                        default=5, type=int)
    parser.add_argument('--save-freq', help='Save the model every n steps (if negative, no checkpoint)',
                        default=-1, type=int)
    parser.add_argument('--save-replay-buffer', help='Save the replay buffer too (when applicable)',
                        action='store_true', default=False)
    parser.add_argument('-f', '--log-folder', help='Log folder', type=str, default='logs')
    parser.add_argument('--seed', help='Random generator seed', type=int, default=-1)
    parser.add_argument('--n-trials', help='Number of trials for optimizing hyperparameters', type=int, default=10)
    parser.add_argument('-optimize', '--optimize-hyperparameters', action='store_true', default=False,
                        help='Run hyperparameters search')
    parser.add_argument('--n-jobs', help='Number of parallel jobs when optimizing hyperparameters', type=int, default=1)
    parser.add_argument('--sampler', help='Sampler to use when optimizing hyperparameters', type=str,
                        default='tpe', choices=['random', 'tpe', 'skopt'])
    parser.add_argument('--pruner', help='Pruner to use when optimizing hyperparameters', type=str,
                        default='median', choices=['halving', 'median', 'none'])
    parser.add_argument('--n-startup-trials', help='Number of trials before using optuna sampler',
                        type=int, default=10)
    parser.add_argument('--n-evaluations', help='Number of evaluations for hyperparameter optimization',
                        type=int, default=20)
    parser.add_argument('--verbose', help='Verbose mode (0: no output, 1: INFO)', default=1,
                        type=int)
    parser.add_argument('--gym-packages', type=str, nargs='+', default=[],
                        help='Additional external Gym environment package modules to import (e.g. gym_minigrid)')
    parser.add_argument('--env-kwargs', type=str, nargs='+', action=StoreDict,
                        help='Optional keyword argument to pass to the env constructor')
    parser.add_argument('-params', '--hyperparams', type=str, nargs='+', action=StoreDict,
                        help='Overwrite hyperparameter (e.g. learning_rate:0.01 train_freq:10)')
    parser.add_argument('-uuid', '--uuid', action='store_true', default=False,
                        help='Ensure that the run has a unique ID')
    args = parser.parse_args()

    # Going through custom gym packages to let them register in the global registory
    for env_module in args.gym_packages:
        importlib.import_module(env_module)

    env_id = args.env
    registered_envs = set(gym.envs.registry.env_specs.keys())  # pytype: disable=module-attr

    # If the environment is not found, suggest the closest match
    if env_id not in registered_envs:
        try:
            closest_match = difflib.get_close_matches(env_id, registered_envs, n=1)[0]
        except IndexError:
            closest_match = "'no close match found...'"
        raise ValueError(f'{env_id} not found in gym registry, you maybe meant {closest_match}?')

    # Unique id to ensure there is no race condition for the folder creation
    uuid_str = f'_{uuid.uuid4()}' if args.uuid else ''
    if args.seed < 0:
        # Seed but with a random one
        args.seed = np.random.randint(2**32 - 1)

    set_random_seed(args.seed)

    # Setting num threads to 1 makes things run faster on cpu
    if args.num_threads > 0:
        if args.verbose > 1:
            print(f"Setting torch.num_threads to {args.num_threads}")
        th.set_num_threads(args.num_threads)

    if args.trained_agent != "":
        assert args.trained_agent.endswith('.zip') and os.path.isfile(args.trained_agent), \
            "The trained_agent must be a valid path to a .zip file"

    tensorboard_log = None if args.tensorboard_log == '' else os.path.join(args.tensorboard_log, env_id)

    is_atari = False
    if 'NoFrameskip' in env_id:
        is_atari = True

    print("=" * 10, env_id, "=" * 10)
    print(f"Seed: {args.seed}")

    # Load hyperparameters from yaml file
    with open(f'hyperparams/{args.algo}.yml', 'r') as f:
        hyperparams_dict = yaml.safe_load(f)
        if env_id in list(hyperparams_dict.keys()):
            hyperparams = hyperparams_dict[env_id]
        elif is_atari:
            hyperparams = hyperparams_dict['atari']
        else:
            raise ValueError(f"Hyperparameters not found for {args.algo}-{env_id}")

    if args.hyperparams is not None:
        # Overwrite hyperparams if needed
        hyperparams.update(args.hyperparams)
    # Sort hyperparams that will be saved
    saved_hyperparams = OrderedDict([(key, hyperparams[key]) for key in sorted(hyperparams.keys())])
    env_kwargs = {} if args.env_kwargs is None else args.env_kwargs

    algo_ = args.algo
    # HER is only a wrapper around an algo
    if args.algo == 'her':
        algo_ = saved_hyperparams['model_class']
        assert algo_ in {'sac', 'ddpg', 'dqn', 'td3'}, "{} is not compatible with HER".format(algo_)
        # Retrieve the model class
        hyperparams['model_class'] = ALGOS[saved_hyperparams['model_class']]

    if args.verbose > 0:
        pprint(saved_hyperparams)

    n_envs = hyperparams.get('n_envs', 1)

    if args.verbose > 0:
        print(f"Using {n_envs} environments")

    # Create schedules
    for key in ['learning_rate', 'clip_range', 'clip_range_vf']:
        if key not in hyperparams:
            continue
        if isinstance(hyperparams[key], str):
            schedule, initial_value = hyperparams[key].split('_')
            initial_value = float(initial_value)
            hyperparams[key] = linear_schedule(initial_value)
        elif isinstance(hyperparams[key], (float, int)):
            # Negative value: ignore (ex: for clipping)
            if hyperparams[key] < 0:
                continue
            hyperparams[key] = constant_fn(float(hyperparams[key]))
        else:
            raise ValueError(f'Invalid value for {key}: {hyperparams[key]}')

    # Should we overwrite the number of timesteps?
    if args.n_timesteps > 0:
        if args.verbose:
            print(f"Overwriting n_timesteps with n={args.n_timesteps}")
        n_timesteps = args.n_timesteps
    else:
        n_timesteps = int(hyperparams['n_timesteps'])

    normalize = False
    normalize_kwargs = {}
    if 'normalize' in hyperparams.keys():
        normalize = hyperparams['normalize']
        if isinstance(normalize, str):
            normalize_kwargs = eval(normalize)
            normalize = True
        del hyperparams['normalize']

    if 'policy_kwargs' in hyperparams.keys():
        # Convert to python object if needed
        if isinstance(hyperparams['policy_kwargs'], str):
            hyperparams['policy_kwargs'] = eval(hyperparams['policy_kwargs'])

    # Delete keys so the dict can be pass to the model constructor
    if 'n_envs' in hyperparams.keys():
        del hyperparams['n_envs']
    del hyperparams['n_timesteps']

    # obtain a class object from a wrapper name string in hyperparams
    # and delete the entry
    env_wrapper = get_wrapper_class(hyperparams)
    if 'env_wrapper' in hyperparams.keys():
        del hyperparams['env_wrapper']

    log_path = f"{args.log_folder}/{args.algo}/"
    save_path = os.path.join(log_path, f"{env_id}_{get_latest_run_id(log_path, env_id) + 1}{uuid_str}")
    params_path = f"{save_path}/{env_id}"
    os.makedirs(params_path, exist_ok=True)

    callbacks = get_callback_class(hyperparams)
    if 'callback' in hyperparams.keys():
        del hyperparams['callback']

    if args.save_freq > 0:
        # Account for the number of parallel environments
        args.save_freq = max(args.save_freq // n_envs, 1)
        callbacks.append(CheckpointCallback(save_freq=args.save_freq,
                                            save_path=save_path, name_prefix='rl_model', verbose=1))

    def create_env(n_envs, eval_env=False):
        """
        Create the environment and wrap it if necessary
        :param n_envs: (int)
        :param eval_env: (bool) Whether is it an environment used for evaluation or not
        :return: (Union[gym.Env, VecEnv])
        """
        global hyperparams
        global env_kwargs

        # Do not log eval env (issue with writing the same file)
        log_dir = None if eval_env else save_path

        if n_envs == 1:
            env = DummyVecEnv([make_env(env_id, 0, args.seed,
                               wrapper_class=env_wrapper, log_dir=log_dir,
                               env_kwargs=env_kwargs)])
        else:
            # env = SubprocVecEnv([make_env(env_id, i, args.seed) for i in range(n_envs)])
            # On most env, SubprocVecEnv does not help and is quite memory hungry
            env = DummyVecEnv([make_env(env_id, i, args.seed, log_dir=log_dir, env_kwargs=env_kwargs,
                                        wrapper_class=env_wrapper) for i in range(n_envs)])
        if normalize:
            if args.verbose > 0:
                if len(normalize_kwargs) > 0:
                    print(f"Normalization activated: {normalize_kwargs}")
                else:
                    print("Normalizing input and reward")
            env = VecNormalize(env, **normalize_kwargs)

        # Optional Frame-stacking
        if hyperparams.get('frame_stack', False):
            n_stack = hyperparams['frame_stack']
            env = VecFrameStack(env, n_stack)
            print(f"Stacking {n_stack} frames")

        if is_image_space(env.observation_space):
            if args.verbose > 0:
                print("Wrapping into a VecTransposeImage")
            env = VecTransposeImage(env)
        return env

    env = create_env(n_envs)

    # Create test env if needed, do not normalize reward
    eval_env = None
    if args.eval_freq > 0:
        # Account for the number of parallel environments
        args.eval_freq = max(args.eval_freq // n_envs, 1)

        if 'NeckEnv' in env_id:
            # Use the training env as eval env when using the neck
            # because there is only one robot
            # there will be an issue with the reset
            eval_callback = EvalCallback(env, callback_on_new_best=None,
                                         best_model_save_path=save_path,
                                         log_path=save_path, eval_freq=args.eval_freq)
            callbacks.append(eval_callback)
        else:
            # Do not normalize the rewards of the eval env
            old_kwargs = None
            if normalize:
                if len(normalize_kwargs) > 0:
                    old_kwargs = normalize_kwargs.copy()
                    normalize_kwargs['norm_reward'] = False
                else:
                    normalize_kwargs = {'norm_reward': False}

            if args.verbose > 0:
                print("Creating test environment")

            save_vec_normalize = SaveVecNormalizeCallback(save_freq=1, save_path=params_path)
            eval_callback = EvalCallback(create_env(1, eval_env=True), callback_on_new_best=save_vec_normalize,
                                         best_model_save_path=save_path, n_eval_episodes=args.eval_episodes,
                                         log_path=save_path, eval_freq=args.eval_freq,
                                         deterministic=not is_atari)
            callbacks.append(eval_callback)

            # Restore original kwargs
            if old_kwargs is not None:
                normalize_kwargs = old_kwargs.copy()

    # TODO: check for hyperparameters optimization
    # TODO: check What happens with the eval env when using frame stack
    if 'frame_stack' in hyperparams:
        del hyperparams['frame_stack']

    # Stop env processes to free memory
    if args.optimize_hyperparameters and n_envs > 1:
        env.close()

    # Parse noise string for DDPG and SAC
    if algo_ in ['ddpg', 'sac', 'td3'] and hyperparams.get('noise_type') is not None:
        noise_type = hyperparams['noise_type'].strip()
        noise_std = hyperparams['noise_std']
        n_actions = env.action_space.shape[0]
        if 'normal' in noise_type:
            if 'lin' in noise_type:
                final_sigma = hyperparams.get('noise_std_final', 0.0) * np.ones(n_actions)
                hyperparams['action_noise'] = LinearNormalActionNoise(mean=np.zeros(n_actions),
                                                                      sigma=noise_std * np.ones(n_actions),
                                                                      final_sigma=final_sigma,
                                                                      max_steps=n_timesteps)
            else:
                hyperparams['action_noise'] = NormalActionNoise(mean=np.zeros(n_actions),
                                                                sigma=noise_std * np.ones(n_actions))
        elif 'ornstein-uhlenbeck' in noise_type:
            hyperparams['action_noise'] = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions),
                                                                       sigma=noise_std * np.ones(n_actions))
        else:
            raise RuntimeError(f'Unknown noise type "{noise_type}"')
        print(f"Applying {noise_type} noise with std {noise_std}")
        del hyperparams['noise_type']
        del hyperparams['noise_std']
        if 'noise_std_final' in hyperparams:
            del hyperparams['noise_std_final']

    if args.trained_agent.endswith('.zip') and os.path.isfile(args.trained_agent):
        # Continue training
        print("Loading pretrained agent")
        # Policy should not be changed
        del hyperparams['policy']

        if 'policy_kwargs' in hyperparams.keys():
            del hyperparams['policy_kwargs']

        model = ALGOS[args.algo].load(args.trained_agent, env=env, seed=args.seed,
                                      tensorboard_log=tensorboard_log, verbose=args.verbose, **hyperparams)

        exp_folder = args.trained_agent.split('.zip')[0]
        if normalize:
            print("Loading saved running average")
            stats_path = os.path.join(exp_folder, env_id)
            if os.path.exists(os.path.join(stats_path, 'vecnormalize.pkl')):
                env = VecNormalize.load(os.path.join(stats_path, 'vecnormalize.pkl'), env)
            else:
                # Legacy:
                env.load_running_average(exp_folder)

        replay_buffer_path = os.path.join(os.path.dirname(args.trained_agent), 'replay_buffer.pkl')
        if os.path.exists(replay_buffer_path):
            print("Loading replay buffer")
            model.load_replay_buffer(replay_buffer_path)

    elif args.optimize_hyperparameters:

        if args.verbose > 0:
            print("Optimizing hyperparameters")

        def create_model(*_args, **kwargs):
            """
            Helper to create a model with different hyperparameters
            """
            return ALGOS[args.algo](env=create_env(n_envs, eval_env=True), tensorboard_log=tensorboard_log,
                                    verbose=0, **kwargs)

        data_frame = hyperparam_optimization(args.algo, create_model, create_env, n_trials=args.n_trials,
                                             n_timesteps=n_timesteps, hyperparams=hyperparams,
                                             n_jobs=args.n_jobs, seed=args.seed,
                                             sampler_method=args.sampler, pruner_method=args.pruner,
                                             n_startup_trials=args.n_startup_trials, n_evaluations=args.n_evaluations,
                                             verbose=args.verbose)

        report_name = "report_{}_{}-trials-{}-{}-{}_{}.csv".format(env_id, args.n_trials, n_timesteps,
                                                                   args.sampler, args.pruner, int(time.time()))

        log_path = os.path.join(args.log_folder, args.algo, report_name)

        if args.verbose:
            print(f"Writing report to {log_path}")

        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        data_frame.to_csv(log_path)
        exit()
    else:
        # Train an agent from scratch
        model = ALGOS[args.algo](env=env, tensorboard_log=tensorboard_log,
                                 seed=args.seed, verbose=args.verbose, **hyperparams)

    kwargs = {}
    if args.log_interval > -1:
        kwargs = {'log_interval': args.log_interval}

    if len(callbacks) > 0:
        kwargs['callback'] = callbacks

    # Save hyperparams
    with open(os.path.join(params_path, 'config.yml'), 'w') as f:
        yaml.dump(saved_hyperparams, f)

    print(f"Log path: {save_path}")

    try:
        model.learn(n_timesteps, eval_log_path=save_path, eval_env=eval_env, eval_freq=args.eval_freq, **kwargs)
    except KeyboardInterrupt:
        pass

    # Save trained model

    print(f"Saving to {save_path}")
    model.save(f"{save_path}/{env_id}")

    if hasattr(model, 'save_replay_buffer') and args.save_replay_buffer:
        print("Saving replay buffer")
        model.save_replay_buffer(save_path)

    if normalize:
        # Important: save the running average, for testing the agent we need that normalization
        model.get_vec_normalize_env().save(os.path.join(params_path, 'vecnormalize.pkl'))
        # Deprecated saving:
        # env.save_running_average(params_path)
