import argparse
import difflib
import os
from collections import OrderedDict
from pprint import pprint
import warnings
import importlib
import time

# For pybullet envs
# warnings.filterwarnings("ignore")
import gym
try:
    import pybullet_envs
except ImportError:
    pybullet_envs = None
import numpy as np
import yaml
try:
    import highway_env
except ImportError:
    highway_env = None
# For custom activation fn
import torch.nn as nn # pylint: disable=unused-import

from torchy_baselines.common.utils import set_random_seed
# from torchy_baselines.common.cmd_util import make_atari_env
from torchy_baselines.common.vec_env import VecFrameStack, SubprocVecEnv, VecNormalize, DummyVecEnv
from torchy_baselines.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from torchy_baselines.common.utils import constant_fn

from utils import make_env, ALGOS, linear_schedule, linear_schedule_std, get_latest_run_id, get_wrapper_class
from utils.hyperparams_opt import hyperparam_optimization
from utils.noise import LinearNormalActionNoise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, nargs='+', default=["CartPole-v1"], help='environment ID(s)')
    parser.add_argument('-tb', '--tensorboard-log', help='Tensorboard log dir', default='', type=str)
    parser.add_argument('-i', '--trained-agent', help='Path to a pretrained agent to continue training',
                        default='', type=str)
    parser.add_argument('--algo', help='RL Algorithm', default='ppo',
                        type=str, required=False, choices=list(ALGOS.keys()))
    parser.add_argument('-n', '--n-timesteps', help='Overwrite the number of timesteps', default=-1,
                        type=int)
    parser.add_argument('--log-interval', help='Override log interval (default: -1, no change)', default=-1,
                        type=int)
    parser.add_argument('--eval-freq', help='Evaluate the agent every n steps (if negative, no evaluation)', default=10000,
                        type=int)
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
    parser.add_argument('--verbose', help='Verbose mode (0: no output, 1: INFO)', default=1,
                        type=int)
    parser.add_argument('--gym-packages', type=str, nargs='+', default=[], help='Additional external Gym environment package modules to import (e.g. gym_minigrid)')
    args = parser.parse_args()

    # Going through custom gym packages to let them register in the global registory
    for env_module in args.gym_packages:
        importlib.import_module(env_module)

    env_ids = args.env
    registered_envs = set(gym.envs.registry.env_specs.keys())

    for env_id in env_ids:
        # If the environment is not found, suggest the closest match
        if env_id not in registered_envs:
            try:
                closest_match = difflib.get_close_matches(env_id, registered_envs, n=1)[0]
            except IndexError:
                closest_match = "'no close match found...'"
            raise ValueError('{} not found in gym registry, you maybe meant {}?'.format(env_id, closest_match))

    if args.seed < 0:
        args.seed = int(time.time() + 1000 * np.random.rand())

    set_random_seed(args.seed)

    if args.trained_agent != "":
        assert args.trained_agent.endswith('.zip') and os.path.isfile(args.trained_agent), \
            "The trained_agent must be a valid path to a .zip file"

    rank = 0

    for env_id in env_ids:
        tensorboard_log = None if args.tensorboard_log == '' else os.path.join(args.tensorboard_log, env_id)

        is_atari = False
        if 'NoFrameskip' in env_id:
            is_atari = True

        print("=" * 10, env_id, "=" * 10)
        print("Seed: {}".format(args.seed))

        # Load hyperparameters from yaml file
        with open('hyperparams/{}.yml'.format(args.algo), 'r') as f:
            hyperparams_dict = yaml.safe_load(f)
            if env_id in list(hyperparams_dict.keys()):
                hyperparams = hyperparams_dict[env_id]
            elif is_atari:
                hyperparams = hyperparams_dict['atari']
            else:
                raise ValueError("Hyperparameters not found for {}-{}".format(args.algo, env_id))

        # Sort hyperparams that will be saved
        saved_hyperparams = OrderedDict([(key, hyperparams[key]) for key in sorted(hyperparams.keys())])

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
            print("Using {} environments".format(n_envs))

        # Create schedules
        for key in ['learning_rate', 'clip_range', 'clip_range_vf', 'sde_log_std_scheduler']:
            if key not in hyperparams:
                continue
            if isinstance(hyperparams[key], str):
                schedule, initial_value = hyperparams[key].split('_')
                initial_value = float(initial_value)
                if key == 'sde_log_std_scheduler':
                    hyperparams[key] = linear_schedule_std(initial_value)
                else:
                    hyperparams[key] = linear_schedule(initial_value)
            elif isinstance(hyperparams[key], (float, int)):
                # Negative value: ignore (ex: for clipping)
                if hyperparams[key] < 0 and key != 'sde_log_std_scheduler':
                    continue
                hyperparams[key] = constant_fn(float(hyperparams[key]))
            else:
                raise ValueError('Invalid value for {}: {}'.format(key, hyperparams[key]))

        # Should we overwrite the number of timesteps?
        if args.n_timesteps > 0:
            if args.verbose:
                print("Overwriting n_timesteps with n={}".format(args.n_timesteps))
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

        def create_env(n_envs):
            """
            Create the environment and wrap it if necessary
            :param n_envs: (int)
            :return: (gym.Env)
            """
            global hyperparams

            if is_atari:
                if args.verbose > 0:
                    print("Using Atari wrapper")
                env = make_atari_env(env_id, num_env=n_envs, seed=args.seed)
                # Frame-stacking with 4 frames
                env = VecFrameStack(env, n_stack=4)
            elif algo_ in ['dqn', 'ddpg']:
                if hyperparams.get('normalize', False):
                    print("WARNING: normalization not supported yet for DDPG/DQN")
                env = gym.make(env_id)
                env.seed(args.seed)
                if env_wrapper is not None:
                    env = env_wrapper(env)
            else:
                if n_envs == 1:
                    env = DummyVecEnv([make_env(env_id, 0, args.seed, wrapper_class=env_wrapper)])
                else:
                    # env = SubprocVecEnv([make_env(env_id, i, args.seed) for i in range(n_envs)])
                    # On most env, SubprocVecEnv does not help and is quite memory hungry
                    env = DummyVecEnv([make_env(env_id, i, args.seed, wrapper_class=env_wrapper) for i in range(n_envs)])
                if normalize:
                    if args.verbose > 0:
                        if len(normalize_kwargs) > 0:
                            print("Normalization activated: {}".format(normalize_kwargs))
                        else:
                            print("Normalizing input and reward")
                    env = VecNormalize(env, **normalize_kwargs)
            # Optional Frame-stacking
            if hyperparams.get('frame_stack', False):
                n_stack = hyperparams['frame_stack']
                env = VecFrameStack(env, n_stack)
                print("Stacking {} frames".format(n_stack))
            return env


        env = create_env(n_envs)
        # Create test env if needed, do not normalize reward
        eval_env = None
        if args.eval_freq > 0:
            old_kwargs = None
            if normalize:
                if len(normalize_kwargs) > 0:
                    old_kwargs = normalize_kwargs.copy()
                    normalize_kwargs['norm_reward'] = False
                else:
                    normalize_kwargs = {'norm_reward': False}

            if args.verbose > 0:
                print("Creating test environment")
            eval_env = create_env(1)

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
                    hyperparams['action_noise'] = LinearNormalActionNoise(mean=np.zeros(n_actions),
                                                                          sigma=noise_std * np.ones(n_actions),
                                                                          final_sigma=hyperparams.get('noise_std_final', 0.0) * np.ones(n_actions),
                                                                          max_steps=n_timesteps)
                else:
                    hyperparams['action_noise'] = NormalActionNoise(mean=np.zeros(n_actions),
                                                                    sigma=noise_std * np.ones(n_actions))
            elif 'ornstein-uhlenbeck' in noise_type:
                hyperparams['action_noise'] = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions),
                                                                           sigma=noise_std * np.ones(n_actions))
            else:
                raise RuntimeError('Unknown noise type "{}"'.format(noise_type))
            print("Applying {} noise with std {}".format(noise_type, noise_std))
            del hyperparams['noise_type']
            del hyperparams['noise_std']
            if 'noise_std_final' in hyperparams:
                del hyperparams['noise_std_final']

        if args.trained_agent.endswith('.pkl') and os.path.isfile(args.trained_agent):
            # Continue training
            print("Loading pretrained agent")
            # Policy should not be changed
            del hyperparams['policy']

            model = ALGOS[args.algo].load(args.trained_agent, env=env, seed=args.seed,
                                          tensorboard_log=tensorboard_log, verbose=args.verbose, **hyperparams)

            exp_folder = args.trained_agent.split('.pkl')[0]
            if normalize:
                print("Loading saved running average")
                env.load_running_average(exp_folder)

        elif args.optimize_hyperparameters:

            if args.verbose > 0:
                print("Optimizing hyperparameters")


            def create_model(*_args, **kwargs):
                """
                Helper to create a model with different hyperparameters
                """
                return ALGOS[args.algo](env=create_env(n_envs), tensorboard_log=tensorboard_log,
                                        verbose=0, **kwargs)


            data_frame = hyperparam_optimization(args.algo, create_model, create_env, n_trials=args.n_trials,
                                                 n_timesteps=n_timesteps, hyperparams=hyperparams,
                                                 n_jobs=args.n_jobs, seed=args.seed,
                                                 sampler_method=args.sampler, pruner_method=args.pruner,
                                                 verbose=args.verbose)

            report_name = "report_{}_{}-trials-{}-{}-{}.csv".format(env_id, args.n_trials, n_timesteps,
                                                                    args.sampler, args.pruner)

            log_path = os.path.join(args.log_folder, args.algo, report_name)

            if args.verbose:
                print("Writing report to {}".format(log_path))

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

        model.learn(n_timesteps, eval_env=eval_env, eval_freq=args.eval_freq, **kwargs)

        # Save trained model
        log_path = "{}/{}/".format(args.log_folder, args.algo)
        save_path = os.path.join(log_path, "{}_{}".format(env_id, get_latest_run_id(log_path, env_id) + 1))
        params_path = "{}/{}".format(save_path, env_id)
        os.makedirs(params_path, exist_ok=True)


        print("Saving to {}".format(save_path))

        model.save("{}/{}".format(save_path, env_id))
        # Save hyperparams
        with open(os.path.join(params_path, 'config.yml'), 'w') as f:
            yaml.dump(saved_hyperparams, f)

        if normalize:
            # Unwrap
            if isinstance(env, VecFrameStack):
                env = env.venv
            # Important: save the running average, for testing the agent we need that normalization
            env.save_running_average(params_path)
