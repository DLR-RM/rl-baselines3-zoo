import argparse
import difflib
import importlib
import os
import uuid

import gym
import numpy as np
import seaborn
import torch as th
from stable_baselines3.common.utils import set_random_seed

try:
    from d3rlpy.algos import AWAC, AWR, BC, BCQ, BEAR, CQL, CRR, TD3PlusBC
    from d3rlpy.models.encoders import VectorEncoderFactory
    from d3rlpy.wrappers.sb3 import SB3Wrapper, to_mdp_dataset

    offline_algos = dict(
        awr=AWR,
        awac=AWAC,
        bc=BC,
        bcq=BCQ,
        bear=BEAR,
        cql=CQL,
        crr=CRR,
        td3bc=TD3PlusBC,
    )
except ImportError:
    offline_algos = {}

# Register custom envs
import utils.import_envs  # noqa: F401 pytype: disable=import-error
from utils.exp_manager import ExperimentManager
from utils.utils import ALGOS, StoreDict, evaluate_policy_add_to_buffer

seaborn.set()

if __name__ == "__main__":  # noqa: C901
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", help="RL Algorithm", default="ppo", type=str, required=False, choices=list(ALGOS.keys()))
    parser.add_argument("--env", type=str, default="CartPole-v1", help="environment ID")
    parser.add_argument("-tb", "--tensorboard-log", help="Tensorboard log dir", default="", type=str)
    parser.add_argument("-i", "--trained-agent", help="Path to a pretrained agent to continue training", default="", type=str)
    parser.add_argument(
        "--truncate-last-trajectory",
        help="When using HER with online sampling the last trajectory "
        "in the replay buffer will be truncated after reloading the replay buffer.",
        default=True,
        type=bool,
    )
    parser.add_argument("-n", "--n-timesteps", help="Overwrite the number of timesteps", default=-1, type=int)
    parser.add_argument("--num-threads", help="Number of threads for PyTorch (-1 to use default)", default=-1, type=int)
    parser.add_argument("--log-interval", help="Override log interval (default: -1, no change)", default=-1, type=int)
    parser.add_argument(
        "--eval-freq",
        help="Evaluate the agent every n steps (if negative, no evaluation). "
        "During hyperparameter optimization n-evaluations is used instead",
        default=10000,
        type=int,
    )
    parser.add_argument(
        "--optimization-log-path",
        help="Path to save the evaluation log and optimal policy for each hyperparameter tried during optimization. "
        "Disabled if no argument is passed.",
        type=str,
    )
    parser.add_argument("--eval-episodes", help="Number of episodes to use for evaluation", default=5, type=int)
    parser.add_argument("--n-eval-envs", help="Number of environments for evaluation", default=1, type=int)
    parser.add_argument("--save-freq", help="Save the model every n steps (if negative, no checkpoint)", default=-1, type=int)
    parser.add_argument(
        "--save-replay-buffer", help="Save the replay buffer too (when applicable)", action="store_true", default=False
    )
    parser.add_argument("-f", "--log-folder", help="Log folder", type=str, default="logs")
    parser.add_argument("--seed", help="Random generator seed", type=int, default=-1)
    parser.add_argument("--vec-env", help="VecEnv type", type=str, default="dummy", choices=["dummy", "subproc"])
    parser.add_argument(
        "--n-trials",
        help="Number of trials for optimizing hyperparameters. "
        "This applies to each optimization runner, not the entire optimization process.",
        type=int,
        default=10,
    )
    parser.add_argument(
        "-optimize", "--optimize-hyperparameters", action="store_true", default=False, help="Run hyperparameters search"
    )
    parser.add_argument(
        "--no-optim-plots", action="store_true", default=False, help="Disable hyperparameter optimization plots"
    )
    parser.add_argument("--n-jobs", help="Number of parallel jobs when optimizing hyperparameters", type=int, default=1)
    parser.add_argument(
        "--sampler",
        help="Sampler to use when optimizing hyperparameters",
        type=str,
        default="tpe",
        choices=["random", "tpe", "skopt"],
    )
    parser.add_argument(
        "--pruner",
        help="Pruner to use when optimizing hyperparameters",
        type=str,
        default="median",
        choices=["halving", "median", "none"],
    )
    parser.add_argument("--n-startup-trials", help="Number of trials before using optuna sampler", type=int, default=10)
    parser.add_argument(
        "--n-evaluations",
        help="Training policies are evaluated every n-timesteps // n-evaluations steps when doing hyperparameter optimization",
        type=int,
        default=20,
    )
    parser.add_argument(
        "--storage", help="Database storage path if distributed optimization should be used", type=str, default=None
    )
    parser.add_argument("--study-name", help="Study name for distributed optimization", type=str, default=None)
    parser.add_argument("--verbose", help="Verbose mode (0: no output, 1: INFO)", default=1, type=int)
    parser.add_argument(
        "--gym-packages",
        type=str,
        nargs="+",
        default=[],
        help="Additional external Gym environment package modules to import (e.g. gym_minigrid)",
    )
    parser.add_argument(
        "--env-kwargs", type=str, nargs="+", action=StoreDict, help="Optional keyword argument to pass to the env constructor"
    )
    parser.add_argument(
        "-params",
        "--hyperparams",
        type=str,
        nargs="+",
        action=StoreDict,
        help="Overwrite hyperparameter (e.g. learning_rate:0.01 train_freq:10)",
    )
    parser.add_argument("-uuid", "--uuid", action="store_true", default=False, help="Ensure that the run has a unique ID")
    parser.add_argument(
        "--offline-algo", help="Offline RL Algorithm", type=str, required=False, choices=list(offline_algos.keys())
    )
    parser.add_argument("-b", "--pretrain-buffer", help="Path to a saved replay buffer for pretraining", type=str)
    parser.add_argument(
        "--pretrain-params",
        type=str,
        nargs="+",
        action=StoreDict,
        help="Optional arguments for pretraining with replay buffer",
    )
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
        raise ValueError(f"{env_id} not found in gym registry, you maybe meant {closest_match}?")

    # Unique id to ensure there is no race condition for the folder creation
    uuid_str = f"_{uuid.uuid4()}" if args.uuid else ""
    if args.seed < 0:
        # Seed but with a random one
        args.seed = np.random.randint(2 ** 32 - 1, dtype="int64").item()

    set_random_seed(args.seed)

    # Setting num threads to 1 makes things run faster on cpu
    if args.num_threads > 0:
        if args.verbose > 1:
            print(f"Setting torch.num_threads to {args.num_threads}")
        th.set_num_threads(args.num_threads)

    if args.trained_agent != "":
        assert args.trained_agent.endswith(".zip") and os.path.isfile(
            args.trained_agent
        ), "The trained_agent must be a valid path to a .zip file"

    print("=" * 10, env_id, "=" * 10)
    print(f"Seed: {args.seed}")

    exp_manager = ExperimentManager(
        args,
        args.algo,
        env_id,
        args.log_folder,
        args.tensorboard_log,
        args.n_timesteps,
        args.eval_freq,
        args.eval_episodes,
        args.save_freq,
        args.hyperparams,
        args.env_kwargs,
        args.trained_agent,
        args.optimize_hyperparameters,
        args.storage,
        args.study_name,
        args.n_trials,
        args.n_jobs,
        args.sampler,
        args.pruner,
        args.optimization_log_path,
        n_startup_trials=args.n_startup_trials,
        n_evaluations=args.n_evaluations,
        truncate_last_trajectory=args.truncate_last_trajectory,
        uuid_str=uuid_str,
        seed=args.seed,
        log_interval=args.log_interval,
        save_replay_buffer=args.save_replay_buffer,
        verbose=args.verbose,
        vec_env_type=args.vec_env,
        n_eval_envs=args.n_eval_envs,
        no_optim_plots=args.no_optim_plots,
    )

    # Prepare experiment and launch hyperparameter optimization if needed
    model = exp_manager.setup_experiment()

    if args.pretrain_buffer is not None and model is not None:
        model.load_replay_buffer(args.pretrain_buffer)
        print(f"Buffer size = {model.replay_buffer.buffer_size}")
        # Artificially reduce buffer size
        # model.replay_buffer.full = False
        # model.replay_buffer.pos = 5000

        print(f"{model.replay_buffer.size()} transitions in the replay buffer")

        n_iterations = args.pretrain_params.get("n_iterations", 10)
        n_epochs = args.pretrain_params.get("n_epochs", 1)
        q_func_factory = args.pretrain_params.get("q_func_factory")
        batch_size = args.pretrain_params.get("batch_size", 512)
        # n_action_samples = args.pretrain_params.get("n_action_samples", 1)
        n_eval_episodes = args.pretrain_params.get("n_eval_episodes", 5)
        add_to_buffer = args.pretrain_params.get("add_to_buffer", False)
        deterministic = args.pretrain_params.get("deterministic", True)
        net_arch = args.pretrain_params.get("net_arch", [256, 256])
        scaler = args.pretrain_params.get("scaler", "standard")
        encoder_factory = VectorEncoderFactory(hidden_units=net_arch)
        for arg_name in {
            "n_iterations",
            "n_epochs",
            "q_func_factory",
            "batch_size",
            "n_eval_episodes",
            "add_to_buffer",
            "deterministic",
            "net_arch",
            "scaler",
        }:
            if arg_name in args.pretrain_params:
                del args.pretrain_params[arg_name]
        try:
            assert args.offline_algo is not None and offline_algos is not None
            kwargs_ = {} if q_func_factory is None else dict(q_func_factory=q_func_factory)
            kwargs_.update(dict(encoder_factory=encoder_factory))
            kwargs_.update(args.pretrain_params)

            offline_model = offline_algos[args.offline_algo](
                batch_size=batch_size,
                **kwargs_,
            )
            offline_model = SB3Wrapper(offline_model)
            offline_model.use_sde = False
            # break the logger...
            # offline_model.replay_buffer = model.replay_buffer

            for i in range(n_iterations):
                dataset = to_mdp_dataset(model.replay_buffer)
                offline_model.fit(dataset.episodes, n_epochs=n_epochs, save_metrics=False)

                mean_reward, std_reward = evaluate_policy_add_to_buffer(
                    offline_model,
                    model.get_env(),
                    n_eval_episodes=n_eval_episodes,
                    replay_buffer=model.replay_buffer if add_to_buffer else None,
                    deterministic=deterministic,
                )
                print(f"Iteration {i + 1} training, mean_reward={mean_reward:.2f} +/- {std_reward:.2f}")
        except KeyboardInterrupt:
            pass
        finally:
            print(f"Saving offline model to {exp_manager.save_path}/policy.pt")
            offline_model.save_policy(f"{exp_manager.save_path}/policy.pt")
            # print("Starting training")
            # TODO: convert d3rlpy weights to DB3
            model.env.close()
            exit()

    # Normal training
    if model is not None:
        exp_manager.learn(model)
        exp_manager.save_trained_model(model)
    else:
        exp_manager.hyperparameters_optimization()
