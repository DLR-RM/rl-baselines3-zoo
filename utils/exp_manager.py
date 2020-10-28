import argparse
import os
import time
import warnings
from collections import OrderedDict
from pprint import pprint
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import yaml
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.preprocessing import is_image_space
from stable_baselines3.common.sb2_compat.rmsprop_tf_like import RMSpropTFLike  # noqa: F401
from stable_baselines3.common.utils import constant_fn
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv, VecFrameStack, VecNormalize, VecTransposeImage
from stable_baselines3.common.vec_env.obs_dict_wrapper import ObsDictWrapper

# For custom activation fn
from torch import nn as nn  # noqa: F401

# Register custom envs
import utils.import_envs  # noqa: F401 pytype: disable=import-error
from utils.callbacks import SaveVecNormalizeCallback
from utils.hyperparams_opt import hyperparam_optimization
from utils.noise import LinearNormalActionNoise
from utils.utils import ALGOS, get_callback_class, get_latest_run_id, get_wrapper_class, linear_schedule, make_env


class ExperimentManager(object):
    def __init__(
        self,
        args: argparse.Namespace,
        algo: str,
        env_id: str,
        log_folder: str,
        tensorboard_log: str = "",
        n_timesteps: int = 0,
        eval_freq: int = 10000,
        n_eval_episodes: int = 5,
        save_freq: int = -1,
        hyperparams: Optional[Dict[str, Any]] = None,
        env_kwargs: Optional[Dict[str, Any]] = None,
        trained_agent: str = "",
        optimize_hyperparameters: bool = False,
        storage: Optional[str] = None,
        study_name: Optional[str] = None,
        n_trials: int = 1,
        n_jobs: int = 1,
        sampler: str = "tpe",
        pruner: str = "median",
        n_startup_trials: int = 0,
        n_evaluations: int = 1,
        is_atari: bool = False,
        truncate_last_trajectory: bool = False,
        uuid_str: str = "",
        seed: int = 0,
        log_interval: int = 0,
        save_replay_buffer: bool = False,
        verbose: int = 1,
    ):
        super(ExperimentManager, self).__init__()
        self.algo = algo
        self.env_id = env_id
        self.log_folder = log_folder
        self.hyperparams = hyperparams
        self.verbose = verbose
        self.env_kwargs = {} if env_kwargs is None else env_kwargs
        self.n_timesteps = n_timesteps
        self.save_freq = save_freq
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.callbacks = []
        self.n_envs = 1
        self.trained_agent = trained_agent
        self.seed = seed
        self.normalize = False
        self.normalize_kwargs = {}
        self.env_wrapper = None
        self.tensorboard_log = None if tensorboard_log == "" else os.path.join(tensorboard_log, env_id)
        self.is_atari = is_atari
        self.frame_stack = None
        self.continue_training = trained_agent.endswith(".zip") and os.path.isfile(trained_agent)
        self.truncate_last_trajectory = truncate_last_trajectory

        self.optimize_hyperparameters = optimize_hyperparameters
        self.storage = storage
        self.study_name = study_name
        self.n_trials = n_trials
        self.n_jobs = n_jobs
        self.sampler = sampler
        self.pruner = pruner
        self.n_startup_trials = n_startup_trials
        self.n_evaluations = n_evaluations

        self.args = args
        self.log_interval = log_interval
        self.save_replay_buffer = save_replay_buffer

        self.log_path = f"{log_folder}/{self.algo}/"
        self.save_path = os.path.join(
            self.log_path, f"{self.env_id}_{get_latest_run_id(self.log_path, self.env_id) + 1}{uuid_str}"
        )
        self.params_path = f"{self.save_path}/{self.env_id}"

    def setup_experiment(self) -> Optional[BaseAlgorithm]:
        hyperparams, saved_hyperparams = self.read_hyperparameters()
        hyperparams, self.env_wrapper, self.callbacks = self._preprocess_hyperparams(hyperparams)

        self.create_log_folder()
        self.create_callbacks()

        # Create env to have access to action space for action noise
        env = self.create_envs(self.n_envs, no_log=False)

        hyperparams = self._preprocess_action_noise(hyperparams, env)

        if self.continue_training:
            model = self._load_pretrained_agent(hyperparams, env)
        elif self.optimize_hyperparameters:
            self._hyperparameters_optimization(hyperparams)
            return None
        else:
            # Train an agent from scratch
            model = ALGOS[self.algo](
                env=env,
                tensorboard_log=self.tensorboard_log,
                seed=self.seed,
                verbose=self.verbose,
                **hyperparams,
            )

        self._save_config(saved_hyperparams)
        return model

    def learn(self, model: BaseAlgorithm) -> None:
        kwargs = {}
        if self.log_interval > -1:
            kwargs = {"log_interval": self.log_interval}

        if len(self.callbacks) > 0:
            kwargs["callback"] = self.callbacks

        try:
            model.learn(self.n_timesteps, **kwargs)
        except KeyboardInterrupt:
            pass
        finally:
            # Release resources
            model.env.close()

    def save_trained_model(self, model: BaseAlgorithm):
        # Save trained model
        print(f"Saving to {self.save_path}")
        model.save(f"{self.save_path}/{self.env_id}")

        if hasattr(model, "save_replay_buffer") and self.save_replay_buffer:
            print("Saving replay buffer")
            model.save_replay_buffer(os.path.join(self.save_path, "replay_buffer.pkl"))

        if self.normalize:
            # Important: save the running average, for testing the agent we need that normalization
            model.get_vec_normalize_env().save(os.path.join(self.params_path, "vecnormalize.pkl"))

    def _save_config(self, saved_hyperparams: Dict[str, Any]):
        # Save hyperparams
        with open(os.path.join(self.params_path, "config.yml"), "w") as f:
            yaml.dump(saved_hyperparams, f)

        # save command line arguments
        with open(os.path.join(self.params_path, "args.yml"), "w") as f:
            ordered_args = OrderedDict([(key, vars(self.args)[key]) for key in sorted(vars(self.args).keys())])
            yaml.dump(ordered_args, f)

        print(f"Log path: {self.save_path}")

    def _load_pretrained_agent(self, hyperparams: Dict[str, Any], env: VecEnv) -> BaseAlgorithm:
        # Continue training
        print("Loading pretrained agent")
        # Policy should not be changed
        del hyperparams["policy"]

        if "policy_kwargs" in hyperparams.keys():
            del hyperparams["policy_kwargs"]

        model = ALGOS[self.algo].load(
            self.trained_agent,
            env=env,
            seed=self.seed,
            tensorboard_log=self.tensorboard_log,
            verbose=self.verbose,
            **hyperparams,
        )

        replay_buffer_path = os.path.join(os.path.dirname(self.trained_agent), "replay_buffer.pkl")

        if os.path.exists(replay_buffer_path):
            print("Loading replay buffer")
            if self.algo == "her":
                # if we use HER we have to add an additional argument
                model.load_replay_buffer(replay_buffer_path, self.truncate_last_trajectory)
            else:
                model.load_replay_buffer(replay_buffer_path)
        return model

    def _hyperparameters_optimization(self, hyperparams: Dict[str, Any]):
        if self.verbose > 0:
            print("Optimizing hyperparameters")

        if self.storage is not None and self.study_name is None:
            warnings.warn(
                f"You passed a remote storage: {self.storage} but no `--study-name`."
                "The study name will be generated by Optuna, make sure to re-use the same study name "
                "when you want to do distributed hyperparameter optimization."
            )

        if self.tensorboard_log is not None:
            warnings.warn("Tensorboard log is deactivated when running hyperparameter optimization")
            self.tensorboard_log = None

        def create_model(*_args, **kwargs) -> BaseAlgorithm:
            """
            Helper to create a model with different hyperparameters
            """
            return ALGOS[self.algo](env=self.create_envs(self.n_envs, no_log=True), tensorboard_log=None, verbose=0, **kwargs)

        data_frame = hyperparam_optimization(
            self.algo,
            create_model,
            self.create_envs,
            n_trials=self.n_trials,
            n_timesteps=self.n_timesteps,
            hyperparams=hyperparams,
            n_jobs=self.n_jobs,
            seed=self.seed,
            sampler_method=self.sampler,
            pruner_method=self.pruner,
            n_startup_trials=self.n_startup_trials,
            n_evaluations=self.n_evaluations,
            storage=self.storage,
            study_name=self.study_name,
            verbose=self.verbose,
            deterministic_eval=not self.is_atari,
        )

        report_name = (
            f"report_{self.env_id}_{self.n_trials}-trials-{self.n_timesteps}"
            f"-{self.sampler}-{self.pruner}_{int(time.time())}.csv"
        )

        log_path = os.path.join(self.log_folder, self.algo, report_name)

        if self.verbose:
            print(f"Writing report to {log_path}")

        # Write report
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        data_frame.to_csv(log_path)

    def read_hyperparameters(self) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        # Load hyperparameters from yaml file
        with open(f"hyperparams/{self.algo}.yml", "r") as f:
            hyperparams_dict = yaml.safe_load(f)
            if self.env_id in list(hyperparams_dict.keys()):
                hyperparams = hyperparams_dict[self.env_id]
            elif self.is_atari:
                hyperparams = hyperparams_dict["atari"]
            else:
                raise ValueError(f"Hyperparameters not found for {self.algo}-{self.env_id}")

        if self.hyperparams is not None:
            # Overwrite hyperparams if needed
            hyperparams.update(self.hyperparams)
        # Sort hyperparams that will be saved
        saved_hyperparams = OrderedDict([(key, hyperparams[key]) for key in sorted(hyperparams.keys())])

        # model_class = self.algo
        # HER is only a wrapper around an algo
        if self.algo == "her":
            model_class = saved_hyperparams["model_class"]
            assert model_class in {"sac", "ddpg", "dqn", "td3", "tqc"}, f"{model_class} is not compatible with HER"
            # Retrieve the model class
            hyperparams["model_class"] = ALGOS[saved_hyperparams["model_class"]]

        if self.verbose > 0:
            pprint(saved_hyperparams)

        return hyperparams, saved_hyperparams

    @staticmethod
    def _preprocess_schedules(hyperparams: Dict[str, Any]) -> Dict[str, Any]:
        # Create schedules
        for key in ["learning_rate", "clip_range", "clip_range_vf"]:
            if key not in hyperparams:
                continue
            if isinstance(hyperparams[key], str):
                schedule, initial_value = hyperparams[key].split("_")
                initial_value = float(initial_value)
                hyperparams[key] = linear_schedule(initial_value)
            elif isinstance(hyperparams[key], (float, int)):
                # Negative value: ignore (ex: for clipping)
                if hyperparams[key] < 0:
                    continue
                hyperparams[key] = constant_fn(float(hyperparams[key]))
            else:
                raise ValueError(f"Invalid value for {key}: {hyperparams[key]}")
        return hyperparams

    def _preprocess_action_noise(self, hyperparams: Dict[str, Any], env: VecEnv) -> Dict[str, Any]:
        # Special case for HER
        algo = hyperparams["model_class"] if self.algo == "her" else self.algo
        # Parse noise string for DDPG and SAC
        if algo in ["ddpg", "sac", "td3", "tqc", "ddpg"] and hyperparams.get("noise_type") is not None:
            noise_type = hyperparams["noise_type"].strip()
            noise_std = hyperparams["noise_std"]

            n_actions = env.action_space.shape[0]

            if "normal" in noise_type:
                if "lin" in noise_type:
                    final_sigma = hyperparams.get("noise_std_final", 0.0) * np.ones(n_actions)
                    hyperparams["action_noise"] = LinearNormalActionNoise(
                        mean=np.zeros(n_actions),
                        sigma=noise_std * np.ones(n_actions),
                        final_sigma=final_sigma,
                        max_steps=self.n_timesteps,
                    )
                else:
                    hyperparams["action_noise"] = NormalActionNoise(
                        mean=np.zeros(n_actions), sigma=noise_std * np.ones(n_actions)
                    )
            elif "ornstein-uhlenbeck" in noise_type:
                hyperparams["action_noise"] = OrnsteinUhlenbeckActionNoise(
                    mean=np.zeros(n_actions), sigma=noise_std * np.ones(n_actions)
                )
            else:
                raise RuntimeError(f'Unknown noise type "{noise_type}"')
            print(f"Applying {noise_type} noise with std {noise_std}")

            del hyperparams["noise_type"]
            del hyperparams["noise_std"]
            if "noise_std_final" in hyperparams:
                del hyperparams["noise_std_final"]

        return hyperparams

    def _preprocess_hyperparams(
        self, hyperparams: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], Optional[Callable], List[BaseCallback]]:
        self.n_envs = hyperparams.get("n_envs", 1)

        if self.verbose > 0:
            print(f"Using {self.n_envs} environments")

        hyperparams = self._preprocess_schedules(hyperparams)

        # Should we overwrite the number of timesteps?
        if self.n_timesteps > 0:
            if self.verbose:
                print(f"Overwriting n_timesteps with n={self.n_timesteps}")
        else:
            self.n_timesteps = int(hyperparams["n_timesteps"])

        # Pre-process normalize config
        if "normalize" in hyperparams.keys():
            self.normalize = hyperparams["normalize"]

            if isinstance(self.normalize, str):
                self.normalize_kwargs = eval(self.normalize)
                self.normalize = True

            if "gamma" in hyperparams:
                self.normalize_kwargs["gamma"] = hyperparams["gamma"]

            del hyperparams["normalize"]

        # Pre-process policy keyword arguments
        if "policy_kwargs" in hyperparams.keys():
            # Convert to python object if needed
            if isinstance(hyperparams["policy_kwargs"], str):
                hyperparams["policy_kwargs"] = eval(hyperparams["policy_kwargs"])

        # Delete keys so the dict can be pass to the model constructor
        if "n_envs" in hyperparams.keys():
            del hyperparams["n_envs"]
        del hyperparams["n_timesteps"]

        if "frame_stack" in hyperparams.keys():
            self.frame_stack = hyperparams["frame_stack"]
            del hyperparams["frame_stack"]

        # obtain a class object from a wrapper name string in hyperparams
        # and delete the entry
        env_wrapper = get_wrapper_class(hyperparams)
        if "env_wrapper" in hyperparams.keys():
            del hyperparams["env_wrapper"]

        callbacks = get_callback_class(hyperparams)
        if "callback" in hyperparams.keys():
            del hyperparams["callback"]

        return hyperparams, env_wrapper, callbacks

    def create_log_folder(self):
        os.makedirs(self.params_path, exist_ok=True)

    def create_callbacks(self):

        if self.save_freq > 0:
            # Account for the number of parallel environments
            self.save_freq = max(self.save_freq // self.n_envs, 1)
            self.callbacks.append(
                CheckpointCallback(
                    save_freq=self.save_freq,
                    save_path=self.save_path,
                    name_prefix="rl_model",
                    verbose=1,
                )
            )

        # Create test env if needed, do not normalize reward
        if self.eval_freq > 0 and not self.optimize_hyperparameters:
            # Account for the number of parallel environments
            self.eval_freq = max(self.eval_freq // self.n_envs, 1)

            # if "NeckEnv" in env_id:
            #     # Use the training env as eval env when using the neck
            #     # because there is only one robot
            #     # there will be an issue with the reset
            #     eval_callback = EvalCallback(
            #         env,
            #         callback_on_new_best=None,
            #         best_model_save_path=save_path,
            #         log_path=save_path,
            #         eval_freq=args.eval_freq,
            #     )
            #     callbacks.append(eval_callback)
            if self.verbose > 0:
                print("Creating test environment")

            save_vec_normalize = SaveVecNormalizeCallback(save_freq=1, save_path=self.params_path)
            eval_callback = EvalCallback(
                self.create_envs(1, eval_env=True),
                callback_on_new_best=save_vec_normalize,
                best_model_save_path=self.save_path,
                n_eval_episodes=self.n_eval_episodes,
                log_path=self.save_path,
                eval_freq=self.eval_freq,
                deterministic=not self.is_atari,
            )

            self.callbacks.append(eval_callback)

    def create_envs(self, n_envs: int, eval_env: bool = False, no_log: bool = False) -> VecEnv:
        """
        Create the environment and wrap it if necessary.

        :param n_envs:
        :param eval_env: Whether is it an environment used for evaluation or not
        :param no_log: Do not log training when doing hyperparameter optim
            (issue with writing the same file)
        :return:
        """
        # Do not log eval env (issue with writing the same file)
        log_dir = None if eval_env or no_log else self.save_path

        # env = SubprocVecEnv([make_env(env_id, i, self.seed) for i in range(n_envs)])
        # On most env, SubprocVecEnv does not help and is quite memory hungry
        env = DummyVecEnv(
            [
                make_env(
                    self.env_id,
                    i,
                    self.seed,
                    log_dir=log_dir,
                    env_kwargs=self.env_kwargs,
                    wrapper_class=self.env_wrapper,
                )
                for i in range(n_envs)
            ]
        )

        # Pretrained model, load normalization
        path_ = os.path.join(os.path.dirname(self.trained_agent), self.env_id)
        path_ = os.path.join(path_, "vecnormalize.pkl")

        if os.path.exists(path_):
            print("Loading saved VecNormalize stats")
            env = VecNormalize.load(path_, env)
            # Deactivate training and reward normalization
            if eval_env:
                env.training = False
                env.norm_reward = False

        elif self.normalize:
            # Copy to avoid changing default values by reference
            local_normalize_kwargs = self.normalize_kwargs.copy()
            # Do not normalize reward for env used for evaluation
            if eval_env:
                if len(local_normalize_kwargs) > 0:
                    local_normalize_kwargs["norm_reward"] = False
                else:
                    local_normalize_kwargs = {"norm_reward": False}

            if self.verbose > 0:
                if len(local_normalize_kwargs) > 0:
                    print(f"Normalization activated: {local_normalize_kwargs}")
                else:
                    print("Normalizing input and reward")
            env = VecNormalize(env, **local_normalize_kwargs)

        # Optional Frame-stacking
        if self.frame_stack is not None:
            n_stack = self.frame_stack
            env = VecFrameStack(env, n_stack)
            if self.verbose > 0:
                print(f"Stacking {n_stack} frames")

        if is_image_space(env.observation_space):
            if self.verbose > 0:
                print("Wrapping into a VecTransposeImage")
            env = VecTransposeImage(env)

        # check if wrapper for dict support is needed
        if self.algo == "her":
            if self.verbose > 0:
                print("Wrapping into a ObsDictWrapper")
            env = ObsDictWrapper(env)

        return env
