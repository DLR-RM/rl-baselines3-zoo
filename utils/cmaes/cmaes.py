import os
import time
from copy import deepcopy
from typing import Optional, Type, Union

import numpy as np
import torch as th
from sb3_contrib import TQC
from sb3_contrib.tqc.policies import TQCPolicy
from stable_baselines3.common import logger
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback
from stable_baselines3.common.utils import safe_mean

from cmaes import CMA, SepCMA


class CMAES(BaseAlgorithm):
    def __init__(
        self,
        policy: Type[BasePolicy],
        env: Union[GymEnv, str],
        n_individuals: int = -1,
        std_init: float = 0.1,
        model_path: str = os.environ.get("MODEL_PATH"),
        diagonal_cov: bool = False,
        pop_size: Optional[int] = None,
        tensorboard_log: Optional[str] = None,
        verbose: int = 0,
        device: Union[th.device, str] = "auto",
        create_eval_env: bool = False,
        monitor_wrapper: bool = True,
        seed: Optional[int] = None,
        _init_setup_model: bool = True,
    ):

        super(CMAES, self).__init__(
            policy=policy,
            env=env,
            policy_base=TQCPolicy,
            learning_rate=0.0,
            policy_kwargs={},
            tensorboard_log=tensorboard_log,
            verbose=verbose,
            device=device,
            support_multi_env=True,
            create_eval_env=create_eval_env,
            monitor_wrapper=monitor_wrapper,
            seed=seed,
            use_sde=False,
        )

        # Pretrained model
        assert model_path is not None
        print(f"Loading model from {model_path}")
        if model_path.endswith(".pt"):
            self.policy = th.jit.load(model_path)
            self._actor = th.jit.load(model_path)
            self.d3rlpy_model = True
        else:
            with th.no_grad():
                self.model = TQC.load(model_path)
                self.policy = self.model.actor
            self.d3rlpy_model = False
            self._actor = deepcopy(self.policy)

        self.model_path = model_path
        self.start_individual = self._params_to_vector(self.policy)
        self.best_individual = self.start_individual.copy()
        self.best_ever = None
        self.std_init = std_init
        self.optimizer = None
        self.diagonal_cov = diagonal_cov
        self.pop_size = pop_size

        if self.d3rlpy_model:
            self.weight_shape = self.policy.code_with_constants[1].c4.shape
            self.bias_shape = self.policy.code_with_constants[1].c5.shape
        else:
            self.weight_shape = self.policy.mu[0].weight.shape
            self.bias_shape = self.policy.mu[0].bias.shape

        self.weight_size = np.prod(self.weight_shape)
        self.bias_size = self.bias_shape[0]
        self.generation = 0

        if _init_setup_model:
            self._setup_model()

    def predict(self, obs, state=None, deterministic=True):
        # TODO: move to gpu when possible
        if self.d3rlpy_model:
            action = self.policy(th.tensor(obs).reshape(1, -1)).cpu().numpy()
        else:
            action, _ = self.policy.predict(obs, deterministic=True)

        return action, None

    @staticmethod
    def to_numpy(tensor):
        return tensor.detach().numpy().flatten()

    def _params_to_vector(self, actor) -> np.ndarray:
        if self.d3rlpy_model:
            weight = actor.code_with_constants[1].c4
            bias = actor.code_with_constants[1].c5
        else:
            weight = actor.mu[0].weight
            bias = actor.mu[0].bias
        return np.concatenate([self.to_numpy(weight), self.to_numpy(bias)])

    def _vector_to_params(self, actor, candidate):
        weight = th.tensor(candidate[: self.weight_size].reshape(self.weight_shape)).to(self.device)
        bias = th.tensor(candidate[-self.bias_size :].reshape(self.bias_shape)).to(self.device)
        if self.d3rlpy_model:
            # Hack to change params
            actor.forward.code_with_constants[1]["c4"] *= 0
            actor.forward.code_with_constants[1]["c4"] += weight
            actor.forward.code_with_constants[1]["c5"] *= 0
            actor.forward.code_with_constants[1]["c5"] += bias
        else:
            params = {
                "mu.0.weight": weight,
                "mu.0.bias": bias,
            }
            actor.load_state_dict(params, strict=False)

    def _setup_model(self) -> None:
        self.set_random_seed(self.seed)
        self._vector_to_params(self.policy, self.best_individual.copy())

    def learn(
        self,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 10,
        eval_env: Optional[GymEnv] = None,
        eval_freq: int = -1,
        n_eval_episodes: int = 5,
        tb_log_name: str = "run",
        eval_log_path: Optional[str] = None,
        reset_num_timesteps: bool = True,
    ) -> "CMAES":

        total_timesteps, callback = self._setup_learn(
            total_timesteps, eval_env, callback, eval_freq, n_eval_episodes, eval_log_path, reset_num_timesteps, tb_log_name
        )

        callback.on_training_start(locals(), globals())

        if self.best_individual is None:
            self.best_individual = self._params_to_vector(self.policy)

        if self.optimizer is None:
            options = {"seed": self.seed}
            # if self.env.num_envs > 1:
            #     options["population_size"] = self.env.num_envs

            if self.pop_size is not None:
                options["population_size"] = self.pop_size

            # bounds can be specified too
            # TODO: use pruning to reduce search space

            # TODO: try more advanced like BIPOP-CMA and Warm Starting CMA-ES
            if self.diagonal_cov:
                self.optimizer = SepCMA(self.best_individual, self.std_init, **options)
            else:
                self.optimizer = CMA(self.best_individual, self.std_init, **options)

        continue_training = True
        actor = self._actor

        if self.verbose > 0:
            print(f"{self.optimizer.population_size} candidates")

        while self.num_timesteps < total_timesteps and not self.optimizer.should_stop() and continue_training:
            candidates = [self.optimizer.ask() for _ in range(self.optimizer.population_size)]
            self.generation += 1
            print()
            print(f"=== Generation {self.generation} ====")

            # Add best (start individual)
            if self.best_ever is None:
                candidates[0] = self.best_individual.copy()

            returns = np.zeros((len(candidates),))
            candidate_idx = 0
            candidate_steps = 0

            self._vector_to_params(actor, candidates[candidate_idx])

            callback.on_rollout_start()

            while candidate_idx < len(candidates):
                # TODO support num_envs > 0
                if self.d3rlpy_model:
                    action = actor(th.tensor(self._last_obs).reshape(1, -1)).cpu().numpy()
                else:
                    action, _ = actor.predict(self._last_obs, deterministic=True)

                # Rescale and perform action
                new_obs, reward, done, infos = self.env.step(action)

                # Only stop training if return value is False, not when it is None.
                if callback.on_step() is False:
                    continue_training = False
                    break

                returns[candidate_idx] += reward

                # Retrieve reward and episode length if using Monitor wrapper
                self._update_info_buffer(infos, done)

                self._last_obs = new_obs

                self.num_timesteps += 1
                candidate_steps += 1
                self._update_current_progress_remaining(self.num_timesteps, self._total_timesteps)

                if done:
                    self._episode_num += 1

                    # Log training infos
                    if log_interval is not None and self._episode_num % log_interval == 0:
                        self._dump_logs()

                if done:
                    if self.verbose > 0:
                        print(f"Candidate {candidate_idx + 1}, return={returns[candidate_idx]:.2f}")

                    if self.best_ever is None or returns[candidate_idx] > self.best_ever:
                        print("New Best!")
                        self.best_ever = returns[candidate_idx]
                        self.best_individual = candidates[candidate_idx].copy()
                        self._vector_to_params(self.policy, candidates[candidate_idx].copy())
                    # force reset
                    # self._last_obs = self.env.reset()
                    candidate_idx += 1
                    candidate_steps = 0
                    if candidate_idx < len(candidates):
                        self._vector_to_params(actor, candidates[candidate_idx])
                    else:
                        break

            callback.on_rollout_end()

            results = [(candidate, -return_) for candidate, return_ in zip(candidates, returns)]
            self.optimizer.tell(results)
            if self.verbose > 0:
                print(f"Mean return={np.mean(returns):.2f} +/- {np.std(returns):.2f}")

        callback.on_training_end()

        return self

    def _dump_logs(self) -> None:
        """
        Write log.
        """
        fps = int(self.num_timesteps / (time.time() - self.start_time))
        logger.record("time/episodes", self._episode_num, exclude="tensorboard")
        if len(self.ep_info_buffer) > 0 and len(self.ep_info_buffer[0]) > 0:
            logger.record("rollout/ep_rew_mean", safe_mean([ep_info["r"] for ep_info in self.ep_info_buffer]))
            logger.record("rollout/ep_len_mean", safe_mean([ep_info["l"] for ep_info in self.ep_info_buffer]))
        logger.record("time/fps", fps)
        logger.record("time/time_elapsed", int(time.time() - self.start_time), exclude="tensorboard")
        logger.record("time/total timesteps", self.num_timesteps, exclude="tensorboard")
        logger.record("rollout/best_ever", self.best_ever)

        if len(self.ep_success_buffer) > 0:
            logger.record("rollout/success rate", safe_mean(self.ep_success_buffer))
        # Pass the number of timesteps for tensorboard
        logger.dump(step=self.num_timesteps)

    def _get_torch_save_params(self):
        """
        Get the name of the torch variables that will be saved.
        ``th.save`` and ``th.load`` will be used with the right device
        instead of the default pickling strategy.

        :return: (Tuple[List[str], List[str]])
            name of the variables with state dicts to save, name of additional torch tensors,
        """
        return [], []

    def _excluded_save_params(self):
        """
        Returns the names of the parameters that should be excluded by default
        when saving the model.

        :return: (List[str]) List of parameters that should be excluded from save
        """
        # Exclude aliases
        exclude = ["policy", "_actor"] if self.d3rlpy_model else []
        return super()._excluded_save_params() + ["model"] + exclude
