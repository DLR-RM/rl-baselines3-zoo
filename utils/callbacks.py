import os

import numpy as np
import matplotlib.pyplot as plt

from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy


class TrialEvalCallback(EvalCallback):
    """
    Callback used for evaluating and reporting a trial.
    """
    def __init__(self, eval_env, trial, n_eval_episodes=5,
                 eval_freq=10000, deterministic=True, verbose=0):

        super(TrialEvalCallback, self).__init__(eval_env=eval_env, n_eval_episodes=n_eval_episodes,
                                                eval_freq=eval_freq,
                                                deterministic=deterministic,
                                                verbose=verbose)
        self.trial = trial
        self.eval_idx = 0
        self.is_pruned = False

    def _on_step(self):
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            super(TrialEvalCallback, self)._on_step()
            self.eval_idx += 1
            # report best or report current ?
            # report num_timesteps or elasped time ?
            self.trial.report(-1 * self.last_mean_reward, self.eval_idx)
            # Prune trial if need
            if self.trial.should_prune(self.eval_idx):
                self.is_pruned = True
                return False
        return True


class SaveVecNormalizeCallback(BaseCallback):
    """
    Callback for saving a VecNormalize wrapper every ``save_freq`` steps

    :param save_freq: (int)
    :param save_path: (str) Path to the folder where ``VecNormalize`` will be saved, as ``vecnormalize.pkl``
    :param name_prefix: (str) Common prefix to the saved ``VecNormalize``, if None (default)
        only one file will be kept.
    """
    def __init__(self, save_freq: int, save_path: str, name_prefix=None, verbose=0):
        super(SaveVecNormalizeCallback, self).__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        self.name_prefix = name_prefix

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            if self.name_prefix is not None:
                path = os.path.join(self.save_path, f"{self.name_prefix}_{self.num_timesteps}_steps.pkl")
            else:
                path = os.path.join(self.save_path, 'vecnormalize.pkl')
            if self.model.get_vec_normalize_env() is not None:
                self.model.get_vec_normalize_env().save(path)
                if self.verbose > 1:
                    print(f"Saving VecNormalize to {path}")
        return True


class PlotNoiseRatioCallback(BaseCallback):
    """
    Callback for plotting noise contribution to the exploration.
    Warning: it only works with 1D action space env for now (like MountainCarContinuous)

    :param display_freq: (int) Display the plot every ``display_freq`` steps.
    :param verbose: (int)
    """
    def __init__(self, display_freq=1000, verbose=0):
        super(PlotNoiseRatioCallback, self).__init__(verbose)
        self.display_freq = display_freq
        # Action buffers
        self.deterministic_actions = []
        self.noisy_actions = []
        self.noises = []

    def _on_step(self) -> bool:
        # We assume this is a DummyVecEnv
        assert isinstance(self.training_env, DummyVecEnv)
        # Retrieve last observation
        obs = self.training_env._obs_from_buf()
        # Retrieve stochastic and deterministic action
        # we can extract the noise contribution from those two
        noisy_action = self.model.predict(obs, deterministic=False)[0].flatten()
        deterministic_action = self.model.predict(obs, deterministic=True)[0].flatten()
        noise = noisy_action - deterministic_action

        self.deterministic_actions.append(deterministic_action)
        self.noisy_actions.append(noisy_action)
        self.noises.append(noise)

        if self.n_calls % self.display_freq == 0:
            x = np.arange(len(self.noisy_actions))

            self.deterministic_actions = np.array(self.deterministic_actions)
            self.noises = np.array(self.noises)

            plt.figure('Deterministic action and noise during exploration', figsize=(6.4, 4.8))
            # plt.title('Deterministic action and noise during exploration', fontsize=14)
            plt.xlabel('Timesteps', fontsize=14)
            plt.xticks(fontsize=13)
            plt.ylabel('Action', fontsize=14)
            plt.plot(x, self.deterministic_actions, label='deterministic action', linewidth=2)
            plt.plot(x, self.noises, label='exploration noise', linewidth=2)
            plt.plot(x, self.noisy_actions, label='noisy action', linewidth=2)
            plt.legend(fontsize=13)
            plt.show()
            # Reset
            self.noisy_actions = []
            self.deterministic_actions = []
            self.noises = []
        return True


class DQNEvalCallback(EvalCallback):
    """
    Callback for evaluating an agent.

    :param eval_env: (Union[gym.Env, VecEnv]) The environment used for initialization
    :param callback_on_new_best: (Optional[BaseCallback]) Callback to trigger
        when there is a new best model according to the ``mean_reward``
    :param n_eval_episodes: (int) The number of episodes to test the agent
    :param eval_freq: (int) Evaluate the agent every eval_freq call of the callback.
    :param log_path: (str) Path to a folder where the evaluations (``evaluations.npz``)
        will be saved. It will be updated at each evaluation.
    :param best_model_save_path: (str) Path to a folder where the best model
        according to performance on the eval env will be saved.
    :param deterministic: (bool) Whether the evaluation should
        use a stochastic or deterministic actions.
    :param deterministic: (bool) Whether to render or not the environment during evaluation
    :param render: (bool) Whether to render or not the environment during evaluation
    :param verbose: (int)
    """
    def __init__(self, eval_env,
                 callback_on_new_best=None,
                 n_eval_episodes: int = 5,
                 eval_epsilon: float = 0.05,
                 eval_freq: int = 10000,
                 log_path: str = None,
                 best_model_save_path: str = None,
                 deterministic: bool = True,
                 render: bool = False,
                 verbose: int = 1):
        super(DQNEvalCallback, self).__init__(eval_env=eval_env, n_eval_episodes=n_eval_episodes,
                                              eval_freq=eval_freq,
                                              deterministic=deterministic,
                                              verbose=verbose)
        self.eval_epsilon = eval_epsilon

    def _on_step(self) -> bool:

        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            # Sync training and eval env if there is VecNormalize
            # sync_envs_normalization(self.training_env, self.eval_env)

            old_epsilon = self.model.policy.epsilon
            self.model.policy.update_exploration_rate(self.eval_epsilon)

            episode_rewards, episode_lengths = evaluate_policy(self.model, self.eval_env,
                                                               n_eval_episodes=self.n_eval_episodes,
                                                               render=self.render,
                                                               deterministic=self.deterministic,
                                                               return_episode_rewards=True)

            self.model.policy.update_exploration_rate(old_epsilon)

            if self.log_path is not None:
                self.evaluations_timesteps.append(self.num_timesteps)
                self.evaluations_results.append(episode_rewards)
                self.evaluations_length.append(episode_lengths)
                np.savez(self.log_path, timesteps=self.evaluations_timesteps,
                         results=self.evaluations_results, ep_lengths=self.evaluations_length)

            mean_reward, std_reward = np.mean(episode_rewards), np.std(episode_rewards)
            mean_ep_length, std_ep_length = np.mean(episode_lengths), np.std(episode_lengths)
            self.last_mean_reward = mean_reward

            if self.verbose > 0:
                print(f"Eval num_timesteps={self.num_timesteps}, "
                      f"episode_reward={mean_reward:.2f} +/- {std_reward:.2f}")
                print(f"Episode length: {mean_ep_length:.2f} +/- {std_ep_length:.2f}")
            # Add to current Logger
            self.logger.record('eval/mean_reward', float(mean_reward))
            self.logger.record('eval/mean_ep_length', mean_ep_length)

            if mean_reward > self.best_mean_reward:
                if self.verbose > 0:
                    print("New best mean reward!")
                if self.best_model_save_path is not None:
                    self.model.save(os.path.join(self.best_model_save_path, 'best_model'))
                self.best_mean_reward = mean_reward
                # Trigger callback if needed
                if self.callback is not None:
                    return self._on_event()

        return True
