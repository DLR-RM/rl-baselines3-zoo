import os
import pickle

from torchy_baselines.common.callbacks import BaseCallback, EvalCallback


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
    Callback for saving a VecNormalize wrapper every `save_freq` steps

    :param save_freq: (int)
    :param save_path: (str) Path to the folder where `VecNormalize` will be saved, as `vecnormalize.pkl`
    :param name_prefix: (str) Common prefix to the saved `VecNormalize`, if None (default)
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
                path = os.path.join(self.save_path, f'{self.name_prefix}_{self.num_timesteps}_steps.pkl')
            else:
                path = os.path.join(self.save_path, 'vecnormalize.pkl')
            if self.model.get_vec_normalize_env() is not None:
                self.model.get_vec_normalize_env().save(path)
                if self.verbose > 1:
                    print(f"Saving VecNormalize to {path}")
        return True
