from typing import Dict, Tuple

import gym
import torch as th
from stable_baselines3.common.distributions import TanhBijector
from stable_baselines3.common.policies import register_policy
from stable_baselines3.common.preprocessing import get_action_dim
from stable_baselines3.tqc.policies import LOG_STD_MAX, LOG_STD_MIN, Actor, TQCPolicy
from torch import nn


class ResidualActor(Actor):
    def __init__(self, *args, **kwargs):
        super(ResidualActor, self).__init__(*args, **kwargs)
        self.action_dim = get_action_dim(self.action_space)
        # Before sigmoid, output close to 1
        init_scale_expert = 5.0
        # Before sigmoid, output close to 0 (here around 0.1)
        init_scale_rl = -2.0
        self.scale_expert = nn.Parameter(th.ones(self.action_dim) * init_scale_expert, requires_grad=True)
        self.scale_rl = nn.Parameter(th.ones(self.action_dim) * init_scale_rl, requires_grad=True)

    def get_action_dist_params(self, obs: th.Tensor) -> Tuple[th.Tensor, th.Tensor, Dict[str, th.Tensor]]:
        """
        Get the parameters for the action distribution.

        :param obs: (th.Tensor)
        :return: (Tuple[th.Tensor, th.Tensor, Dict[str, th.Tensor]])
            Mean, standard deviation and optional keyword arguments.
        """
        features = self.extract_features(obs)
        latent_pi = self.latent_pi(features)
        mean_actions = self.mu(latent_pi)

        # Assume that expert action is are the last features
        expert_actions = TanhBijector.inverse(obs[:, -self.action_dim :])
        # expert_actions = obs[:, -self.action_dim :]
        mean_actions_ = th.sigmoid(self.scale_rl) * mean_actions + th.sigmoid(self.scale_expert) * expert_actions

        if self.use_sde:
            latent_sde = latent_pi
            if self.sde_features_extractor is not None:
                latent_sde = self.sde_features_extractor(features)
            return mean_actions_, self.log_std, dict(latent_sde=latent_sde)
        # Unstructured exploration (Original implementation)
        log_std = self.log_std(latent_pi)
        # Original Implementation to cap the standard deviation
        log_std = th.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        return mean_actions_, log_std, {}


class MlpResidualPolicy(TQCPolicy):
    def __init__(self, *args, **kwargs):
        super(MlpResidualPolicy, self).__init__(*args, **kwargs)

    def make_actor(self) -> Actor:
        return ResidualActor(**self.actor_kwargs).to(self.device)


register_policy("MlpResidualPolicy", MlpResidualPolicy)
