import os
from typing import Dict, List, Optional, Tuple, Type

import gym
import torch as th
from sb3_contrib import TQC
from sb3_contrib.tqc.policies import TQCPolicy
from stable_baselines3.common.policies import BasePolicy, register_policy
from stable_baselines3.common.preprocessing import get_action_dim
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, create_mlp
from torch import nn

# CAP the standard deviation of the actor
LOG_STD_MAX = 2
LOG_STD_MIN = -20


class MixtureActor(BasePolicy):
    """
    Actor network (policy) for SAC.

    :param observation_space: Obervation space
    :param action_space: Action space
    :param net_arch: Network architecture
    :param features_extractor: Network to extract features
        (a CNN when using images, a nn.Flatten() layer otherwise)
    :param features_dim: Number of features
    :param activation_fn: Activation function
    :param use_sde: Whether to use State Dependent Exploration or not
    :param log_std_init: Initial value for the log standard deviation
    :param full_std: Whether to use (n_features x n_actions) parameters
        for the std instead of only (n_features,) when using gSDE.
    :param sde_net_arch: Network architecture for extracting features
        when using gSDE. If None, the latent features from the policy will be used.
        Pass an empty list to use the states as features.
    :param use_expln: Use ``expln()`` function instead of ``exp()`` when using gSDE to ensure
        a positive standard deviation (cf paper). It allows to keep variance
        above zero and prevent it from growing too fast. In practice, ``exp()`` is usually enough.
    :param clip_mean: Clip the mean output when using gSDE to avoid numerical instability.
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    """

    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        net_arch: List[int],
        features_extractor: nn.Module,
        features_dim: int,
        activation_fn: Type[nn.Module] = nn.ReLU,
        normalize_images: bool = True,
        n_additional_experts: int = 0,
        # ignore
        *_args,
        **_kwargs,
    ):
        super().__init__(
            observation_space,
            action_space,
            features_extractor=features_extractor,
            normalize_images=normalize_images,
            squash_output=True,
        )

        # Pretrained model
        # set BACKWARD_CONTROLLER_PATH=logs\pretrained-tqc\SE-Symmetric-v1_2\SE-Symmetric-v1.zip
        # set FORWARD_CONTROLLER_PATH=logs\pretrained-tqc\SE-Symmetric-v1_1\SE-Symmetric-v1.zip
        # set TURN_LEFT_CONTROLLER_PATH=logs\pretrained-tqc\SE-TurnLeft-v1_1\SE-TurnLeft-v1.zip
        # set TURN_RIGHT_CONTROLLER_PATH=logs\pretrained-tqc\SE-TurnLeft-v1_2\SE-TurnLeft-v1.zip
        # set RANDOM_CONTROLLER_PATH=logs\pretrained-tqc\SE-Random-small\SE-TurnLeft-v1.zip

        # expert_paths = ["FORWARD", "BACKWARD", "TURN_LEFT", "TURN_RIGHT"]
        expert_paths = []
        self.num_experts = len(expert_paths)
        self.n_additional_experts = n_additional_experts
        print(f"{n_additional_experts} additional experts")
        self.num_experts += self.n_additional_experts

        self.experts = []
        for path in expert_paths:
            actor = TQC.load(os.environ[f"{path}_CONTROLLER_PATH"]).actor
            self.experts.append(actor)

        # Add additional experts
        for _ in range(self.n_additional_experts):
            actor = TQC.load(os.environ["RANDOM_CONTROLLER_PATH"]).actor
            self.experts.append(actor)

        features_dim = self.experts[0].features_dim
        self.experts = nn.ModuleList(self.experts)
        # TODO: replace with MLP?
        # self.w_gate = nn.Parameter(th.zeros(features_dim, num_experts), requires_grad=True)
        gating_net_arch = [64, 64]
        gating_net = create_mlp(features_dim, self.num_experts, gating_net_arch, activation_fn)
        gating_net += [nn.Softmax(1)]
        self.gating_net = nn.Sequential(*gating_net)
        self.action_dim = get_action_dim(self.action_space)
        self.action_dist = self.experts[0].action_dist

    def get_action_dist_params(self, obs: th.Tensor) -> Tuple[th.Tensor, th.Tensor, Dict[str, th.Tensor]]:
        """
        Get the parameters for the action distribution.

        :param obs:
        :return:
            Mean, standard deviation and optional keyword arguments.
        """
        features = self.extract_features(obs)
        expert_means = th.zeros(obs.shape[0], self.num_experts, self.action_dim).to(obs.device)
        expert_stds = th.zeros(obs.shape[0], self.num_experts, self.action_dim).to(obs.device)

        for i in range(self.num_experts):
            # Allow grad for one expert only
            with th.set_grad_enabled(i >= self.num_experts - self.n_additional_experts):
                latent_pi = self.experts[i].latent_pi(features)
                expert_means[:, i, :] = self.experts[i].mu(latent_pi)
                # Unstructured exploration (Original implementation)
                log_std = self.experts[i].log_std(latent_pi)
                # Original Implementation to cap the standard deviation
                expert_stds[:, i, :] = th.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)

        # gates: [batch_size, num_experts]
        input_commands = features.clone()
        # TODO: extract task features only?
        # input_commands[:-2] = 0.0
        gates = self.gating_net(input_commands).unsqueeze(-1)

        # expert_means: [batch_size, num_experts, action_dim]
        # mean_actions: [batch_size, action_dim]
        mean_actions = (gates * expert_means).sum(dim=1)
        log_std = (gates * expert_stds).sum(dim=1)

        return mean_actions, log_std, {}

    def forward(self, obs: th.Tensor, deterministic: bool = False) -> th.Tensor:
        mean_actions, log_std, kwargs = self.get_action_dist_params(obs)
        # Note: the action is squashed
        return self.action_dist.actions_from_params(mean_actions, log_std, deterministic=deterministic, **kwargs)

    def action_log_prob(self, obs: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        mean_actions, log_std, kwargs = self.get_action_dist_params(obs)
        # return action and associated log prob
        return self.action_dist.log_prob_from_params(mean_actions, log_std, **kwargs)

    def _predict(self, observation: th.Tensor, deterministic: bool = False) -> th.Tensor:
        return self.forward(observation, deterministic)


class MixtureMlpPolicy(TQCPolicy):
    """
    Policy class (with both actor and critic) for TQC.

    :param observation_space: Observation space
    :param action_space: Action space
    :param lr_schedule: Learning rate schedule (could be constant)
    :param net_arch: The specification of the policy and value networks.
    :param activation_fn: Activation function
    :param use_sde: Whether to use State Dependent Exploration or not
    :param log_std_init: Initial value for the log standard deviation
    :param sde_net_arch: Network architecture for extracting features
        when using gSDE. If None, the latent features from the policy will be used.
        Pass an empty list to use the states as features.
    :param use_expln: Use ``expln()`` function instead of ``exp()`` when using gSDE to ensure
        a positive standard deviation (cf paper). It allows to keep variance
        above zero and prevent it from growing too fast. In practice, ``exp()`` is usually enough.
    :param clip_mean: Clip the mean output when using gSDE to avoid numerical instability.
    :param features_extractor_class: Features extractor to use.
    :param features_extractor_kwargs: Keyword arguments
        to pass to the features extractor.
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    :param optimizer_class: The optimizer to use,
        ``th.optim.Adam`` by default
    :param optimizer_kwargs: Additional keyword arguments,
        excluding the learning rate, to pass to the optimizer
    :param n_critics: Number of critic networks to create.
    :param share_features_extractor: Whether to share or not the features extractor
        between the actor and the critic (this saves computation time)
    """

    def __init__(self, *args, n_additional_experts: int = 0, **kwargs):
        self.n_additional_experts = n_additional_experts
        super(MixtureMlpPolicy, self).__init__(
            *args,
            **kwargs,
        )

    def make_actor(self, features_extractor: Optional[BaseFeaturesExtractor] = None) -> MixtureActor:
        actor_kwargs = self._update_features_extractor(self.actor_kwargs, features_extractor)
        return MixtureActor(n_additional_experts=self.n_additional_experts, **actor_kwargs).to(self.device)


register_policy("MixtureMlpPolicy", MixtureMlpPolicy)
