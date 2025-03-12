from typing import Any

import numpy as np
import optuna
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from torch import nn as nn


def convert_onpolicy_params(sampled_params: dict[str, Any]) -> dict[str, Any]:
    hyperparams = sampled_params.copy()

    # TODO: account when using multiple envs
    # if batch_size > n_steps:
    #     batch_size = n_steps

    hyperparams["gamma"] = 1 - sampled_params["one_minus_gamma"]
    del hyperparams["one_minus_gamma"]

    hyperparams["gae_lambda"] = 1 - sampled_params["one_minus_gae_lambda"]
    del hyperparams["one_minus_gae_lambda"]

    net_arch = sampled_params["net_arch"]
    del hyperparams["net_arch"]

    for name in ["batch_size", "n_steps"]:
        if f"{name}_pow" in sampled_params:
            hyperparams[name] = 2 ** sampled_params[f"{name}_pow"]
            del hyperparams[f"{name}_pow"]

    net_arch = {
        "tiny": dict(pi=[64], vf=[64]),
        "small": dict(pi=[64, 64], vf=[64, 64]),
        "medium": dict(pi=[256, 256], vf=[256, 256]),
    }[net_arch]

    activation_fn_name = sampled_params["activation_fn"]
    del hyperparams["activation_fn"]

    activation_fn = {
        "tanh": nn.Tanh,
        "relu": nn.ReLU,
        "elu": nn.ELU,
        "leaky_relu": nn.LeakyReLU,
    }[activation_fn_name]

    return {
        "policy_kwargs": {
            "net_arch": net_arch,
            "activation_fn": activation_fn,
        },
        **hyperparams,
    }


def convert_offpolicy_params(sampled_params: dict[str, Any]) -> dict[str, Any]:
    hyperparams = sampled_params.copy()

    hyperparams["gamma"] = 1 - sampled_params["one_minus_gamma"]
    del hyperparams["one_minus_gamma"]

    net_arch = sampled_params["net_arch"]
    del hyperparams["net_arch"]

    for name in ["batch_size"]:
        if f"{name}_pow" in sampled_params:
            hyperparams[name] = 2 ** sampled_params[f"{name}_pow"]
            del hyperparams[f"{name}_pow"]

    net_arch = {
        "small": [64, 64],
        "medium": [256, 256],
        "big": [400, 300],
        "large": [256, 256, 256],
        "verybig": [512, 512, 512],
    }[net_arch]

    if "train_freq" in sampled_params:
        # Update to data ratio of 1, for n_envs=1
        hyperparams["gradient_steps"] = sampled_params["train_freq"]

        if "subsample_steps" in sampled_params:
            hyperparams["gradient_steps"] = max(sampled_params["train_freq"] // sampled_params["subsample_steps"], 1)
            del hyperparams["subsample_steps"]

    hyperparams["policy_kwargs"] = hyperparams.get("policy_kwargs", {})
    hyperparams["policy_kwargs"]["net_arch"] = net_arch

    if "activation_fn" in sampled_params:
        activation_fn_name = sampled_params["activation_fn"]
        del hyperparams["activation_fn"]

        activation_fn = {
            "tanh": nn.Tanh,
            "relu": nn.ReLU,
            "elu": nn.ELU,
            "leaky_relu": nn.LeakyReLU,
        }[activation_fn_name]
        hyperparams["policy_kwargs"]["activation_fn"] = activation_fn

    # TQC/QRDQN
    if "n_quantiles" in sampled_params:
        del hyperparams["n_quantiles"]
        hyperparams["policy_kwargs"].update({"n_quantiles": sampled_params["n_quantiles"]})

    return hyperparams


def sample_ppo_params(trial: optuna.Trial, n_actions: int, n_envs: int, additional_args: dict) -> dict[str, Any]:
    """
    Sampler for PPO hyperparams.

    :param trial:
    :return:
    """
    # From 2**5=32 to 2**10=1024
    batch_size_pow = trial.suggest_int("batch_size_pow", 2, 10)
    # From 2**5=32 to 2**12=4096
    n_steps_pow = trial.suggest_int("n_steps_pow", 5, 12)
    one_minus_gamma = trial.suggest_float("one_minus_gamma", 0.0001, 0.03, log=True)
    one_minus_gae_lambda = trial.suggest_float("one_minus_gae_lambda", 0.0001, 0.1, log=True)

    learning_rate = trial.suggest_float("learning_rate", 1e-5, 0.002, log=True)
    ent_coef = trial.suggest_float("ent_coef", 0.00000001, 0.1, log=True)
    clip_range = trial.suggest_categorical("clip_range", [0.1, 0.2, 0.3, 0.4])
    n_epochs = trial.suggest_categorical("n_epochs", [1, 5, 10, 20])

    max_grad_norm = trial.suggest_float("max_grad_norm", 0.3, 2)
    net_arch = trial.suggest_categorical("net_arch", ["tiny", "small", "medium"])
    activation_fn = trial.suggest_categorical("activation_fn", ["tanh", "relu"])
    # lr_schedule = "constant"
    # Uncomment to enable learning rate schedule
    # lr_schedule = trial.suggest_categorical('lr_schedule', ['linear', 'constant'])
    # if lr_schedule == "linear":
    #     learning_rate = linear_schedule(learning_rate)

    # Display true values
    trial.set_user_attr("gamma", 1 - one_minus_gamma)
    trial.set_user_attr("n_steps", 2**n_steps_pow)
    trial.set_user_attr("batch_size", 2**batch_size_pow)
    sampled_params = {
        "n_steps_pow": n_steps_pow,
        "batch_size_pow": batch_size_pow,
        "one_minus_gamma": one_minus_gamma,
        "one_minus_gae_lambda": one_minus_gae_lambda,
        "learning_rate": learning_rate,
        "ent_coef": ent_coef,
        "clip_range": clip_range,
        "n_epochs": n_epochs,
        "max_grad_norm": max_grad_norm,
        "net_arch": net_arch,
        "activation_fn": activation_fn,
    }

    return convert_onpolicy_params(sampled_params)


def sample_ppo_lstm_params(trial: optuna.Trial, n_actions: int, n_envs: int, additional_args: dict) -> dict[str, Any]:
    """
    Sampler for RecurrentPPO hyperparams.
    uses sample_ppo_params(), this function samples for the policy_kwargs
    :param trial:
    :return:
    """
    return sample_ppo_params(trial, n_actions, n_envs, additional_args)
    # enable_critic_lstm = trial.suggest_categorical("enable_critic_lstm", [False, True])
    # lstm_hidden_size = trial.suggest_categorical("lstm_hidden_size", [16, 32, 64, 128, 256, 512])

    # hyperparams["policy_kwargs"].update(
    #     {
    #         "enable_critic_lstm": enable_critic_lstm,
    #         "lstm_hidden_size": lstm_hidden_size,
    #     }
    # )

    # return hyperparams


def sample_trpo_params(trial: optuna.Trial, n_actions: int, n_envs: int, additional_args: dict) -> dict[str, Any]:
    """
    Sampler for TRPO hyperparams.

    :param trial:
    :return:
    """
    # From 2**5=32 to 2**10=1024
    batch_size_pow = trial.suggest_int("batch_size_pow", 2, 10)
    # From 2**5=32 to 2**12=4096
    n_steps_pow = trial.suggest_int("n_steps_pow", 5, 12)
    one_minus_gamma = trial.suggest_float("one_minus_gamma", 0.0001, 0.03, log=True)
    one_minus_gae_lambda = trial.suggest_float("one_minus_gae_lambda", 0.0001, 0.1, log=True)

    learning_rate = trial.suggest_float("learning_rate", 1e-5, 0.002, log=True)
    # line_search_shrinking_factor = trial.suggest_categorical("line_search_shrinking_factor", [0.6, 0.7, 0.8, 0.9])
    n_critic_updates = trial.suggest_int("n_critic_updates", 5, 30)
    cg_max_steps = trial.suggest_int("cg_max_steps", 5, 30)
    # cg_damping = trial.suggest_categorical("cg_damping", [0.5, 0.2, 0.1, 0.05, 0.01])
    target_kl = trial.suggest_float("target_kl", 0.001, 0.1, log=True)

    net_arch = trial.suggest_categorical("net_arch", ["small", "medium"])
    activation_fn = trial.suggest_categorical("activation_fn", ["tanh", "relu"])

    # Display true values
    trial.set_user_attr("gamma", 1 - one_minus_gamma)
    trial.set_user_attr("n_steps", 2**n_steps_pow)
    trial.set_user_attr("batch_size", 2**batch_size_pow)
    return convert_onpolicy_params(
        {
            "n_steps_pow": n_steps_pow,
            "batch_size_pow": batch_size_pow,
            "one_minus_gamma": one_minus_gamma,
            "one_minus_gae_lambda": one_minus_gae_lambda,
            "learning_rate": learning_rate,
            "cg_max_steps": cg_max_steps,
            "target_kl": target_kl,
            "net_arch": net_arch,
            "activation_fn": activation_fn,
            "n_critic_updates": n_critic_updates,
        }
    )


def sample_a2c_params(trial: optuna.Trial, n_actions: int, n_envs: int, additional_args: dict) -> dict[str, Any]:
    """
    Sampler for A2C hyperparams.

    :param trial:
    :return:
    """
    n_steps = trial.suggest_categorical("n_steps", [32, 64, 128, 256, 512, 1024, 2048])
    one_minus_gamma = trial.suggest_float("one_minus_gamma", 0.0001, 0.03, log=True)
    one_minus_gae_lambda = trial.suggest_float("one_minus_gae_lambda", 0.0001, 0.1, log=True)

    learning_rate = trial.suggest_float("learning_rate", 1e-5, 0.002, log=True)
    max_grad_norm = trial.suggest_float("max_grad_norm", 0.3, 2)

    ent_coef = trial.suggest_float("ent_coef", 0.00000001, 0.1, log=True)
    net_arch = trial.suggest_categorical("net_arch", ["small", "medium"])
    activation_fn = trial.suggest_categorical("activation_fn", ["tanh", "relu"])

    # Display true values
    trial.set_user_attr("gamma", 1 - one_minus_gamma)

    sampled_params = {
        "n_steps": n_steps,
        "one_minus_gamma": one_minus_gamma,
        "one_minus_gae_lambda": one_minus_gae_lambda,
        "learning_rate": learning_rate,
        "ent_coef": ent_coef,
        "max_grad_norm": max_grad_norm,
        "net_arch": net_arch,
        "activation_fn": activation_fn,
    }

    return convert_onpolicy_params(sampled_params)


def sample_sac_params(trial: optuna.Trial, n_actions: int, n_envs: int, additional_args: dict) -> dict[str, Any]:
    """
    Sampler for SAC hyperparams.

    :param trial:
    :return:
    """
    one_minus_gamma = trial.suggest_float("one_minus_gamma", 0.0001, 0.03, log=True)
    # From 2**5=32 to 2**11=2048
    batch_size_pow = trial.suggest_int("batch_size_pow", 2, 11)

    learning_rate = trial.suggest_float("learning_rate", 1e-5, 0.002, log=True)
    train_freq = trial.suggest_int("train_freq", 1, 10)
    # Polyak coeff
    tau = trial.suggest_float("tau", 0.001, 0.08, log=True)

    # NOTE: Add "verybig" to net_arch when tuning HER
    net_arch = trial.suggest_categorical("net_arch", ["small", "medium", "big"])
    # activation_fn = trial.suggest_categorical('activation_fn', [nn.Tanh, nn.ReLU, nn.ELU, nn.LeakyReLU])

    trial.set_user_attr("gamma", 1 - one_minus_gamma)
    trial.set_user_attr("batch_size", 2**batch_size_pow)

    hyperparams = {
        "one_minus_gamma": one_minus_gamma,
        "learning_rate": learning_rate,
        "batch_size_pow": batch_size_pow,
        "train_freq": train_freq,
        "tau": tau,
        "net_arch": net_arch,
    }

    if additional_args["using_her_replay_buffer"]:
        hyperparams = sample_her_params(trial, hyperparams, additional_args["her_kwargs"])

    if "sample_tqc" in additional_args:
        n_quantiles = trial.suggest_int("n_quantiles", 5, 50)
        top_quantiles_to_drop_per_net = trial.suggest_int("top_quantiles_to_drop_per_net", 0, min(n_quantiles - 1, 5))
        hyperparams.update(
            {
                "n_quantiles": n_quantiles,
                "top_quantiles_to_drop_per_net": top_quantiles_to_drop_per_net,
            }
        )

    return convert_offpolicy_params(hyperparams)


def sample_td3_params(trial: optuna.Trial, n_actions: int, n_envs: int, additional_args: dict) -> dict[str, Any]:
    """
    Sampler for TD3 hyperparams.

    :param trial:
    :return:
    """
    one_minus_gamma = trial.suggest_float("one_minus_gamma", 0.0001, 0.03, log=True)
    # From 2**5=32 to 2**11=2048
    batch_size_pow = trial.suggest_int("batch_size_pow", 2, 11)

    learning_rate = trial.suggest_float("learning_rate", 1e-5, 0.002, log=True)
    train_freq = trial.suggest_int("train_freq", 1, 10)
    # Polyak coeff
    tau = trial.suggest_float("tau", 0.001, 0.08, log=True)

    noise_type = trial.suggest_categorical("noise_type", ["ornstein-uhlenbeck", "normal", None])
    noise_std = trial.suggest_float("noise_std", 0, 0.5)

    # NOTE: Add "verybig" to net_arch when tuning HER
    net_arch = trial.suggest_categorical("net_arch", ["small", "medium", "big"])

    trial.set_user_attr("gamma", 1 - one_minus_gamma)
    trial.set_user_attr("batch_size", 2**batch_size_pow)

    hyperparams = {
        "one_minus_gamma": one_minus_gamma,
        "learning_rate": learning_rate,
        "batch_size_pow": batch_size_pow,
        "train_freq": train_freq,
        "tau": tau,
        "net_arch": net_arch,
    }

    if noise_type == "normal":
        hyperparams["action_noise"] = NormalActionNoise(mean=np.zeros(n_actions), sigma=noise_std * np.ones(n_actions))
    elif noise_type == "ornstein-uhlenbeck":
        hyperparams["action_noise"] = OrnsteinUhlenbeckActionNoise(
            mean=np.zeros(n_actions), sigma=noise_std * np.ones(n_actions)
        )

    if additional_args["using_her_replay_buffer"]:
        hyperparams = sample_her_params(trial, hyperparams, additional_args["her_kwargs"])

    return convert_offpolicy_params(hyperparams)


def sample_dqn_params(trial: optuna.Trial, n_actions: int, n_envs: int, additional_args: dict) -> dict[str, Any]:
    """
    Sampler for DQN hyperparams.

    :param trial:
    :return:
    """
    one_minus_gamma = trial.suggest_float("one_minus_gamma", 0.0001, 0.03, log=True)
    # From 2**5=32 to 2**11=2048
    batch_size_pow = trial.suggest_int("batch_size_pow", 2, 11)

    learning_rate = trial.suggest_float("learning_rate", 1e-5, 0.002, log=True)
    train_freq = trial.suggest_int("train_freq", 1, 10)
    subsample_steps = trial.suggest_int("subsample_steps", 1, min(train_freq, 8))

    exploration_final_eps = trial.suggest_float("exploration_final_eps", 0, 0.2)
    exploration_fraction = trial.suggest_float("exploration_fraction", 0, 0.5)
    target_update_interval = trial.suggest_int("target_update_interval", 1, 20000, log=True)

    net_arch = trial.suggest_categorical("net_arch", ["small", "medium", "big"])

    trial.set_user_attr("gamma", 1 - one_minus_gamma)
    trial.set_user_attr("batch_size", 2**batch_size_pow)

    hyperparams = {
        "one_minus_gamma": one_minus_gamma,
        "learning_rate": learning_rate,
        "batch_size_pow": batch_size_pow,
        "train_freq": train_freq,
        "subsample_steps": subsample_steps,
        "exploration_fraction": exploration_fraction,
        "exploration_final_eps": exploration_final_eps,
        "target_update_interval": target_update_interval,
        "net_arch": net_arch,
    }

    if additional_args["using_her_replay_buffer"]:
        hyperparams = sample_her_params(trial, hyperparams, additional_args["her_kwargs"])

    if "sample_qrdqn" in additional_args:
        n_quantiles = trial.suggest_int("n_quantiles", 5, 200)
        hyperparams["n_quantiles"] = n_quantiles

    return convert_offpolicy_params(hyperparams)


def sample_her_params(trial: optuna.Trial, hyperparams: dict[str, Any], her_kwargs: dict[str, Any]) -> dict[str, Any]:
    """
    Sampler for HerReplayBuffer hyperparams.

    :param trial:
    :parma hyperparams:
    :return:
    """
    her_kwargs = her_kwargs.copy()
    her_kwargs["n_sampled_goal"] = trial.suggest_int("n_sampled_goal", 1, 5)
    her_kwargs["goal_selection_strategy"] = trial.suggest_categorical(
        "goal_selection_strategy", ["final", "episode", "future"]
    )
    hyperparams["replay_buffer_kwargs"] = her_kwargs
    return hyperparams


def sample_tqc_params(trial: optuna.Trial, n_actions: int, n_envs: int, additional_args: dict) -> dict[str, Any]:
    """
    Sampler for TQC hyperparams.

    :param trial:
    :return:
    """
    additional_args["sample_tqc"] = True
    # TQC is SAC + Distributional RL
    return sample_sac_params(trial, n_actions, n_envs, additional_args)


def sample_qrdqn_params(trial: optuna.Trial, n_actions: int, n_envs: int, additional_args: dict) -> dict[str, Any]:
    """
    Sampler for QR-DQN hyperparams.

    :param trial:
    :return:
    """
    # TQC is DQN + Distributional RL
    additional_args["sample_qrdqn"] = True
    return sample_dqn_params(trial, n_actions, n_envs, additional_args)


def convert_ars_params(sampled_params: dict[str, Any]) -> dict[str, Any]:
    hyperparams = sampled_params.copy()

    # Note: remove bias to be as the original linear policy
    # and do not squash output
    # Comment out when doing hyperparams search with linear policy only
    # net_arch = {
    #     "linear": [],
    #     "tiny": [16],
    #     "small": [32],
    # }[net_arch]
    for name in ["n_delta"]:
        if f"{name}_pow" in sampled_params:
            hyperparams[name] = 2 ** sampled_params[f"{name}_pow"]
            del hyperparams[f"{name}_pow"]

    top_frac_size = sampled_params["top_frac_size"]
    hyperparams["n_top"] = max(int(top_frac_size * hyperparams["n_delta"]), 1)
    del hyperparams["top_frac_size"]

    return hyperparams


def sample_ars_params(trial: optuna.Trial, n_actions: int, n_envs: int, additional_args: dict) -> dict[str, Any]:
    """
    Sampler for ARS hyperparams.
    :param trial:
    :return:
    """
    # n_eval_episodes = trial.suggest_categorical("n_eval_episodes", [1, 2])
    # From 2**2 to 2**6 = 64
    n_delta_pow = trial.suggest_int("n_delta_pow", 2, 6)

    learning_rate = trial.suggest_float("learning_rate", 1e-5, 0.002, log=True)
    delta_std = trial.suggest_float("delta_std", 0.01, 0.3)
    top_frac_size = trial.suggest_float("top_frac_size", 0.05, 1.0)
    # zero_policy = trial.suggest_categorical("zero_policy", [True, False])

    # net_arch = trial.suggest_categorical("net_arch", ["linear", "tiny", "small"])
    # TODO: optimize the alive_bonus_offset too

    return convert_ars_params(
        {
            # "n_eval_episodes": n_eval_episodes,
            "n_delta_pow": n_delta_pow,
            "learning_rate": learning_rate,
            "delta_std": delta_std,
            "top_frac_size": top_frac_size,
            # "zero_policy": zero_policy,
        }
    )


HYPERPARAMS_SAMPLER = {
    "a2c": sample_a2c_params,
    "ars": sample_ars_params,
    "ddpg": sample_td3_params,
    "dqn": sample_dqn_params,
    "qrdqn": sample_qrdqn_params,
    "ppo": sample_ppo_params,
    "ppo_lstm": sample_ppo_lstm_params,
    "sac": sample_sac_params,
    "tqc": sample_tqc_params,
    "td3": sample_td3_params,
    "trpo": sample_trpo_params,
}

# Convert sampled value to hyperparameters
HYPERPARAMS_CONVERTER = {
    "a2c": convert_onpolicy_params,
    "ars": convert_ars_params,
    "ddpg": convert_offpolicy_params,
    "dqn": convert_offpolicy_params,
    "qrdqn": convert_offpolicy_params,
    "ppo": convert_onpolicy_params,
    "ppo_lstm": convert_onpolicy_params,
    "sac": convert_offpolicy_params,
    "tqc": convert_offpolicy_params,
    "td3": convert_offpolicy_params,
    "trpo": convert_onpolicy_params,
}
