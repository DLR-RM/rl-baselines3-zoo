import imitation

hyperparams = {
    "seals/CartPole-v0": dict(
        n_envs=1, n_timesteps=50000.0, policy="LinearPolicy", n_delta=2
    ),
    "seals/MountainCar-v0": dict(
        n_envs=1,
        n_timesteps=500000.0,
        policy="MlpPolicy",
        normalize=dict(norm_obs=False, norm_reward=False),
        learning_rate=0.018,
        n_delta=8,
        n_top=1,
        delta_std=0.1,
        policy_kwargs=dict(
            net_arch=[16],
            features_extractor_class=imitation.policies.base.NormalizeFeaturesExtractor,
        ),
        zero_policy=False,
    ),
    "seals/Swimmer-v0": dict(
        n_envs=1,
        policy="LinearPolicy",
        n_timesteps=2000000.0,
        learning_rate=0.02,
        delta_std=0.01,
        n_delta=1,
        n_top=1,
        alive_bonus_offset=0,
    ),
    "seals/Hopper-v0": dict(
        n_envs=1,
        policy="LinearPolicy",
        n_timesteps=7000000.0,
        learning_rate=0.01,
        delta_std=0.025,
        n_delta=8,
        n_top=4,
        alive_bonus_offset=-1,
        normalize=dict(norm_obs=False, norm_reward=False),
        policy_kwargs=dict(
            features_extractor_class=imitation.policies.base.NormalizeFeaturesExtractor
        ),
    ),
    "seals/HalfCheetah-v0": dict(
        n_envs=1,
        policy="LinearPolicy",
        n_timesteps=12500000.0,
        learning_rate=0.02,
        delta_std=0.03,
        n_delta=32,
        n_top=4,
        alive_bonus_offset=0,
        normalize=dict(norm_obs=False, norm_reward=False),
        policy_kwargs=dict(
            features_extractor_class=imitation.policies.base.NormalizeFeaturesExtractor
        ),
    ),
    "seals/Walker2d-v0": dict(
        n_envs=1,
        policy="LinearPolicy",
        n_timesteps=75000000.0,
        learning_rate=0.03,
        delta_std=0.025,
        n_delta=40,
        n_top=30,
        alive_bonus_offset=-1,
        normalize=dict(norm_obs=False, norm_reward=False),
        policy_kwargs=dict(
            features_extractor_class=imitation.policies.base.NormalizeFeaturesExtractor
        ),
    ),
    "seals/Ant-v0": dict(
        n_envs=1,
        policy="LinearPolicy",
        n_timesteps=75000000.0,
        learning_rate=0.015,
        delta_std=0.025,
        n_delta=60,
        n_top=20,
        alive_bonus_offset=-1,
        normalize=dict(norm_obs=False, norm_reward=False),
        policy_kwargs=dict(
            features_extractor_class=imitation.policies.base.NormalizeFeaturesExtractor
        ),
    ),
    "seals/Humanoid-v0": dict(
        n_envs=1,
        policy="LinearPolicy",
        n_timesteps=250000000.0,
        learning_rate=0.02,
        delta_std=0.0075,
        n_delta=256,
        n_top=256,
        alive_bonus_offset=-5,
        normalize=dict(norm_obs=False, norm_reward=False),
        policy_kwargs=dict(
            features_extractor_class=imitation.policies.base.NormalizeFeaturesExtractor
        ),
    ),
}
