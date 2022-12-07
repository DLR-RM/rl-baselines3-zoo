import imitation

hyperparams = {
    "seals/CartPole-v0": dict(
        n_envs=8, n_timesteps=500000.0, policy="MlpPolicy", ent_coef=0.0
    ),
    "seals/MountainCar-v0": dict(
        normalize=dict(norm_obs=False, norm_reward=True),
        n_envs=16,
        n_timesteps=1000000.0,
        policy="MlpPolicy",
        ent_coef=0.0,
        policy_kwargs=dict(
            features_extractor_class=imitation.policies.base.NormalizeFeaturesExtractor
        ),
    ),
    "seals/HalfCheetah-v0": dict(
        normalize=dict(norm_obs=False, norm_reward=True),
        n_timesteps=1000000.0,
        policy="MlpPolicy",
        policy_kwargs=dict(
            features_extractor_class=imitation.policies.base.NormalizeFeaturesExtractor
        ),
    ),
    "seals/Ant-v0": dict(
        normalize=dict(norm_obs=False, norm_reward=True),
        n_timesteps=1000000.0,
        policy="MlpPolicy",
        policy_kwargs=dict(
            features_extractor_class=imitation.policies.base.NormalizeFeaturesExtractor
        ),
    ),
    "seals/Hopper-v0": dict(
        normalize=dict(norm_obs=False, norm_reward=True),
        n_timesteps=1000000.0,
        policy="MlpPolicy",
        policy_kwargs=dict(
            features_extractor_class=imitation.policies.base.NormalizeFeaturesExtractor
        ),
    ),
    "seals/Walker2d-v0": dict(
        normalize=dict(norm_obs=False, norm_reward=True),
        n_timesteps=1000000.0,
        policy="MlpPolicy",
        policy_kwargs=dict(
            features_extractor_class=imitation.policies.base.NormalizeFeaturesExtractor
        ),
    ),
    "seals/Humanoid-v0": dict(
        normalize=dict(norm_obs=False, norm_reward=True),
        n_timesteps=2000000.0,
        policy="MlpPolicy",
        policy_kwargs=dict(
            features_extractor_class=imitation.policies.base.NormalizeFeaturesExtractor
        ),
    ),
    "seals/Swimmer-v0": dict(
        normalize=dict(norm_obs=False, norm_reward=True),
        n_timesteps=1000000.0,
        policy="MlpPolicy",
        gamma=0.9999,
        policy_kwargs=dict(
            features_extractor_class=imitation.policies.base.NormalizeFeaturesExtractor
        ),
    ),
}
