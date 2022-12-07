hyperparams = {
    "seals/HalfCheetah-v0": dict(
        n_timesteps=1000000.0, policy="MlpPolicy", learning_starts=10000
    ),
    "seals/Ant-v0": dict(
        n_timesteps=1000000.0, policy="MlpPolicy", learning_starts=10000
    ),
    "seals/Hopper-v0": dict(
        n_timesteps=1000000.0,
        policy="MlpPolicy",
        learning_starts=10000,
        top_quantiles_to_drop_per_net=5,
    ),
    "seals/Walker2d-v0": dict(
        n_timesteps=1000000.0, policy="MlpPolicy", learning_starts=10000
    ),
    "seals/Humanoid-v0": dict(
        n_timesteps=2000000.0, policy="MlpPolicy", learning_starts=10000
    ),
    "seals/Swimmer-v0": dict(
        n_timesteps=1000000.0, policy="MlpPolicy", learning_starts=10000, gamma=0.9999
    ),
}
