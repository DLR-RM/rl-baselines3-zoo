"""THis file just serves as an example on how to configure the zoo using python scripts instead of yaml files."""

hyperparams = {
    "MountainCarContinuous-v0":
    dict(normalize=True,
         n_envs=1,
         n_timesteps=20000.0,
         policy='MlpPolicy',
         batch_size=256,
         n_steps=8,
         gamma=0.9999,
         learning_rate=7.77e-05,
         ent_coef=0.00429,
         clip_range=0.1,
         n_epochs=10,
         gae_lambda=0.9,
         max_grad_norm=5,
         vf_coef=0.19,
         use_sde=True,
         policy_kwargs=dict(log_std_init=-3.29, ortho_init=False))
}