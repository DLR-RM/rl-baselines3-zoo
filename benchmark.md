
## Performance of trained agents

Final performance of the trained agents can be found in the table below.
This was computed by running `python -m utils.benchmark`:
it runs the trained agent for `n_timesteps` and then reports the mean episode reward
during this evaluation.

It uses the deterministic policy except for Atari games.

*NOTE: this is not a quantitative benchmark as it corresponds to only one run
(cf [issue #38](https://github.com/araffin/rl-baselines-zoo/issues/38)).
This benchmark is meant to check algorithm (maximal) performance, find potential bugs
and also allow users to have access to pretrained agents.*

|algo|  env_id   |mean_reward|std_reward|n_timesteps|n_episodes|
|----|-----------|----------:|---------:|----------:|---------:|
|a2c |CartPole-v1|      500.0|      0.00|     150000|       300|
|ppo |CartPole-v1|      500.0|      0.00|     150000|       300|
|sac |Pendulum-v0|     -157.0|     88.71|     150000|       750|
