
## Performance of trained agents

Final performance of the trained agents can be found in the table below.
This was computed by running `python -m utils.benchmark`:
it runs the trained agent (trained on `n_timesteps`) for `eval_timesteps` and then reports the mean episode reward
during this evaluation.

It uses the deterministic policy except for Atari games.

*NOTE: this is not a quantitative benchmark as it corresponds to only one run
(cf [issue #38](https://github.com/araffin/rl-baselines-zoo/issues/38)).
This benchmark is meant to check algorithm (maximal) performance, find potential bugs
and also allow users to have access to pretrained agents.*

"M" stands for Million (1e6)

|algo|  env_id   |mean_reward|std_reward|n_timesteps|eval_timesteps|eval_episodes|
|----|-----------|----------:|---------:|-----------|-------------:|------------:|
|a2c |CartPole-v1|      500.0|      0.00|500k       |          1000|            2|
|ppo |CartPole-v1|      500.0|      0.00|100k       |          1000|            2|
|sac |Pendulum-v0|     -202.0|     60.50|20k        |          1000|            5|
