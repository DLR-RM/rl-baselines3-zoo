(config)=

# Configuration

## Hyperparameter YAML syntax

The syntax used in `hyperparameters/algo_name.yml` for setting
hyperparameters (likewise the syntax to [overwrite
hyperparameters](https://github.com/DLR-RM/rl-baselines3-zoo#overwrite-hyperparameters)
on the cli) may be specialized if the argument is a function. See
examples in the `hyperparameters/` directory. For example:

- Specify a linear schedule for the learning rate:

```yaml
learning_rate: lin_0.012486195510232303
```

Specify a different activation function for the network:

```yaml
policy_kwargs: "dict(activation_fn=nn.ReLU)"
```

For a custom policy:

```yaml
policy: my_package.MyCustomPolicy  # for instance stable_baselines3.ppo.MlpPolicy
```

## Env Normalization

In the hyperparameter file, `normalize: True` means that the training
environment will be wrapped in a
[VecNormalize](https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/common/vec_env/vec_normalize.py#L13)
wrapper.

[Normalization
uses](https://github.com/DLR-RM/rl-baselines3-zoo/issues/64) the
default parameters of `VecNormalize`, with the exception of `gamma`
which is set to match that of the agent. This can be
[overridden](https://github.com/DLR-RM/rl-baselines3-zoo/blob/v0.10.0/hyperparams/sac.yml#L239)
using the appropriate `hyperparameters/algo_name.yml`, e.g.

```yaml
normalize: "{'norm_obs': True, 'norm_reward': False}"
```

## Env Wrappers

You can specify in the hyperparameter config one or more wrapper to use
around the environment:

for one wrapper:

```yaml
env_wrapper: gym_minigrid.wrappers.FlatObsWrapper
```

for multiple, specify a list:

```yaml
env_wrapper:
    - rl_zoo3.wrappers.TruncatedOnSuccessWrapper:
        reward_offset: 1.0
    - sb3_contrib.common.wrappers.TimeFeatureWrapper
```

Note that you can easily specify parameters too.

By default, the environment is wrapped with a `Monitor` wrapper to
record episode statistics. You can specify arguments to it using
`monitor_kwargs` parameter to log additional data. That data *must* be
present in the info dictionary at the last step of each episode.

For instance, for recording success with goal envs
(e.g. `FetchReach-v1`):

```yaml
monitor_kwargs: dict(info_keywords=('is_success',))
```

or recording final x position with `Ant-v3`:

```yaml
monitor_kwargs: dict(info_keywords=('x_position',))
```

Note: for known `GoalEnv` like `FetchReach`,
`info_keywords=('is_success',)` is actually the default.

You can also specify environment keyword arguments with:

```yaml
env_kwargs:
  gravity: 0.0
```

## VecEnvWrapper

You can specify which `VecEnvWrapper` to use in the config, the same
way as for env wrappers (see above), using the `vec_env_wrapper` key:

For instance:

```yaml
vec_env_wrapper: stable_baselines3.common.vec_env.VecMonitor
```

Note: `VecNormalize` is supported separately using `normalize`
keyword, and `VecFrameStack` has a dedicated keyword `frame_stack`.

## Callbacks

Following the same syntax as env wrappers, you can also add custom
callbacks to use during training.

```yaml
callback:
  - rl_zoo3.callbacks.ParallelTrainCallback:
      gradient_steps: 256
```

## Default Hyperparameters

You can use a `default` entry in your hyperparameter YAML file to provide fallback hyperparameters for environments that don't have specific entries.
This is useful when training on environments for which you don't have tuned hyperparameters.

The `default` hyperparameters will be used when:
1. The environment is not explicitly listed in the config file
2. The environment is not an Atari game (which uses the `atari` entry)

Example:

```yaml
# Specific hyperparameters for CartPole-v1
CartPole-v1:
  n_envs: 8
  n_timesteps: !!float 1e5
  policy: 'MlpPolicy'
  learning_rate: 1e-3

# Fallback hyperparameters for any other environment
default:
  n_envs: 4
  n_timesteps: !!float 1e6
  policy: 'MlpPolicy'
```

When training on an environment not explicitly listed, the Zoo will print `Using 'default' hyperparameters` and apply the default settings.
