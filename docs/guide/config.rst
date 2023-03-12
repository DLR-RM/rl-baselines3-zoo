.. _config:

=============
Configuration
=============

Hyperparameter yaml syntax
--------------------------

The syntax used in ``hyperparameters/algo_name.yml`` for setting
hyperparameters (likewise the syntax to `overwrite
hyperparameters <https://github.com/DLR-RM/rl-baselines3-zoo#overwrite-hyperparameters>`__
on the cli) may be specialized if the argument is a function. See
examples in the ``hyperparameters/`` directory. For example:

-  Specify a linear schedule for the learning rate:

.. code:: yaml

     learning_rate: lin_0.012486195510232303

Specify a different activation function for the network:

.. code:: yaml

     policy_kwargs: "dict(activation_fn=nn.ReLU)"

For a custom policy:

.. code:: yaml

     policy: my_package.MyCustomPolicy  # for instance stable_baselines3.ppo.MlpPolicy

Env Normalization
-----------------

In the hyperparameter file, ``normalize: True`` means that the training
environment will be wrapped in a
`VecNormalize <https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/common/vec_env/vec_normalize.py#L13>`__
wrapper.

`Normalization
uses <https://github.com/DLR-RM/rl-baselines3-zoo/issues/64>`__ the
default parameters of ``VecNormalize``, with the exception of ``gamma``
which is set to match that of the agent. This can be
`overridden <https://github.com/DLR-RM/rl-baselines3-zoo/blob/v0.10.0/hyperparams/sac.yml#L239>`__
using the appropriate ``hyperparameters/algo_name.yml``, e.g.

.. code:: yaml

    normalize: "{'norm_obs': True, 'norm_reward': False}"

Env Wrappers
------------

You can specify in the hyperparameter config one or more wrapper to use
around the environment:

for one wrapper:

.. code:: yaml

  env_wrapper: gym_minigrid.wrappers.FlatObsWrapper

for multiple, specify a list:

.. code:: yaml

  env_wrapper:
      - rl_zoo3.wrappers.TruncatedOnSuccessWrapper:
          reward_offset: 1.0
      - sb3_contrib.common.wrappers.TimeFeatureWrapper

Note that you can easily specify parameters too.

By default, the environment is wrapped with a ``Monitor`` wrapper to
record episode statistics. You can specify arguments to it using
``monitor_kwargs`` parameter to log additional data. That data *must* be
present in the info dictionary at the last step of each episode.

For instance, for recording success with goal envs
(e.g. ``FetchReach-v1``):

.. code:: yaml

  monitor_kwargs: dict(info_keywords=('is_success',))

or recording final x position with ``Ant-v3``:

.. code:: yaml

  monitor_kwargs: dict(info_keywords=('x_position',))

Note: for known ``GoalEnv`` like ``FetchReach``,
``info_keywords=('is_success',)`` is actually the default.

VecEnvWrapper
-------------

You can specify which ``VecEnvWrapper`` to use in the config, the same
way as for env wrappers (see above), using the ``vec_env_wrapper`` key:

For instance:

.. code:: yaml

  vec_env_wrapper: stable_baselines3.common.vec_env.VecMonitor

Note: ``VecNormalize`` is supported separately using ``normalize``
keyword, and ``VecFrameStack`` has a dedicated keyword ``frame_stack``.

Callbacks
---------

Following the same syntax as env wrappers, you can also add custom
callbacks to use during training.

.. code:: yaml

  callback:
    - rl_zoo3.callbacks.ParallelTrainCallback:
        gradient_steps: 256
