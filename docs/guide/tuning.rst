.. _tuning:

=====================
Hyperparameter Tuning
=====================

Hyperparameter Tuning
---------------------

We use `Optuna <https://optuna.org/>`__ for optimizing the
hyperparameters. Not all hyperparameters are tuned, and tuning enforces
certain default hyperparameter settings that may be different from the
official defaults. See
`rl_zoo3/hyperparams_opt.py <https://github.com/DLR-RM/rl-baselines3-zoo/blob/master/rl_zoo3/hyperparams_opt.py>`__
for the current settings for each agent.

Hyperparameters not specified in
`rl_zoo3/hyperparams_opt.py <https://github.com/DLR-RM/rl-baselines3-zoo/blob/master/rl_zoo3/hyperparams_opt.py>`__
are taken from the associated YAML file and fallback to the default
values of SB3 if not present.

Note: when using SuccessiveHalvingPruner (“halving”), you must specify
``--n-jobs > 1``

Budget of 1000 trials with a maximum of 50000 steps:

::

   python train.py --algo ppo --env MountainCar-v0 -n 50000 -optimize --n-trials 1000 --n-jobs 2 \
     --sampler tpe --pruner median

Distributed optimization using a shared database is also possible (see
the corresponding `Optuna
documentation <https://optuna.readthedocs.io/en/stable/tutorial/10_key_features/004_distributed.html>`__):

::

   python train.py --algo ppo --env MountainCar-v0 -optimize --study-name test --storage sqlite:///example.db

Print and save best hyperparameters of an Optuna study:

::

   python scripts/parse_study.py -i path/to/study.pkl --print-n-best-trials 10 --save-n-best-hyperparameters 10

The default budget for hyperparameter tuning is 500 trials and there is
one intermediate evaluation for pruning/early stopping per 100k time
steps.

Hyperparameters search space
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Note that the default hyperparameters used in the zoo when tuning are
not always the same as the defaults provided in
`stable-baselines3 <https://stable-baselines3.readthedocs.io/en/master/modules/base.html>`__.
Consult the latest source code to be sure of these settings. For
example:

-  PPO tuning assumes a network architecture with ``ortho_init = False``
   when tuning, though it is ``True`` by
   `default <https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html#ppo-policies>`__.
   You can change that by updating
   `rl_zoo3/hyperparams_opt.py <https://github.com/DLR-RM/rl-baselines3-zoo/blob/master/rl_zoo3/hyperparams_opt.py>`__.

-  Non-episodic rollout in TD3 and DDPG assumes
   ``gradient_steps = train_freq`` and so tunes only ``train_freq`` to
   reduce the search space.

When working with continuous actions, we recommend to enable
`gSDE <https://arxiv.org/abs/2005.05719>`__ by uncommenting lines in
`rl_zoo3/hyperparams_opt.py <https://github.com/DLR-RM/rl-baselines3-zoo/blob/master/rl_zoo3/hyperparams_opt.py>`__.
