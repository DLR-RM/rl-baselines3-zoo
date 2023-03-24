.. _train:

==============
Train an Agent
==============

Basic Usage
-----------

The hyperparameters for each environment are defined in
``hyperparameters/algo_name.yml``.


.. note::

	Once RL Zoo3 is install, you can do ``python -m rl_zoo3.train`` from any folder, it is equivalent to ``python train.py``


If the environment exists in this file, then you can train an agent using:

::

  python train.py --algo algo_name --env env_id


.. note::

	You can use ``-P`` (``--progress``) option to display a progress bar.


Custom Config File
------------------

Using a custom config file when it is a yaml file with a which contains a ``env_id`` entry:

::

  python train.py --algo algo_name --env env_id --conf-file my_yaml.yml


You can also use a python file that contains a dictionary called `hyperparams` with an entry for each ``env_id``.
(see ``hyperparams/python/ppo_config_example.py`` for an example)

::

  # You can pass a path to a python file
  python train.py --algo ppo --env MountainCarContinuous-v0 --conf-file hyperparams/python/ppo_config_example.py
  # Or pass a path to a file from a module (for instance my_package.my_file)
  python train.py --algo ppo --env MountainCarContinuous-v0 --conf-file hyperparams.python.ppo_config_example

The advantage of this approach is that you can specify arbitrary python dictionaries
and ensure that all their dependencies are imported in the config file itself.

Tensorboard, Checkpoints, Evaluation
------------------------------------

For example (with tensorboard support):

::

  python train.py --algo ppo --env CartPole-v1 --tensorboard-log /tmp/stable-baselines/


Evaluate the agent every 10000 steps using 10 episodes for evaluation (using only one evaluation env):

::

  python train.py --algo sac --env AntBulletEnv-v0 --eval-freq 10000 --eval-episodes 10 --n-eval-envs 1


Save a checkpoint of the agent every 100000 steps:

::

  python train.py --algo td3 --env AntBulletEnv-v0 --save-freq 100000

Resume Training
---------------

Continue training (here, load pretrained agent for Breakout and continue training for 5000 steps):

::

  python train.py --algo a2c --env BreakoutNoFrameskip-v4 -i rl-trained-agents/a2c/BreakoutNoFrameskip-v4_1/BreakoutNoFrameskip-v4.zip -n 5000

Save Replay Buffer
------------------

When using off-policy algorithms, you can also **save the replay buffer** after training:

::

  python train.py --algo sac --env Pendulum-v1 --save-replay-buffer

It will be automatically loaded if present when continuing training.


Env keyword arguments
---------------------

You can specify keyword arguments to pass to the env constructor in the
command line, using ``--env-kwargs``:

::

   python enjoy.py --algo ppo --env MountainCar-v0 --env-kwargs goal_velocity:10


Overwrite hyperparameters
-------------------------

You can easily overwrite hyperparameters in the command line, using
``--hyperparams``:

::

   python train.py --algo a2c --env MountainCarContinuous-v0 --hyperparams learning_rate:0.001 policy_kwargs:"dict(net_arch=[64, 64])"

Note: if you want to pass a string, you need to escape it like that:
``my_string:"'value'"``
