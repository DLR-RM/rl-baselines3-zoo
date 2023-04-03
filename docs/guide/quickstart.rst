.. _quickstart:

===============
Getting Started
===============

.. note::

  You can try the following examples online using Google Colab |Colab|
  notebook: `RL Baselines zoo notebook`_


.. _RL Baselines zoo notebook: https://colab.research.google.com/github/Stable-Baselines-Team/rl-colab-notebooks/blob/sb3/rl-baselines-zoo.ipynb
.. |Colab| image:: ../_static/img/colab.svg


The hyperparameters for each environment are defined in
``hyperparameters/algo_name.yml``.

If the environment exists in this file, then you can train an agent
using:

::

 python -m rl_zoo3.train --algo algo_name --env env_id

Or if you are in the RL Zoo3 folder:

::

  python train.py --algo algo_name --env env_id

For example (with evaluation and checkpoints):

::

 python -m rl_zoo3.train --algo ppo --env CartPole-v1 --eval-freq 10000 --save-freq 50000



If the trained agent exists, then you can see it in action using:

::

 python -m rl_zoo3.enjoy --algo algo_name --env env_id

For example, enjoy A2C on Breakout during 5000 timesteps:

::

 python -m rl_zoo3.enjoy --algo a2c --env BreakoutNoFrameskip-v4 --folder rl-trained-agents/ -n 5000
