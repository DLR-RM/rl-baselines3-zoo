.. _enjoy:

=====================
Enjoy a Trained Agent
=====================

.. note::

  To download the repo with the trained agents, you must use
  ``git clone --recursive https://github.com/DLR-RM/rl-baselines3-zoo``
  in order to clone the submodule too.


Enjoy a trained agent
---------------------

If the trained agent exists, then you can see it in action using:

::

   python enjoy.py --algo algo_name --env env_id

For example, enjoy A2C on Breakout during 5000 timesteps:

::

   python enjoy.py --algo a2c --env BreakoutNoFrameskip-v4 --folder rl-trained-agents/ -n 5000

If you have trained an agent yourself, you need to do:

::

   # exp-id 0 corresponds to the last experiment, otherwise, you can specify another ID
   python enjoy.py --algo algo_name --env env_id -f logs/ --exp-id 0

Load Checkpoints, Best Model
-----------------------------

To load the best model (when using evaluation environment):

::

   python enjoy.py --algo algo_name --env env_id -f logs/ --exp-id 1 --load-best

To load a checkpoint (here the checkpoint name is
``rl_model_10000_steps.zip``):

::

   python enjoy.py --algo algo_name --env env_id -f logs/ --exp-id 1 --load-checkpoint 10000

To load the latest checkpoint:

::

   python enjoy.py --algo algo_name --env env_id -f logs/ --exp-id 1 --load-last-checkpoint


Record a Video of a Trained Agent
---------------------------------

Record 1000 steps with the latest saved model:

::

  python -m rl_zoo3.record_video --algo ppo --env BipedalWalkerHardcore-v3 -n 1000

Use the best saved model instead:

::

  python -m rl_zoo3.record_video --algo ppo --env BipedalWalkerHardcore-v3 -n 1000 --load-best

Record a video of a checkpoint saved during training (here the
checkpoint name is ``rl_model_10000_steps.zip``):

::

  python -m rl_zoo3.record_video --algo ppo --env BipedalWalkerHardcore-v3 -n 1000 --load-checkpoint 10000


Record a Video of a Training Experiment
---------------------------------------

Apart from recording videos of specific saved models, it is also
possible to record a video of a training experiment where checkpoints
have been saved.

Record 1000 steps for each checkpoint, latest and best saved models:

::

  python -m rl_zoo3.record_training --algo ppo --env CartPole-v1 -n 1000 -f logs --deterministic

The previous command will create a ``mp4`` file. To convert this file to
``gif`` format as well:

::

  python -m rl_zoo3.record_training --algo ppo --env CartPole-v1 -n 1000 -f logs --deterministic --gif
