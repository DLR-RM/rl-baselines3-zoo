.. _examples:

Examples
========

.. note::

        These examples are only to demonstrate the use of the library and its functions, and the trained agents may not solve the environments. Optimized               hyperparameters can be found in the RL Zoo `repository <https://github.com/DLR-RM/rl-baselines3-zoo>`_.


Try it online with Colab Notebooks!
-----------------------------------

All the following examples can be executed online using Google colab |colab|
notebooks:

-  `Full Tutorial <https://github.com/araffin/rl-tutorial-jnrr19/tree/sb3>`_
-  `All Notebooks <https://github.com/Stable-Baselines-Team/rl-colab-notebooks/tree/sb3>`_
-  `Getting Started`_
-  `Training, Saving, Loading`_
-  `Multiprocessing`_
-  `Monitor Training and Plotting`_
-  `Atari Games`_
-  `RL Baselines zoo`_
-  `PyBullet`_
-  `Hindsight Experience Replay`_
-  `Advanced Saving and Loading`_

.. _Getting Started: https://colab.research.google.com/github/Stable-Baselines-Team/rl-colab-notebooks/blob/sb3/stable_baselines_getting_started.ipynb
.. _Training, Saving, Loading: https://colab.research.google.com/github/Stable-Baselines-Team/rl-colab-notebooks/blob/sb3/saving_loading_dqn.ipynb
.. _Multiprocessing: https://colab.research.google.com/github/Stable-Baselines-Team/rl-colab-notebooks/blob/sb3/multiprocessing_rl.ipynb
.. _Monitor Training and Plotting: https://colab.research.google.com/github/Stable-Baselines-Team/rl-colab-notebooks/blob/sb3/monitor_training.ipynb
.. _Atari Games: https://colab.research.google.com/github/Stable-Baselines-Team/rl-colab-notebooks/blob/sb3/atari_games.ipynb
.. _Hindsight Experience Replay: https://colab.research.google.com/github/Stable-Baselines-Team/rl-colab-notebooks/blob/sb3/stable_baselines_her.ipynb
.. _RL Baselines zoo: https://colab.research.google.com/github/Stable-Baselines-Team/rl-colab-notebooks/blob/sb3/rl-baselines-zoo.ipynb
.. _PyBullet: https://colab.research.google.com/github/Stable-Baselines-Team/rl-colab-notebooks/blob/sb3/pybullet.ipynb
.. _Advanced Saving and Loading: https://colab.research.google.com/github/Stable-Baselines-Team/rl-colab-notebooks/blob/sb3/advanced_saving_loading.ipynb

.. |colab| image:: ../_static/img/colab.svg

Basic Usage: Training, Saving, Loading
--------------------------------------

In the following example, we will train, save and load a DQN model on the Lunar Lander environment.

.. image:: ../_static/img/colab-badge.svg
   :target: https://colab.research.google.com/github/Stable-Baselines-Team/rl-colab-notebooks/blob/sb3/saving_loading_dqn.ipynb


.. code-block:: python

  import gym

  from stable_baselines3 import DQN
  from stable_baselines3.common.evaluation import evaluate_policy


  # Create environment
  env = gym.make("LunarLander-v2")

  # Instantiate the agent
  model = DQN("MlpPolicy", env, verbose=1)
  # Train the agent and display a progress bar
  model.learn(total_timesteps=int(2e5), progress_bar=True)
  # Save the agent
  model.save("dqn_lunar")
  del model  # delete trained model to demonstrate loading

  # Load the trained agent
  # NOTE: if you have loading issue, you can pass `print_system_info=True`
  # to compare the system on which the model was trained vs the current one
  # model = DQN.load("dqn_lunar", env=env, print_system_info=True)
  model = DQN.load("dqn_lunar", env=env)

  # Evaluate the agent
  # NOTE: If you use wrappers with your environment that modify rewards,
  #       this will be reflected here. To evaluate with original rewards,
  #       wrap environment in a "Monitor" wrapper before other wrappers.
  mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)

  # Enjoy trained agent
  vec_env = model.get_env()
  obs = vec_env.reset()
  for i in range(1000):
      action, _states = model.predict(obs, deterministic=True)
      obs, rewards, dones, info = vec_env.step(action)
      vec_env.render()
