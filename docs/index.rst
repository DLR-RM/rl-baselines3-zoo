RL Baselines3 Zoo Docs - A Training Framework for Stable Baselines3
===================================================================

`RL Baselines3 Zoo  <https://github.com/DLR-RM/stable-baselines3>`_  s a training framework for Reinforcement Learning (RL), using `Stable Baselines3 (SB3) <https://github.com/DLR-RM/stable-baselines3>`_,
reliable implementations of reinforcement learning algorithms in PyTorch.

Github repository: https://github.com/DLR-RM/rl-baselines3-zoo

It provides scripts for training, evaluating agents, tuning hyperparameters, plotting results and recording videos.

In addition, it includes a collection of tuned hyperparameters for common environments and RL algorithms, and agents trained with those settings.

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   guide/install
   guide/quickstart
   guide/train
   guide/plot
   guide/enjoy
   guide/custom_env
   guide/config
   guide/integrations
   guide/tuning
   guide/sbx


.. toctree::
  :maxdepth: 1
  :caption: RL Zoo API

  modules/exp_manager
  modules/wrappers
  modules/callbacks
  modules/utils

.. toctree::
  :maxdepth: 1
  :caption: Misc

  misc/changelog


Citing RL Baselines3 Zoo
------------------------
To cite this project in publications:

.. code-block:: bibtex

  @misc{rl-zoo3,
    author = {Raffin, Antonin},
    title = {RL Baselines3 Zoo},
    year = {2020},
    publisher = {GitHub},
    journal = {GitHub repository},
    howpublished = {\url{https://github.com/DLR-RM/rl-baselines3-zoo}},
  }

Contributing
------------

To any interested in making the rl baselines better, there are still some improvements
that need to be done.
You can check issues in the `repo <https://github.com/DLR-RM/rl-baselines3-zoo/issues>`_.

If you want to contribute, please read `CONTRIBUTING.md <https://github.com/DLR-RM/stable-baselines3/blob/master/CONTRIBUTING.md>`_ first.

Indices and tables
-------------------

* :ref:`genindex`
* :ref:`search`
* :ref:`modindex`
