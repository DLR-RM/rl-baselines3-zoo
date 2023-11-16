.. _install:

Installation
============

Prerequisites
-------------

RL Zoo requires python 3.8+ and PyTorch >= 1.13


Minimal Installation
--------------------

To install RL Zoo with pip, execute:

.. code-block:: bash

    pip install rl_zoo3

From source:

.. code-block:: bash

	git clone https://github.com/DLR-RM/rl-baselines3-zoo
	cd rl-baselines3-zoo/
	pip install -e .

.. note::

	You can do ``python -m rl_zoo3.train`` from any folder and you have access to ``rl_zoo3`` command line interface, for instance, ``rl_zoo3 train`` is equivalent to ``python train.py``



Full installation
-----------------

With extra envs and test dependencies:


.. note::

  If you want to use Atari games, you will need to do ``pip install "autorom[accept-rom-license]"``
  additionally to download the ROMs


.. code-block:: bash

	apt-get install swig cmake ffmpeg
	pip install -r requirements.txt


Please see `Stable Baselines3 documentation <https://stable-baselines3.readthedocs.io/en/master/>`_ for alternatives to install stable baselines3.


Docker Images
-------------

Build docker image (CPU):

::

   make docker-cpu

GPU:

::

   USE_GPU=True make docker-gpu

Pull built docker image (CPU):

::

   docker pull stablebaselines/rl-baselines3-zoo-cpu

GPU image:

::

   docker pull stablebaselines/rl-baselines3-zoo

Run script in the docker image:

::

   ./scripts/run_docker_cpu.sh python train.py --algo ppo --env CartPole-v1
