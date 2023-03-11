.. _install:

Installation
============

Prerequisites
-------------

RL Zoo requires python 3.7+ and PyTorch >= 1.11


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

.. code-block:: bash

	apt-get install swig cmake ffmpeg
	pip install -r requirements.txt


Please see `Stable Baselines3 documentation <https://stable-baselines3.readthedocs.io/en/master/>`_ for alternatives to install stable baselines3.
