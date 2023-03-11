.. _plot:

============
Plot Scripts
============


Plot scripts (to be documented, see "Results" sections in SB3
documentation):

- ``scripts/all_plots.py``/``scripts/plot_from_file.py`` for plotting evaluations

- ``scripts/plot_train.py`` for plotting training reward/success


Examples
--------

Plot training success (y-axis) w.r.t. timesteps (x-axis) with a moving
window of 500 episodes for all the ``Fetch`` environment with ``HER``
algorithm:

::

   python scripts/plot_train.py -a her -e Fetch -y success -f rl-trained-agents/ -w 500 -x steps

Plot evaluation reward curve for TQC, SAC and TD3 on the HalfCheetah and
Ant PyBullet environments:

::

   python3 scripts/all_plots.py -a sac td3 tqc --env HalfCheetahBullet AntBullet -f rl-trained-agents/

Plot with the rliable library
-----------------------------

The RL zoo integrates some of
`rliable <https://agarwl.github.io/rliable/>`__ library features. You
can find a visual explanation of the tools used by rliable in this `blog
post <https://araffin.github.io/post/rliable/>`__.

First, you need to install
`rliable <https://github.com/google-research/rliable>`__.

Note: Python 3.7+ is required in that case.

Then export your results to a file using the ``all_plots.py`` script
(see above):

::

   python scripts/all_plots.py -a sac td3 tqc --env Half Ant -f logs/ -o logs/offpolicy

You can now use the ``plot_from_file.py`` script with ``--rliable``,
``--versus`` and ``--iqm`` arguments:

::

   python scripts/plot_from_file.py -i logs/offpolicy.pkl --skip-timesteps --rliable --versus -l SAC TD3 TQC

.. note::

  you may need to edit ``plot_from_file.py``, in particular the
  ``env_key_to_env_id`` dictionary and the
  ``scripts/score_normalization.py`` which stores min and max score for
  each environment.


Remark: plotting with the ``--rliable`` option is usually slow as
confidence interval need to be computed using bootstrap sampling.
