.. _integrations:

============
Integrations
============

Huggingface Hub Integration
---------------------------

List and videos of trained agents can be found on our Huggingface page: https://huggingface.co/sb3


Upload model to hub (same syntax as for ``enjoy.py``):

::

   python -m rl_zoo3.push_to_hub --algo ppo --env CartPole-v1 -f logs/ -orga sb3 -m "Initial commit"

you can choose custom ``repo-name`` (default: ``{algo}-{env_id}``) by
passing a ``--repo-name`` argument.

Download model from hub:

::

   python -m rl_zoo3.load_from_hub --algo ppo --env CartPole-v1 -f logs/ -orga sb3


Experiment tracking
-------------------

We support tracking experiment data such as learning curves and
hyperparameters via `Weights and Biases <https://wandb.ai>`__.

The following command

::

  python train.py --algo ppo --env CartPole-v1 --track --wandb-project-name sb3

yields a tracked experiment at this
`URL <https://wandb.ai/openrlbenchmark/sb3/runs/1b65ldmh>`__.

To add a tag to the run, (e.g. ``optimized``), use the argument
``--wandb-tags optimized``.
