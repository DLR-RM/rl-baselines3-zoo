.. _custom:

==================
Custom Environment
==================

The easiest way to add support for a custom or external environment is to import it using the ``--gym-packages my_package`` argument or to edit
``rl_zoo3/import_envs.py`` and register your environment here. Then, you
need to add a section for it in the hyperparameters file
(``hyperparams/algo.yml`` or a custom yaml file that you can specify
using ``--conf-file`` argument).
