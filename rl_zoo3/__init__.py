import os

# isort: off

import rl_zoo3.gym_patches  # noqa: F401

# isort: on

from rl_zoo3.utils import (
    ALGOS,
    create_test_env,
    get_latest_run_id,
    get_saved_hyperparams,
    get_trained_models,
    get_wrapper_class,
    linear_schedule,
)

# Read version from file
version_file = os.path.join(os.path.dirname(__file__), "version.txt")
with open(version_file) as file_handler:
    __version__ = file_handler.read().strip()

__all__ = [
    "ALGOS",
    "create_test_env",
    "get_latest_run_id",
    "get_saved_hyperparams",
    "get_trained_models",
    "get_wrapper_class",
    "linear_schedule",
]
