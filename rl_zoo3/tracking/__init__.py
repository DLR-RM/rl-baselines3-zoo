from rl_zoo3.tracking.args_parse import argparse_add_track_arguments, argparse_filter_track_arguments
from rl_zoo3.tracking.git import UncommittedChangesError
from rl_zoo3.tracking.tracking_backend import TrackingBackend

__all__ = [
    "argparse_add_track_arguments",
    "argparse_filter_track_arguments",
    "UncommittedChangesError",
    "TrackingBackend",
]

# Automatic detection of available ML tracking packages
try:
    from rl_zoo3.tracking.wandb import WandbBackend

    __all__.append("WandbBackend")
except ImportError:
    pass

try:
    from rl_zoo3.tracking.mlflow import MLflowBackend

    __all__.append("MLflowBackend")
except ImportError:
    pass
