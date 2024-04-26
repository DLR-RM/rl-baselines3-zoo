import argparse
import sys
from typing import Dict

import wandb
from stable_baselines3.common.logger import Logger, HumanOutputFormat

from rl_zoo3.tracking.tracking_backend import TrackingBackend


class WandbBackend(TrackingBackend):
    def __init__(self):
        super().__init__()
        self.run = None

    @classmethod
    def argparse_add_arguments(cls, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--wandb-entity", type=str, default=None, help="the entity (team) of wandb's project")
        parser.add_argument("--wandb-project-name", type=str, default="sb3", help="the wandb's project name")
        parser.add_argument(
            "-tags", "--wandb-tags", type=str, default=[], nargs="+", help="Tags for wandb run, e.g.: -tags optimized pr-123"
        )

    @classmethod
    def argparse_del_arguments(cls, parsed_args) -> None:
        del parsed_args.mlflow_experiment_name
        del parsed_args.mlflow_run_description
        del parsed_args.mlflow_tags
        del parsed_args.mlflow_tracking_uri

    def _setup_tracking(self, args) -> None:
        assert "wandb_entity" in vars(args), "Missing W&B entity (--wandb-entity)."

        tags = set([f"{k}-{v}" for k, v in self.get_tracking_commit_hashes().items()])
        tags.update([f"{k}-{v}" for k, v in self.get_version_tags().items()])

        self.run = wandb.init(
            name=args.run_name,
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            tags=tags,
            config=vars(args),
            sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
            monitor_gym=True,  # auto-upload the videos of agents playing the game
            save_code=True,  # optional
        )

    def _finish_tracking(self) -> None:
        pass

    def _get_sb3_logger(self, verbose: bool) -> Logger | None:
        loggers = []
        if verbose:
            loggers.append(HumanOutputFormat(sys.stdout))
        return Logger(folder=None, output_formats=loggers)

    def _log_params(self, params: Dict) -> None:
        assert self.run is not None  # make mypy happy
        self.run.config.setdefaults(params)

    def _log_directory(self, local_dir: str, artifacts_dir: str = None) -> None:
        raise NotImplementedError("This feature has been not implemented for W&B.")
