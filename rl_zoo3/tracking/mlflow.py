import argparse
import json
import os
import sys
from typing import Any, Dict, Tuple, Union

import mlflow
import numpy as np

from stable_baselines3.common.logger import KVWriter
from stable_baselines3.common.logger import Logger, HumanOutputFormat
from rl_zoo3.tracking.tracking_backend import TrackingBackend


class MLflowBackend(TrackingBackend):
    def __init__(self):
        super().__init__()

    @classmethod
    def argparse_add_arguments(cls, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--mlflow-experiment-name", type=str, default="sb3", help="the mlflow experiment name")
        parser.add_argument(
            "--mlflow-run-description", type=str, default="Generic run.", help="the description for the mlflow run"
        )
        parser.add_argument(
            "--mlflow-tracking-uri", type=str, default="http://mlflow:5000", help="the uri of the mlflow server"
        )

        def parse_json_tags(json_str):
            if json_str is None or not len(json_str):
                return dict()
            return json.loads(json_str)

        parser.add_argument(
            "--mlflow-tags",
            type=parse_json_tags,
            default=dict(),
            help='Extra args for mlflow experiment provided as JSON string e.g.: --mlflow-tags \'{"Optimized": "true", "Name": "Value"}\')',
        )

    @classmethod
    def argparse_del_arguments(cls, parsed_args) -> None:
        del parsed_args.wandb_entity
        del parsed_args.wandb_project_name
        del parsed_args.wandb_tags

    def _setup_tracking(self, args) -> None:
        logname = os.environ["LOGNAME"]
        logging_user_name = f"{os.environ['HOST_USER_NAME']}@{os.environ['HOST_MACHINE_NAME']}"
        os.environ["LOGNAME"] = logging_user_name

        tags = {}
        tags.update(self.get_tracking_commit_hashes())
        tags.update(self.get_version_tags())
        tags.update(args.mlflow_tags)

        mlflow.set_tracking_uri(uri=args.mlflow_tracking_uri)
        exp = mlflow.set_experiment(args.mlflow_experiment_name)
        mlflow.start_run(
            run_name=args.run_name,
            experiment_id=exp.experiment_id,
            description=args.mlflow_run_description,
            log_system_metrics=True,
        )
        mlflow.set_tags(tags)
        os.environ["LOGNAME"] = logname

    def _finish_tracking(self) -> None:
        mlflow.end_run()

    def _get_sb3_logger(self, verbose: bool) -> Logger | None:
        loggers = [MLflowOutputFormat()]
        if verbose:
            loggers.append(HumanOutputFormat(sys.stdout))
        return Logger(folder=None, output_formats=loggers)

    def _log_params(self, params: Dict) -> None:
        mlflow.log_params(params)

    def _log_directory(self, local_dir: str, artifacts_dir: str = None) -> None:
        mlflow.log_artifacts(local_dir=local_dir, artifact_path=artifacts_dir)


class MLflowOutputFormat(KVWriter):
    """
    Dumps key/value pairs into MLflow's numeric format.
    """

    def write(
        self,
        key_values: Dict[str, Any],
        key_excluded: Dict[str, Union[str, Tuple[str, ...]]],
        step: int = 0,
    ) -> None:

        for (key, value), (_, excluded) in zip(sorted(key_values.items()), sorted(key_excluded.items())):

            if excluded is not None and "mlflow" in excluded:
                continue

            if isinstance(value, np.ScalarType):
                if not isinstance(value, str):
                    mlflow.log_metric(key, value, step)
