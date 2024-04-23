import argparse
import json
import sys
import os
import stable_baselines3 as sb3
import rl_zoo3
from stable_baselines3.common.logger import Logger, HumanOutputFormat
from rl_zoo3.mlflow import MLflowOutputFormat

INITALIZED = False
BACKEND = None
WANDB_RUN = None
MLFLOW = None

BACKEND_WANDB = "wandb"
BACKEND_MLFLOW = "mlflow"
BACKENDS_LIST = [BACKEND_WANDB, BACKEND_MLFLOW]


def argparse_add_track_arguments(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--track-backend",
        default=BACKEND_WANDB,
        choices=BACKENDS_LIST,
        help="select ML platform for tracking experiments",
    )
    _argparse_wandb(parser)
    _argparse_mlflow(parser)


def _argparse_wandb(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--wandb-entity", type=str, default=None, help="the entity (team) of wandb's project")
    parser.add_argument("--wandb-project-name", type=str, default="sb3", help="the wandb's project name")
    parser.add_argument(
        "-tags", "--wandb-tags", type=str, default=[], nargs="+", help="Tags for wandb run, e.g.: -tags optimized pr-123"
    )


def _argparse_mlflow(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--mlflow-experiment-name", type=str, default="sb3", help="the mlflow experiment name")
    parser.add_argument(
        "--mlflow-run-description", type=str, default="Generic run.", help="the description for the mlflow run"
    )
    parser.add_argument("--mlflow-tracking-uri", type=str, default="http://mlflow:5000", help="the uri of the mlflow server")

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


def argparse_filter_track_args(parsed_args):
    if parsed_args.track_backend == BACKEND_WANDB:
        del parsed_args.mlflow_experiment_name
        del parsed_args.mlflow_run_description
        del parsed_args.mlflow_tags
        del parsed_args.mlflow_tracking_uri
    elif parsed_args.track_backend == BACKEND_MLFLOW:
        del parsed_args.wandb_entity
        del parsed_args.wandb_project_name
        del parsed_args.wandb_entity
        del parsed_args.wandb_tags
    return parsed_args


def setup_tracking(args) -> None:
    global INITALIZED
    global BACKEND
    if INITALIZED:
        print(f"WARNING: Tried to reinitalize the tracking backend '{BACKEND}'! Ignornig.")
        return
    INITALIZED = True

    BACKEND = args.track_backend
    if BACKEND == BACKEND_WANDB:
        _setup_wandb(args)
    elif BACKEND == BACKEND_MLFLOW:
        _setup_mlflow(args)
    else:
        raise ImportError(f"if you want to track model data, please select one of the valid backends: '{BACKENDS_LIST}'")


def _setup_wandb(args) -> None:
    try:
        import wandb
    except ImportError as e:
        raise ImportError(
            "if you want to use Weights & Biases to track experiment, please install W&B via `pip install wandb`"
        ) from e

    assert "wandb_entity" in vars(args), "Missing W&B entity (--wandb-entity)."

    tags = [*args.wandb_tags, f"v{sb3.__version__}"]
    global WANDB_RUN
    WANDB_RUN = wandb.init(
        name=args.run_name,
        project=args.wandb_project_name,
        entity=args.wandb_entity,
        tags=tags,
        config=vars(args),
        sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
        monitor_gym=True,  # auto-upload the videos of agents playing the game
        save_code=True,  # optional
    )


def _setup_mlflow(args) -> None:
    try:
        import mlflow
    except ImportError as e:
        raise ImportError("if you want to use MLflow to track experiment, please install it via `pip install mlflow`") from e

    # Other functions need access to the mlflow import
    global MLFLOW
    MLFLOW = mlflow

    tags = {
        "SB3_version": f"v{sb3.__version__}",
        "rl_zoo3_version": f"v{rl_zoo3.__version__}",
        "docker_image_hash": os.environ["DOCKER_IMAGE_HASH"],
    }
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


def get_sb3_logger(verbose: bool = False) -> Logger:
    global INITALIZED
    if not INITALIZED:
        print(f"WARNING: Tried to obtain SB3 Logger without initalization of the tracking backend! Falling back to defaults.")
        return None

    loggers = []
    if verbose:
        loggers.append(HumanOutputFormat(sys.stdout))

    global BACKEND
    if BACKEND == BACKEND_MLFLOW:
        loggers.append(MLflowOutputFormat())

    return Logger(folder=None, output_formats=loggers)


def finish_tracking() -> None:
    global INITALIZED
    if not INITALIZED:
        return

    global BACKEND
    if BACKEND == BACKEND_MLFLOW:
        MLFLOW.end_run()


def log_artifacts_directory(local_dir: str, artifacts_dir: str = None) -> None:
    global INITALIZED
    if not INITALIZED:
        return

    global BACKEND
    if BACKEND == BACKEND_MLFLOW:
        MLFLOW.log_artifacts(local_dir=local_dir, artifact_path=artifacts_dir)


def log_params(args) -> None:
    global INITALIZED
    if not INITALIZED:
        return

    global BACKEND
    if BACKEND == BACKEND_WANDB:
        global WANDB_RUN
        assert WANDB_RUN is not None  # make mypy happy
        WANDB_RUN.config.setdefaults(vars(args))
    elif BACKEND == BACKEND_MLFLOW:
        MLFLOW.log_params(vars(args))
