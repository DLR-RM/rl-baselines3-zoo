import stable_baselines3 as sb3

INITALIZED = False
BACKEND = None
WANDB_RUN = None
MLFLOW = None

BACKEND_WANDB = "wandb"
BACKEND_MLFLOW = "mlflow"


def setup_tracking(args) -> None:
    global INITALIZED
    global BACKEND
    if INITALIZED:
        print(f"WARNING: Tried to reinitalize the tracking backend '{BACKEND}'! Ignornig.")
        return

    BACKEND = args.track_backend
    if BACKEND == BACKEND_MLFLOW:
        try:
            import mlflow
        except ImportError as e:
            raise ImportError(
                "if you want to use MLflow to track experiment, please install it via `pip install mlflow`"
            ) from e

        # Other functions need access to the mlflow import
        global MLFLOW
        MLFLOW = mlflow

        tags = {"SB3_version": f"v{sb3.__version__}"}
        tags.update(args.mlflow_tags)

        mlflow.set_tracking_uri(uri=args.mlflow_tracking_uri)
        mlflow.set_experiment(args.mlflow_experiment_name)
        # TODO: Fill start_run with params
        mlflow.start_run()
        mlflow.set_tags(tags)

    elif BACKEND == BACKEND_WANDB:
        try:
            import wandb
        except ImportError as e:
            raise ImportError(
                "if you want to use Weights & Biases to track experiment, please install W&B via `pip install wandb`"
            ) from e

        assert (args.wandb_entity is not None, "Missing W&B entity (--wandb-entity).")

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
    else:
        raise ImportError(
            f"if you want to track model data, please install select valid backend: '{BACKEND_WANDB}' or '{BACKEND_MLFLOW}'"
        )


def finish_tracking():
    global INITALIZED
    if not INITALIZED:
        return

    global BACKEND
    if BACKEND == BACKEND_MLFLOW:
        MLFLOW.end_run()


def log_params(args):
    global INITALIZED
    if not INITALIZED:
        return

    global BACKEND
    if BACKEND == BACKEND_WANDB:
        global WANDB_RUN
        assert WANDB_RUN is not None  # make mypy happy
        WANDB_RUN.config.setdefaults(vars(args))
    if BACKEND == BACKEND_MLFLOW:
        MLFLOW.log_params(args)
