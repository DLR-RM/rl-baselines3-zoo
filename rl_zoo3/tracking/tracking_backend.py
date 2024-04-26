import os
import argparse
from typing import Dict, Any, Set

from abc import ABC, abstractmethod
from rl_zoo3.tracking.git import track_git_repos

import rl_zoo3
import sb3_contrib
import stable_baselines3 as sb3
from stable_baselines3.common.logger import Logger


class TrackingBackend(ABC):
    implemented_backends: Dict[str, "TrackingBackend"] = {}
    tracking_commit_hashes = {f"version/{k}": v for k, v in track_git_repos().items()}
    version_tags = {
        "version/SB3": f"v{sb3.__version__}",
        "version/sb3_contrib": f"v{sb3_contrib.__version__}",
        "version/rl_zoo3": f"v{rl_zoo3.__version__}",
        "version/docker_image_hash": os.environ["DOCKER_IMAGE_HASH"],
    }

    def __init__(self):
        self.initalized = False

    def __init_subclass__(cls, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        short_name = cls.__name__[: -len("Backend")].lower()
        TrackingBackend.implemented_backends[short_name] = cls

    @classmethod
    def get_tracker(
        cls,
        backend: str,
    ) -> "TrackingBackend":
        bd_str = backend.lower()
        try:
            return TrackingBackend.implemented_backends[bd_str]()
        except KeyError:
            info = ""
            if bd_str == "wandb":
                info = "If you want to use Weights & Biases to track experiment, please install W&B via `pip install wandb`."
            elif bd_str == "mlflow":
                info = "If you want to use MLflow to track experiment, please install it via `pip install mlflow`."
            raise KeyError(f"There is no registered backend '{bd_str}'. {info}")

    @classmethod
    def list_backends(cls) -> Set:
        return set([k for k in TrackingBackend.implemented_backends.keys()])

    @classmethod
    @abstractmethod
    def argparse_add_arguments(cls, parser: argparse.ArgumentParser) -> None:
        raise NotImplementedError()

    @classmethod
    @abstractmethod
    def argparse_del_arguments(cls, parsed_args) -> None:
        raise NotImplementedError()

    def is_initalized(self) -> bool:
        return self.initalized

    def get_tracking_commit_hashes(self) -> Dict[str, str]:
        return TrackingBackend.tracking_commit_hashes

    def get_version_tags(self) -> Dict[str, str]:
        return TrackingBackend.version_tags

    def setup_tracking(self, args) -> None:
        if self.is_initalized():
            return
        self._setup_tracking(args=args)
        self.initalized = True

    def finish_tracking(self) -> None:
        if not self.is_initalized():
            return
        self._finish_tracking()
        self.initalized = False

    def get_sb3_logger(self, verbose: bool) -> Logger | None:
        if not self.is_initalized():
            print(
                f"WARNING: Tried to obtain SB3 Logger without initalization of the tracking backend! Falling back to defaults."
            )
            return None
        return self._get_sb3_logger(verbose=verbose)

    def log_params(self, params: Dict | Any) -> None:
        if not self.is_initalized():
            return

        params_dict = params
        if not isinstance(params, dict):
            params_dict = vars(params)

        self._log_params(params=params_dict)

    def log_directory(self, local_dir: str, artifacts_dir: str = None) -> None:
        if not self.is_initalized():
            return
        self._log_directory(local_dir=local_dir, artifacts_dir=artifacts_dir)

    @abstractmethod
    def _setup_tracking(self, args) -> None:
        raise NotImplementedError()

    @abstractmethod
    def _finish_tracking(self) -> None:
        raise NotImplementedError()

    @abstractmethod
    def _get_sb3_logger(self, verbose: bool) -> Logger | None:
        raise NotImplementedError()

    @abstractmethod
    def _log_params(self, params: Dict) -> None:
        raise NotImplementedError()

    @abstractmethod
    def _log_directory(self, local_dir: str, artifacts_dir: str = None) -> None:
        raise NotImplementedError()
