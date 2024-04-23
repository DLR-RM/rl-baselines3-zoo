import mlflow
import os
from typing import Optional
from stable_baselines3.common.callbacks import BaseCallback  # type: ignore


class MLflowCallback(BaseCallback):
    """Callback for logging experiments to MLflow.

    Log SB3 experiments to MLflow
        - Note that mlflow import must be initalized before the MLflowCallback can be used.

    Args:
        verbose: The verbosity of sb3 output
        model_save_path: Path to the folder where the model will be saved, The default value is `None` so the model is not logged
        model_save_freq: Frequency to save the model
    """

    def __init__(self, verbose: int = 0, model_save_path: Optional[str] = None, model_save_freq: int = 0) -> None:
        super().__init__(verbose)
        if mlflow.active_run() is None:
            raise RuntimeError("You must call initalize mlflow run before MLflowCallback()")

        self.run_id = mlflow.active_run()
        self.model_save_freq = model_save_freq
        self.model_save_path = model_save_path

        # Create folder if needed
        if self.model_save_path is not None:
            os.makedirs(self.model_save_path, exist_ok=True)
            self.path = os.path.join(self.model_save_path, "model.zip")
        else:
            assert self.model_save_freq == 0, "to use the `model_save_freq` you have to set the `model_save_path` parameter"

    def _init_callback(self) -> None:
        pass

    def _on_step(self) -> bool:
        if self.model_save_freq > 0:
            if self.model_save_path is not None:
                if self.n_calls % self.model_save_freq == 0:
                    self.save_model()
        return True

    def _on_training_end(self) -> None:
        if self.model_save_path is not None:
            self.save_model()

    def save_model(self) -> None:
        self.model.save(self.path)
        mlflow.log_artifact(local_path=self.path, run_id=self.run_id)
        if self.verbose > 1:
            print(f"Saving model checkpoint to {self.path}")
