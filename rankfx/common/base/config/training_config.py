"""Module with basic trainer config implementation."""

from __future__ import annotations

import pathlib
from dataclasses import field
from typing import Literal

from pydantic.dataclasses import dataclass

from rankfx.common.base.config.base_config import BaseConfig


@dataclass
class TrainingConfig(BaseConfig):
    """Basic trainer config."""
    # data params
    batch_size: int = field(default=64)
    num_workers: int = field(default=12)
    return_dict_batches: bool = field(default=False)
    drop_last_batch: bool = field(default=False)

    # training params
    num_epochs: int = field(default=10)

    grad_clip_threshold: float = field(default=10.)

    seed: int = field(default=42)
    device: str = field(default="cpu")

    artifacts_path: pathlib.Path | str = field(default="./artifacts")
    checkpoint_filename: str = field(default="checkpoint.pth")
    best_model_filename: str = field(default="best_model.pth")

    # validation params
    validate_every_n_epochs: int = field(default=1)
    patience: int = field(default=5)
    eval_metric_name: str | None = field(default=None)
    eval_mode: Literal["min", "max"] = field(default="max")

    def checkpoint_path(self, **kwargs) -> pathlib.Path:
        """Get path to save checkpoint.

        Args:
            kwargs: will be used in checkpoint filename

        Returns:
            Checkpoint save path
        """
        checkpoint_filename = self.checkpoint_filename
        if kwargs:
            params_str = "_".join([f"{key}={val}" for key, val in kwargs.items()])
            checkpoint_filename = f"{params_str}_{checkpoint_filename}"
        return pathlib.Path(self.artifacts_path).resolve() / checkpoint_filename

    @property
    def best_model_path(self) -> pathlib.Path:
        """Get path to save best model.

        Returns:
            Best model save path
        """
        return pathlib.Path(self.artifacts_path).resolve() / self.best_model_filename
