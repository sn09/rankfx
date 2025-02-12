"""Module with base model implementation."""

import enum
import gc
from abc import ABC, abstractmethod
from typing import Any, Literal

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from models.common.base.config.training_config import TrainingConfig
from models.common.features.datasets import PandasDataset
from models.common.utils.logging_utils import get_logger
from models.common.utils.training_utils import seed_everything

LOGGER = get_logger(__name__)


@enum.unique
class ModelPhase(enum.Enum):
    """Possible model usage phases."""
    TRAIN = "Train"
    VALIDATION = "Validation"
    TEST = "Test"
    INFERENCE = "Inference"


class NNPandasModel(ABC, nn.Module):
    """Base model class."""
    def __init__(self):
        """Instantiate model class."""
        super().__init__()

    @abstractmethod
    def forward(self, input_: dict[str, torch.Tensor] | torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            input_: input tensor [{feature_name: feature_tensor} or tensor (batch_size, concat_features_len)]

        Returns:
            Tensor after model applying
        """
        raise NotImplementedError

    @abstractmethod
    def train_step(self, batch: dict[str, torch.Tensor] | torch.Tensor) -> dict[str, Any]:
        """Model training step.

        Output MUST have `loss` key

        Args:
            batch: batch of data

        Returns:
            Loss/metrics after training step
        """
        raise NotImplementedError

    @abstractmethod
    def val_step(self, batch: dict[str, torch.Tensor] | torch.Tensor) -> dict[str, Any]:
        """Model validation step.

        Args:
            batch: batch of data

        Returns:
            Loss/metrics after validation step
        """
        raise NotImplementedError

    @abstractmethod
    def test_step(self, batch: dict[str, torch.Tensor] | torch.Tensor) -> dict[str, Any]:
        """Model testing step.

        Args:
            batch: batch of data

        Returns:
            Loss/metrics after testing step
        """
        raise NotImplementedError

    @abstractmethod
    def inference_step(self, batch: dict[str, torch.Tensor] | torch.Tensor) -> Any:
        """Model inference step.

        Args:
            batch: batch of data

        Returns:
            Model inference output
        """
        raise NotImplementedError

    def _get_dataloader_from_dataframes(
        self,
        config: TrainingConfig,
        features: pd.DataFrame,
        target: pd.Series | np.ndarray | None = None,
        shuffle: bool = False,
    ) -> DataLoader:
        """Create dataloaders from input pandas dataframes.

        Args:
            config: config instance
            features: dataframe with input features
            target: dataframe with train target
            shuffle: shuffle elements flag

        Returns:
            Created dataloader
        """
        merged_df = features.copy(deep=True)
        target_col = None
        if target is not None:
            merged_df["target"] = target
            target_col = "target"

        dataset = PandasDataset(merged_df, return_dicts=config.return_dict_batches, target_col=target_col)
        return DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=shuffle,
            num_workers=config.num_workers,
            pin_memory=True,
            drop_last=config.drop_last_batch,
        )

    def _move_batch_to_device(
        self,
        batch: dict[str, torch.Tensor] | torch.Tensor,
        device: torch.device,
    ) -> dict[str, torch.Tensor] | torch.Tensor:
        """Move whole batch to device.

        Args:
            batch: input batch of data
            device: device to put batch on

        Returns:
            Moved batch
        """
        if isinstance(batch, torch.Tensor):
            return batch.to(device)
        for key in batch:
            batch[key] = batch[key].to(device)
        return batch

    def _clear_resources(self) -> None:
        """Clear temporary resources."""
        gc.collect()
        torch.cuda.empty_cache()

    def _check_metrics_improvement(
        self,
        new_value: float,
        prev_value: float | None,
        eval_mode: Literal["min", "max"] = "max",
    ) -> bool:
        """Check if metrics are improving based on eval mode.

        Args:
            new_value: new metric value
            prev_value: previous metric value
            eval_mode: eval rule for metrics

        Returns:
            True if metric has improved
        """
        if prev_value is None:
            return True

        if eval_mode == "min":
            return new_value <= prev_value

        if eval_mode == "max":
            return new_value >= prev_value

        raise RuntimeError(f"Got unknown eval_mode {eval_mode}, should be on of (`min`, `max`)")

    def _run_model_step(
        self,
        batch: dict[str, torch.Tensor] | torch.Tensor,
        phase: ModelPhase,
    ) -> Any:
        """Run model step depending on current phae.

        Args:
            batch: batch of data
            phase: model phase

        Returns:
            Loss/metrics after step
        """
        if phase == ModelPhase.TRAIN:
            return self.train_step(batch)

        if phase == ModelPhase.VALIDATION:
            return self.val_step(batch)

        if phase == ModelPhase.TEST:
            return self.test_step(batch)

        return self.inference_step(batch)

    def _run_inference_epoch(
        self,
        dataloader: DataLoader,
        config: TrainingConfig,
    ) -> torch.Tensor:
        """Run single inference epoch.

        Args:
            dataloader: torch dataloader
            config: config instance

        Returns:
            Output tensors after model inference
        """
        self.eval()

        batch_results = []

        progress_bar = tqdm(dataloader, desc="Inference epoch")
        for batch in progress_bar:
            batch = self._move_batch_to_device(batch, device=config.device)
            with torch.inference_mode():
                step_output = self._run_model_step(batch, phase=ModelPhase.INFERENCE)
            batch_results.append(step_output)
        return torch.cat(batch_results, dim=0)

    def _run_epoch(
        self,
        dataloader: DataLoader,
        epoch_num: int,
        optimizer: Optimizer,
        config: TrainingConfig,
        phase: ModelPhase = ModelPhase.TRAIN,
        scheduler: LRScheduler | None = None,
    ) -> dict[str, Any]:
        """Run single epoch.

        Args:
            dataloader: torch dataloader
            epoch_num: number of epoch
            optimizer: model optimizer
            config: training config instance
            phase: model phase
            scheduler: learning rate scheduler

        Returns:
            Average metrics after training epoch
        """
        if phase == ModelPhase.TRAIN:
            self.train()
            optimizer.zero_grad()
        else:
            self.eval()

        steps_per_epoch = 0
        epoch_metrics = {}

        progress_bar = tqdm(dataloader, desc=f"{phase.value} epoch #{epoch_num}")
        for batch in progress_bar:
            batch = self._move_batch_to_device(batch, device=config.device)
            steps_per_epoch += 1

            if phase == ModelPhase.TRAIN:
                optimizer.zero_grad()
                step_output = self._run_model_step(batch, phase=phase)
                loss = step_output["loss"]
                loss.backward()

                if config.grad_clip_threshold:
                    torch.nn.utils.clip_grad_norm_(self.parameters(), config.grad_clip_threshold)

                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
            else:
                with torch.inference_mode():
                    step_output = self._run_model_step(batch, phase=phase)

            for key, value in step_output.items():
                epoch_metrics[key] = epoch_metrics.get(key, 0) + value

            epoch_metrics_str = ", ".join(
                [f"{key}: {value / steps_per_epoch:.5f}" for key, value in epoch_metrics.items()]
            )
            # update progress bar
            progress_bar.set_postfix(metrics=f"[{epoch_metrics_str}]")

        epoch_metrics = {key: value / steps_per_epoch for key, value in epoch_metrics.items()}
        epoch_metrics_str = ", ".join([f"{key}: {value:.5f}" for key, value in epoch_metrics.items()])
        LOGGER.info("Finished %s Epoch #%s, average metrics - [%s]", phase.value, epoch_num, epoch_metrics_str)

        self._clear_resources()
        return epoch_metrics

    def fit(
        self,
        features: pd.DataFrame,
        config: TrainingConfig,
        target: pd.Series | np.ndarray = None,
        optimizer: Optimizer | None = None,
        scheduler: LRScheduler | None = None,
        val_features: pd.DataFrame | None = None,
        val_target: pd.Series | np.ndarray = None,
    ) -> tuple[dict[str, Any], dict[str, Any] | None]:
        """Train model using pandas dataframes.

        Args:
            features: features to train
            target: target to train
            config: training config instance
            optimizer: model optimizer
            scheduler: model scheduler
            val_features: features for validation, if None dont validate
            val_target: target for validation

        Returns:
            Output metrics from train and validation steps
        """
        seed_everything(config.seed)

        self = self.to(config.device)
        optimizer = optimizer or torch.optim.Adam(self.parameters())

        train_dataloader = self._get_dataloader_from_dataframes(
            config=config,
            features=features,
            target=target,
            shuffle=True,
        )

        val_dataloader = None
        if val_features is not None:
            val_dataloader = self._get_dataloader_from_dataframes(
                config=config,
                features=val_features,
                target=val_target,
                shuffle=False,
            )

        LOGGER.info("Starting training process")
        train_output, val_output = None, None
        patience_epochs, best_eval_metric = 0, None

        for epoch in range(config.num_epochs):
            train_output = self._run_epoch(
                train_dataloader,
                epoch_num=epoch,
                phase=ModelPhase.TRAIN,
                optimizer=optimizer,
                scheduler=scheduler,
                config=config,
            )

            if (
                val_dataloader is not None
                and (not epoch % config.validate_every_n_epochs or epoch == config.num_epochs - 1)
            ):
                val_output = self._run_epoch(
                    val_dataloader,
                    epoch_num=epoch,
                    phase=ModelPhase.VALIDATION,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    config=config,
                )

                current_eval_metric = val_output[config.eval_metric_name]
                if self._check_metrics_improvement(current_eval_metric, best_eval_metric, eval_mode=config.eval_mode):
                    patience_epochs = 0
                    best_eval_metric = current_eval_metric

                    torch.save(self.state_dict(), config.best_model_path)
                    LOGGER.info(
                        "Best model with %s = %s was saved to %s",
                        config.eval_metric_name,
                        best_eval_metric,
                        config.best_model_path,
                    )
                    continue

                patience_epochs += 1
                if patience_epochs > config.patience:
                    LOGGER.info("Metrics not increasing during %s epochs. Stop training", config.patience)
                    break

        return train_output, val_output

    def test(
        self,
        features: pd.DataFrame,
        config: TrainingConfig,
        target: pd.Series | np.ndarray = None,
    ) -> dict[str, Any]:
        """Run prediction on pandas dataframe.

        Args:
            features: features for test
            config: config instance
            target: target to for test

        Returns:
            Test metrics
        """
        self = self.to(config.device)
        dataloader = self._get_dataloader_from_dataframes(
            config=config,
            features=features,
            target=target,
            shuffle=False,
        )
        test_output = self._run_epoch(
            dataloader,
            epoch_num=-1,
            phase=ModelPhase.TEST,
            optimizer=None,
            config=config,
        )
        return test_output

    def predict(
        self,
        features: pd.DataFrame,
        config: TrainingConfig,
    ) -> np.ndarray:
        """Run prediction on pandas dataframe.

        Args:
            features: features to train
            config: config instance

        Returns:
            Predicted items
        """
        self = self.to(config.device)
        dataloader = self._get_dataloader_from_dataframes(
            config=config,
            features=features,
            target=None,
            shuffle=False,
        )
        inference_output = self._run_inference_epoch(
            dataloader,
            epoch_num=-1,
            phase=ModelPhase.INFERENCE,
            optimizer=None,
            config=config,
        )
        return inference_output.detach().cpu().numpy()
