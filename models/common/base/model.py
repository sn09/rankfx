"""Module with base model implementation."""

import enum
import gc
import os
import pathlib
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any, Literal

import numpy as np
import pandas as pd
import torch
from pandas.api.types import is_float_dtype, is_list_like, is_numeric_dtype
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from common.features.config import Feature, FeaturesConfig, FeatureType
from common.features.datasets import PandasDataset
from common.utils.import_utils import import_module_by_path
from common.utils.logging_utils import get_logger
from common.utils.training_utils import seed_everything

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
    def __init__(self, infer_feature_config: bool = True):
        """Instantiate model class.

        Args:
            infer_feature_config: infer features config from input pandas dataframe
        """
        super().__init__()
        self.infer_feature_config = infer_feature_config

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

    @abstractmethod
    def _init_modules(self, features_config: FeaturesConfig) -> None:
        """Instantiate model modules based on features config.

        Args:
            features_config: features config
        """
        raise NotImplementedError

    def _init_optimizers(
        self,
        optimizer_cls: str | None = None,
        optimizer_params: dict[str, Any] | None = None,
        scheduler_cls: str | None = None,
        scheduler_params: dict[str, Any] | None = None,
    ) -> tuple[Optimizer | None, LRScheduler | None]:
        """Instatiate optimizer and scheduler objects if provided.

        Args:
            optimizer_cls: optimizer import path
            optimizer_params: optimizer parameters
            scheduler_cls: scheduler import path
            scheduler_params: scheduler parameters

        Returns:
            Tuple of optimizer and scheduler
        """
        optimizer = None
        if optimizer_cls:
            optimizer_cls = import_module_by_path(optimizer_cls)
            optimizer = optimizer_cls(self.parameters(), **(optimizer_params or {}))

        scheduler = None
        if scheduler_cls and optimizer is not None:
            scheduler_cls = import_module_by_path(scheduler_cls)
            scheduler = scheduler_cls(optimizer, **(scheduler_params or {}))

        return optimizer, scheduler

    def _calculate_uniq_categories(self, series: pd.Series, list_like: bool = False) -> int:
        """Calculate number of unique categories in column.

        Args:
            series: pandas series
            list_like: is series consists of sequences

        Returns:
            Number of unique categories
        """
        uniq_categories = series.nunique() if not list_like else series.explode().nunique()
        min_value = series.values.as_ordeted().min() if isinstance(series, pd.Categorical) else series.min()
        max_value = series.values.as_ordeted().max() if isinstance(series, pd.Categorical) else series.max()
        if min_value != 0 and max_value != uniq_categories - 1:
            LOGGER.warning(
                "Number of unique `%s` values is %s, but feature is in interval [%s, %s]",
                series.name,
                uniq_categories,
                min_value,
                max_value,
            )
        return uniq_categories

    def _infer_features_config_from_dataframe(
        self,
        data: pd.DataFrame,
        default_embedding_size: int = 10,
        custom_embedding_sizes: dict[str, int] | None = None,
        embedded_features: Sequence[str] | None = None,
    ) -> FeaturesConfig:
        """Create feature config from pandas dataframe.

        Args:
            data: pandas dataframe to infrence features config
            default_embedding_size: default features embedding size
            custom_embedding_sizes: custom embeddings mapping {feature: feature: embedding_size}
            embedded_features: features to embed

        Returns:
            Created features config
        """
        custom_embedding_sizes = custom_embedding_sizes or {}

        features = []
        for col in data.columns:
            num_embed_params = {}
            if col in embedded_features:
                num_embed_params = {
                    "needs_embed": True,
                    "embedding_size": custom_embedding_sizes.get(col, default_embedding_size),
                }

            if isinstance(data[col].dtype, pd.CategoricalDtype):
                uniq_categories = self._calculate_uniq_categories(data[col])
                features.append(
                    Feature(
                        name=col,
                        feature_type=FeatureType.CATEGORICAL,
                        needs_embed=True,
                        embedding_size=custom_embedding_sizes.get(col, default_embedding_size),
                        embedding_vocab_size=uniq_categories,
                    )
                )
                continue

            if is_numeric_dtype(data[col].dtype):
                features.append(Feature(name=col, feature_type=FeatureType.NUMERICAL, **num_embed_params))
                continue

            list_value = data.at[0, col]
            if not is_list_like(list_value):
                raise RuntimeError(
                    f"Feature `{col}` is not category, not numerical and not list-like, got {list_value}"
                )

            feature_size = data[col].apply(len).unique()
            if len(feature_size) > 1:
                raise RuntimeError(f"Got multiple sequence lengths for feature `{col}`")
            feature_size = feature_size[0]

            if is_float_dtype(np.concat(data[col].values)):
                features.append(
                    Feature(
                        name=col,
                        feature_type=FeatureType.NUMERICAL_SEQUENCE,
                        feature_size=feature_size,
                        **num_embed_params,
                    )
                )
                continue

            uniq_categories = self._calculate_uniq_categories(data[col], list_like=True)
            features.append(
                Feature(
                    name=col,
                    feature_type=FeatureType.CATEGORICAL_SEQUENCE,
                    feature_size=feature_size,
                    needs_embed=True,
                    embedding_size=custom_embedding_sizes.get(col, default_embedding_size),
                    embedding_vocab_size=uniq_categories,
                )
            )

        return FeaturesConfig(features=features)

    def _get_dataloader_from_dataframes(
        self,
        features: pd.DataFrame,
        target: pd.Series | np.ndarray | None = None,
        shuffle: bool = False,
        batch_size: int = 64,
        num_workers: int = 12,
        drop_last_batch: bool = False,
    ) -> DataLoader:
        """Create dataloaders from input pandas dataframes.

        Args:
            features: dataframe with input features
            target: dataframe with train target
            shuffle: shuffle elements flag
            batch_size: size of dataloader batches
            num_workers: number of dataloader workers
            drop_last_batch: drop last incomplete batch
        Returns:
            Created dataloader
        """
        merged_df = features.copy(deep=True)
        target_col = None
        if target is not None:
            merged_df["target"] = target
            target_col = "target"

        dataset = PandasDataset(merged_df, return_dicts=True, target_col=target_col)
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=drop_last_batch,
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

        if isinstance(batch, dict):
            for key in batch:
                batch[key] = batch[key].to(device)
        else:
            for idx in range(len(batch)):
                batch[idx] = batch[idx].to(device)
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
        """Run model step depending on current phase.

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
        device: torch.device | str,
    ) -> torch.Tensor:
        """Run single inference epoch.

        Args:
            dataloader: torch dataloader
            device: working device

        Returns:
            Output tensors after model inference
        """
        self.eval()

        batch_results = []

        progress_bar = tqdm(dataloader, desc="Inference epoch")
        for batch in progress_bar:
            batch = self._move_batch_to_device(batch, device=device)
            with torch.inference_mode():
                step_output = self._run_model_step(batch, phase=ModelPhase.INFERENCE)
            batch_results.append(step_output)
        return torch.cat(batch_results, dim=0)

    def _run_epoch(
        self,
        dataloader: DataLoader,
        epoch_num: int,
        optimizer: Optimizer | None = None,
        phase: ModelPhase = ModelPhase.TRAIN,
        scheduler: LRScheduler | None = None,
        device: torch.device | str = "cpu",
        grad_clip_threshold: float | None = None,
    ) -> dict[str, Any]:
        """Run single epoch.

        Args:
            dataloader: torch dataloader
            epoch_num: number of epoch
            optimizer: model optimizer
            device: working device
            phase: model phase
            scheduler: learning rate scheduler
            grad_clip_threshold: value to clip gradients

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
            batch = self._move_batch_to_device(batch, device=device)
            steps_per_epoch += 1

            if phase == ModelPhase.TRAIN:
                optimizer.zero_grad()
                step_output = self._run_model_step(batch, phase=phase)
                loss = step_output["loss"]
                loss.backward()

                if grad_clip_threshold:
                    torch.nn.utils.clip_grad_norm_(self.parameters(), grad_clip_threshold)

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
        target: pd.Series | np.ndarray = None,
        val_features: pd.DataFrame | None = None,
        val_target: pd.Series | np.ndarray = None,
        optimizer_cls: str | None = None,
        optimizer_params: dict[str, Any] | None = None,
        scheduler_cls: str | None = None,
        scheduler_params: dict[str, Any] | None = None,
        num_epochs: int = 10,
        seed: int | None = None,
        artifacts_path: str | None = None,
        device: str | torch.device = "cpu",
        batch_size: int = 64,
        num_workers: int = 12,
        drop_last_batch: bool = False,
        grad_clip_threshold: float | None = None,
        validate_every_n_epochs: int = 1,
        eval_metric_name: str | None = None,
        eval_mode: Literal["min", "max"] = "max",
        patience: int | None = None,
        default_embedding_size: int = 10,
        custom_embedding_sizes: dict[str, int] | None = None,
        embedded_features: Sequence[str] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any] | None]:
        """Train model using pandas dataframes.

        Args:
            features: features to train
            target: target to train
            val_features: features for validation, if None dont validate
            val_target: target for validation
            optimizer_cls: optimizer import path
            optimizer_params: optimizer parameters
            scheduler_cls: scheduler import path
            scheduler_params: scheduler parameters
            num_epochs: number of training epochs
            seed: random seed
            artifacts_path: path to save artifacts
            device: working device
            batch_size: size of dataloader batches
            num_workers: number of dataloader workers
            drop_last_batch: drop last incomplete batch
            grad_clip_threshold: value to clip gradients
            validate_every_n_epochs: evaulation frequency
            eval_metric_name: metric name to evaluate
            eval_mode: eval rule for metrics
            patience: number of epochs to wait for metrics improvement, after that stop training
            default_embedding_size: default features embedding size
            custom_embedding_sizes: custom embeddings mapping {feature: feature: embedding_size}
            embedded_features: features to embed

        Returns:
            Output metrics from train and validation steps
        """
        if self.infer_feature_config:
            features_config = self._infer_features_config_from_dataframe(
                data=features,
                default_embedding_size=default_embedding_size,
                custom_embedding_sizes=custom_embedding_sizes,
                embedded_features=embedded_features,
            )
            self._init_modules(features_config=features_config)
        LOGGER.info("Used features config: %s", features_config)

        optimizer, scheduler = self._init_optimizers(
            optimizer_cls=optimizer_cls,
            optimizer_params=optimizer_params,
            scheduler_cls=scheduler_cls,
            scheduler_params=scheduler_params,
        )

        eval_metric_name = eval_metric_name or "loss"
        if seed:
            seed_everything(seed)

        best_model_path = None
        if artifacts_path:
            LOGGER.info("Artifacts path is %s", pathlib.Path(artifacts_path).resolve())
            if not os.path.isdir(artifacts_path):
                LOGGER.info("Creating artifacts path")
                os.makedirs(artifacts_path)

            best_model_path = str(pathlib.Path(artifacts_path) / "best_model.pt")
            LOGGER.info("Best model path is %s", best_model_path)

        self = self.to(device)
        optimizer = optimizer or torch.optim.Adam(self.parameters())

        train_dataloader = self._get_dataloader_from_dataframes(
            features=features,
            target=target,
            shuffle=True,
            batch_size=batch_size,
            num_workers=num_workers,
            drop_last_batch=drop_last_batch,
        )

        val_dataloader = None
        if val_features is not None:
            val_dataloader = self._get_dataloader_from_dataframes(
                features=val_features,
                target=val_target,
                shuffle=False,
                batch_size=batch_size,
                num_workers=num_workers,
                drop_last_batch=False,
            )

        LOGGER.info("Starting training process")
        train_output, val_output = None, None
        patience_epochs, best_eval_metric = 0, None

        for epoch in range(num_epochs):
            train_output = self._run_epoch(
                train_dataloader,
                epoch_num=epoch,
                phase=ModelPhase.TRAIN,
                optimizer=optimizer,
                scheduler=scheduler,
                device=device,
                grad_clip_threshold=grad_clip_threshold,
            )

            if (
                val_dataloader is not None
                and (not epoch % validate_every_n_epochs or epoch == num_epochs - 1)
            ):
                val_output = self._run_epoch(
                    val_dataloader,
                    epoch_num=epoch,
                    phase=ModelPhase.VALIDATION,
                    device=device,
                )

                current_eval_metric = val_output[eval_metric_name]
                if self._check_metrics_improvement(current_eval_metric, best_eval_metric, eval_mode=eval_mode):
                    patience_epochs = 0
                    best_eval_metric = current_eval_metric

                    if best_model_path:
                        torch.save(self.state_dict(), best_model_path)
                        LOGGER.info(
                            "Best model with %s = %s was saved to %s",
                            eval_metric_name,
                            best_eval_metric,
                            best_model_path,
                        )
                    continue

                patience_epochs += 1
                if patience and patience_epochs > patience:
                    LOGGER.info("Metrics not increasing during %s epochs. Stop training", patience)
                    break

        return train_output, val_output

    def test(
        self,
        features: pd.DataFrame,
        target: pd.Series | np.ndarray = None,
        device: str | torch.device = "cpu",
        batch_size: int = 64,
        num_workers: int = 12,
    ) -> dict[str, Any]:
        """Run prediction on pandas dataframe.

        Args:
            features: features for test
            target: target to for test
            device: working device
            batch_size: size of dataloader batches
            num_workers: number of dataloader workers

        Returns:
            Test metrics
        """
        self = self.to(device)
        dataloader = self._get_dataloader_from_dataframes(
            features=features,
            target=target,
            shuffle=False,
            batch_size=batch_size,
            num_workers=num_workers,
            drop_last_batch=False,
        )
        test_output = self._run_epoch(
            dataloader,
            epoch_num=-1,
            phase=ModelPhase.TEST,
            device=device,
        )
        return test_output

    def predict(
        self,
        features: pd.DataFrame,
        device: str | torch.device = "cpu",
        batch_size: int = 64,
        num_workers: int = 12,
    ) -> np.ndarray:
        """Run prediction on pandas dataframe.

        Args:
            features: features to train
            device: working device
            batch_size: size of dataloader batches
            num_workers: number of dataloader workers

        Returns:
            Predicted items
        """
        self = self.to(device)
        dataloader = self._get_dataloader_from_dataframes(
            features=features,
            target=None,
            shuffle=False,
            batch_size=batch_size,
            num_workers=num_workers,
            drop_last_batch=False,
        )
        inference_output = self._run_inference_epoch(dataloader, device=device)
        return inference_output.detach().cpu().numpy()
