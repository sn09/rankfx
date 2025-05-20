"""Module with DCNv2 model class implementation."""

from collections.abc import Callable, Sequence
from typing import Any, Literal

import torch
from sklearn.metrics import log_loss, roc_auc_score
from torch import nn

from rankfx.common.base.model import NNPandasModel
from rankfx.common.features.config import FeaturesConfig
from rankfx.common.modules import CrossNetMix, CrossNetV2, EmbeddingLayer, MLPBlock
from rankfx.common.utils.import_utils import import_module_by_path
from rankfx.dcnv2.config.model_config import ModelStructure


class DCNv2(NNPandasModel):
    """DCNv2 implementation."""
    def __init__(
        self,
        loss_fn: str = "torch.nn.BCEWithLogitsLoss",
        loss_params: dict[str, Any] | None = None,
        model_structure: Literal["crossnet_only", "stacked", "parallel", "stacked_parallel"] = "stacked",
        use_low_rank_mixture: bool = True,
        cross_low_rank_dim: int = 32,
        num_cross_layers: int = 4,
        num_cross_experts: int = 4,
        parallel_hidden_dims: Sequence[int] | None = None,
        parallel_dropout: float = 0.,
        parallel_use_batch_norm: bool = True,
        parallel_activation: Callable = nn.ReLU,
        stacked_hidden_dims: Sequence[int] | None = None,
        stacked_dropout: float = 0.,
        stacked_use_batch_norm: bool = True,
        stacked_activation: Callable = nn.ReLU,
        output_dim: int = 1,
        proj_output_embeddings: bool = False,
        features_config: FeaturesConfig | None = None,
        oov_idx: int = 0,
    ):
        """Instantiate DCNv2 module.

        Features are taken in the order, specified in features_config.features

        Args:
            loss_fn: loss function import path
            loss_params: model loss parameters
            model_structure: model structure type
            use_low_rank_mixture: use low rank projections inside the net
            cross_low_rank_dim: size of low rank cross net projections
            num_cross_layers: number of cross layers
            num_cross_experts: number of cross experts
            parallel_hidden_dims: hidden layers dimensions of parallel net
            parallel_dropout: dropout value of parallel net
            parallel_use_batch_norm: user batch norm in parallel net
            parallel_activation: activation function in parallel net
            stacked_hidden_dims: hidden layers dimensions of stacked net
            stacked_dropout: dropout value of stacked net
            stacked_use_batch_norm: user batch norm in stacked net
            stacked_activation: activation function in stacked net
            output_dim: net output dimensions
            proj_output_embeddings: apply linear layer to concatted embeddings
            features_config: features config if known beforehand
            oov_idx: index to use as OOV for all categorical features embeddings
        """
        super().__init__(
            infer_feature_config=features_config is None,
            oov_idx=oov_idx,
        )

        if model_structure not in [
            ModelStructure.CROSSNET_ONLY.value,
            ModelStructure.STACKED.value,
            ModelStructure.PARALLEL.value,
            ModelStructure.STACKED_PARALLEL.value,
        ]:
            raise RuntimeError(f"Unknown model structure `{model_structure}`")

        if (
            parallel_hidden_dims is None
            and model_structure in [ModelStructure.PARALLEL.value, ModelStructure.STACKED_PARALLEL.value]
        ):
            raise RuntimeError("No hidden sizes provided for Parallel DNN")

        if (
            stacked_hidden_dims is None
            and model_structure in [ModelStructure.STACKED.value, ModelStructure.STACKED_PARALLEL.value]
        ):
            raise RuntimeError("No hidden sizes provided for Stacked DNN")

        self.model_structure = model_structure
        self.proj_output_embeddings = proj_output_embeddings

        self.use_low_rank_mixture = use_low_rank_mixture
        self.cross_low_rank_dim = cross_low_rank_dim
        self.num_cross_layers = num_cross_layers
        self.num_cross_experts = num_cross_experts

        self.parallel_hidden_dims = parallel_hidden_dims
        self.parallel_dropout = parallel_dropout
        self.parallel_use_batch_norm = parallel_use_batch_norm
        self.parallel_activation = parallel_activation

        self.stacked_hidden_dims = stacked_hidden_dims
        self.stacked_dropout = stacked_dropout
        self.stacked_use_batch_norm = stacked_use_batch_norm
        self.stacked_activation = stacked_activation

        self.output_dim = output_dim

        loss_cls = import_module_by_path(loss_fn)
        self.loss_fn = loss_cls(**(loss_params or {}))

        if features_config is not None:
            self._init_modules(features_config=features_config)

    def _init_modules(self, features_config: FeaturesConfig) -> None:
        """Instantiate DCNv2 modules based on features config.

        Args:
            features_config: features config
        """
        self.embedding_layer = EmbeddingLayer(
            features_config=features_config,
            is_dict_input=True,
            proj_output_features=self.proj_output_embeddings
        )
        input_dim = features_config.num_final_features

        if self.use_low_rank_mixture:
            self.crossnet = CrossNetMix(
                input_dim,
                low_rank_dim=self.cross_low_rank_dim,
                num_layers=self.num_cross_layers,
                num_experts=self.num_cross_experts,
            )
        else:
            self.crossnet = CrossNetV2(input_dim, num_layers=self.num_cross_layers)

        if self.model_structure in (ModelStructure.STACKED.value, ModelStructure.STACKED_PARALLEL.value):
            self.stacked_dnn = MLPBlock(
                in_features=input_dim,
                hidden_dims=self.stacked_hidden_dims,
                activation_fn=self.stacked_activation,
                dropout=self.stacked_dropout,
                use_batch_norm=self.stacked_use_batch_norm,
            )

        if self.model_structure in (ModelStructure.PARALLEL.value, ModelStructure.STACKED_PARALLEL.value):
            self.parallel_dnn = MLPBlock(
                in_features=input_dim,
                hidden_dims=self.parallel_hidden_dims,
                activation_fn=self.parallel_activation,
                dropout=self.parallel_dropout,
                use_batch_norm=self.parallel_use_batch_norm,
            )

        final_dim = self._get_backbone_output_dim(
            input_dim=input_dim,
            parallel_hidden_dims=self.parallel_hidden_dims,
            stacked_hidden_dims=self.stacked_hidden_dims,
        )
        self.fc = nn.Linear(final_dim, self.output_dim)

    def _get_backbone_output_dim(
        self,
        input_dim: int | None = None,
        parallel_hidden_dims: Sequence[int] | None = None,
        stacked_hidden_dims: Sequence[int] | None = None,
    ) -> int:
        """Get output dim after CrossNet + DNN steps.

        Args:
            input_dim: number of input dimensions
            model_structure: model structure type
            parallel_hidden_dims: hidden layers dimensions of parallel net
            stacked_hidden_dims: hidden layers dimensions of stacked net

        Returns:
            Number of output dimensions after crossnet and dnn
        """
        if self.model_structure == ModelStructure.CROSSNET_ONLY.value:
            return input_dim
        if self.model_structure == ModelStructure.STACKED.value:
            return stacked_hidden_dims[-1]
        if self.model_structure == ModelStructure.PARALLEL.value:
            return input_dim + parallel_hidden_dims[-1]
        return stacked_hidden_dims[-1] + parallel_hidden_dims[-1]

    def forward(self, input_: dict[str, torch.Tensor] | torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            input_: input tensor [{feature_name: feature_tensor} or tensor (batch_size, concat_features_len)]

        Returns:
            Tensor after DCNv2 applying
        """
        embeddings_out = self.embedding_layer(input_)
        cross_out = self.crossnet(embeddings_out)
        if self.model_structure == ModelStructure.CROSSNET_ONLY.value:
            final_out = cross_out
        elif self.model_structure == ModelStructure.STACKED.value:
            final_out = self.stacked_dnn(cross_out)
        elif self.model_structure == ModelStructure.PARALLEL.value:
            final_out = torch.cat([cross_out, self.parallel_dnn(embeddings_out)], dim=-1)
        elif self.model_structure == ModelStructure.STACKED_PARALLEL.value:
            final_out = torch.cat([self.stacked_dnn(cross_out), self.parallel_dnn(embeddings_out)], dim=-1)
        else:
            raise RuntimeError(f"Unknown model structure: `{self.model_structure}`")

        return self.fc(final_out)

    def train_step(self, batch: dict[str, torch.Tensor] | torch.Tensor) -> dict[str, Any]:
        """Model training step.

        Args:
            batch: batch of data

        Returns:
            Loss/metrics after training step
        """
        target = batch["target"]
        logits = self.forward(batch).squeeze()
        loss = self.loss_fn(logits, target.float())

        return {"loss": loss}

    def val_step(self, batch: dict[str, torch.Tensor] | torch.Tensor) -> dict[str, Any]:
        """Model validation step.

        Args:
            batch: batch of data

        Returns:
            Loss/metrics after validation step
        """
        target = batch["target"]
        logits = self.forward(batch).squeeze()
        probs = torch.sigmoid(logits).detach().cpu().numpy()
        target = target.detach().cpu().numpy()

        roc_auc = roc_auc_score(target, probs)
        logloss = log_loss(target, probs)

        return {"AUC": roc_auc, "log_loss": logloss}

    def test_step(self, batch: dict[str, torch.Tensor] | torch.Tensor) -> dict[str, Any]:
        """Model testing step.

        Args:
            batch: batch of data

        Returns:
            Loss/metrics after testing step
        """
        target = batch["target"]
        logits = self.forward(batch).squeeze()
        probs = torch.sigmoid(logits).detach().cpu().numpy()
        target = target.detach().cpu().numpy()

        roc_auc = roc_auc_score(target, probs)
        logloss = log_loss(target, probs)

        return {"AUC": roc_auc, "log_loss": logloss}

    def inference_step(self, batch: dict[str, torch.Tensor] | torch.Tensor) -> Any:
        """Model inference step.

        Args:
            batch: batch of data

        Returns:
            Model inference output
        """
        logits = self.forward(batch).squeeze()
        return logits
