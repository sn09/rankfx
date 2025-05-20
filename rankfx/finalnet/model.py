"""Module with FinalNet model class implementation."""

from collections.abc import Callable, Sequence
from typing import Any, Literal

import torch
from sklearn.metrics import log_loss, roc_auc_score
from torch import nn

from rankfx.common.base.model import NNPandasModel
from rankfx.common.features.config import FeaturesConfig
from rankfx.common.modules import EmbeddingLayer, FieldGate, FinalBlock
from rankfx.common.utils.import_utils import import_module_by_path


class FinalNet(NNPandasModel):
    """FinalNet implementation."""
    def __init__(
        self,
        loss_fn: str = "torch.nn.BCEWithLogitsLoss",
        loss_params: dict[str, Any] | None = None,
        block_type: Literal["1B", "2B"] = "2B",
        use_field_gate: bool = True,
        use_batch_norm: bool = True,
        add_bias: bool = True,
        block1_hidden_dims: Sequence[int] = (64, 64, 64),
        block1_hidden_activations: Sequence[Callable] | Callable | None = None,
        block1_dropout_rates: Sequence[float] | float | None = None,
        block2_hidden_dims: Sequence[int] = (64, 64, 64),
        block2_hidden_activations: Sequence[Callable] | Callable | None = None,
        block2_dropout_rates: Sequence[float] | float | None = None,
        residual_type: Literal["sum", "concat"] = "concat",
        proj_output_embeddings: bool = False,
        features_config: FeaturesConfig | None = None,
        oov_idx: int = 0,
    ):
        """Instantiate FinalNet module.

        Features are taken in the order, specified in features_config.features

        Args:
            loss_fn: loss function import path
            loss_params: model loss parameters
            block_type: type of blocks for FinalNet model
            use_field_gate: use FieldGate layer for the first block
            use_batch_norm: use batch norm for FinalBlock hidden layers
            add_bias: add bias to FactorizedInteraction layer
            block1_hidden_dims: hidden layers dimensions for block 1
            block1_hidden_activations: hidden layers activations for block 1
            block1_dropout_rates: hidden layers dropout rates for block 1
            block2_hidden_dims: hidden layers dimensions for block 2
            block2_hidden_activations: hidden layers activations for block 2
            block2_dropout_rates: hidden layers dropout rates for block 2
            residual_type: type of residual for FactorizedInteraction, one of `sum` and `concat`
            proj_output_embeddings: apply linear layer to concatted embeddings
            features_config: features config if known beforehand
            oov_idx: index to use as OOV for all categorical features embeddings
        """
        super().__init__(
            infer_feature_config=features_config is None,
            oov_idx=oov_idx,
        )

        if block_type not in ["1B", "2B"]:
            raise RuntimeError(f"block_type should be on of [`1B`, `2B`], got `{block_type}`")

        self.use_field_gate = use_field_gate
        self.use_batch_norm = use_batch_norm
        self.add_bias = add_bias
        self.block_type = block_type

        self.block1_hidden_dims = block1_hidden_dims
        self.block1_hidden_activations = block1_hidden_activations
        self.block1_dropout_rates = block1_dropout_rates

        self.block2_hidden_dims = block2_hidden_dims
        self.block2_hidden_activations = block2_hidden_activations
        self.block2_dropout_rates = block2_dropout_rates

        self.residual_type = residual_type

        self.proj_output_embeddings = proj_output_embeddings

        loss_cls = import_module_by_path(loss_fn)
        self.loss_fn = loss_cls(**(loss_params or {}))

        if features_config is not None:
            self._init_modules(features_config=features_config)

    def _init_modules(self, features_config: FeaturesConfig) -> None:
        """Instantiate FinalNet modules based on features config.

        Args:
            features_config: features config
        """
        self.embedding_layer = EmbeddingLayer(
            features_config=features_config,
            is_dict_input=True,
            proj_output_features=self.proj_output_embeddings,
            flatten_emb=False,
        )

        input_dim = features_config.num_final_features
        num_fields = len(features_config.features)

        self.fields_gate = nn.Identity()
        if self.use_field_gate:
            self.fields_gate = FieldGate(num_fields=num_fields)

        self.block1 = FinalBlock(
            input_dim=input_dim if not self.use_field_gate else input_dim * 2,
            hidden_dims=self.block1_hidden_dims,
            hidden_activations=self.block1_hidden_activations,
            dropout_rates=self.block1_dropout_rates,
            use_batch_norm=self.use_batch_norm,
            add_bias=self.add_bias,
            residual_type=self.residual_type,
        )
        self.proj1 = nn.Linear(self.block1_hidden_dims[-1], 1)

        self.block2, self.proj2 = None, None
        if self.block_type == "2B":
            self.block2 = FinalBlock(
                input_dim=input_dim,
                hidden_dims=self.block2_hidden_dims,
                hidden_activations=self.block2_hidden_activations,
                dropout_rates=self.block2_dropout_rates,
                use_batch_norm=self.use_batch_norm,
                add_bias=self.add_bias,
                residual_type=self.residual_type,
            )
            self.proj2 = nn.Linear(self.block1_hidden_dims[-1], 1)

    def forward(self, input_: dict[str, torch.Tensor] | torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            input_: input tensor [{feature_name: feature_tensor} or tensor (batch_size, concat_features_len)]

        Returns:
            Tensor after FinalNet applying
        """
        embeddings_out = self.embedding_layer(input_)
        fields_gate_out = self.fields_gate(embeddings_out)
        block1_out = self.proj1(self.block1(fields_gate_out.flatten(start_dim=1)))

        if self.block_type == "1B":
            return block1_out

        block2_out = self.proj2(self.block2(embeddings_out.flatten(start_dim=1)))
        return (block1_out + block2_out) * 0.5

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
