"""Module with DCNv2 model class implementation."""

from typing import Any

import torch
from sklearn.metrics import log_loss, roc_auc_score
from torch import nn

from common.base.model import NNPandasModel
from common.features.config import FeaturesConfig
from common.modules import CrossNetMix, CrossNetV2, EmbeddingLayer, MLPBlock
from common.utils.import_utils import import_module_by_path
from dcnv2.config.model_config import DCNv2Config, ModelStructure


class DCNv2(NNPandasModel):
    """DCNv2 implementation."""
    def __init__(self, model_config: DCNv2Config, features_config: FeaturesConfig, is_dict_input: bool = True):
        """Instantiate DCNv2 module.

        Features are taken in the order, specified in features_config.features
        If is_dict_input is False, feature will be taken as slices from input tensor

        Args:
            model_config: model config instance
            features_config: features config instance
            is_dict_input: is input looks like {feature_name: feature_tensor} or just simple tensor
        """
        super().__init__()

        self.model_config = model_config
        self.features_config = features_config
        self.is_dict_input = is_dict_input

        loss_cls = import_module_by_path(model_config.loss_fn)
        self.loss_fn = loss_cls(**model_config.loss_params)

        self.embedding_layer = EmbeddingLayer(features_config=features_config, is_dict_input=is_dict_input)
        input_dim = features_config.num_final_features

        if model_config.use_low_rank_mixture:
            self.crossnet = CrossNetMix(
                input_dim,
                low_rank_dim=model_config.cross_low_rank_dim,
                num_layers=model_config.num_cross_layers,
                num_experts=model_config.num_cross_experts,
            )
        else:
            self.crossnet = CrossNetV2(input_dim, num_layers=model_config.num_cross_layers)

        if model_config.model_structure in (ModelStructure.STACKED, ModelStructure.STACKED_PARALLEL):
            self.stacked_dnn = MLPBlock(
                in_features=input_dim,
                hidden_dims=model_config.stacked_hidden_dims,
                activation_fn=model_config.stacked_activation,
                dropout=model_config.stacked_dropout,
                use_batch_norm=model_config.stacked_use_batch_norm,
            )

        if model_config.model_structure in (ModelStructure.PARALLEL, ModelStructure.STACKED_PARALLEL):
            self.parallel_dnn = MLPBlock(
                in_features=input_dim,
                hidden_dims=model_config.parallel_hidden_dims,
                activation_fn=model_config.parallel_activation,
                dropout=model_config.parallel_dropout,
                use_batch_norm=model_config.parallel_use_batch_norm,
            )

        final_dim = model_config.get_backbone_output_dim(input_dim=input_dim)
        self.fc = nn.Linear(final_dim, model_config.output_dim)

    def forward(self, input_: dict[str, torch.Tensor] | torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            input_: input tensor [{feature_name: feature_tensor} or tensor (batch_size, concat_features_len)]

        Returns:
            Tensor after DCNv2 applying
        """
        embeddings_out = self.embedding_layer(input_)
        cross_out = self.crossnet(embeddings_out)
        if self.model_config.model_structure == ModelStructure.CROSSNET_ONLY:
            final_out = cross_out
        elif self.model_config.model_structure == ModelStructure.STACKED:
            final_out = self.stacked_dnn(cross_out)
        elif self.model_config.model_structure == ModelStructure.PARALLEL:
            final_out = torch.cat([cross_out, self.parallel_dnn(embeddings_out)], dim=-1)
        elif self.model_config.model_structure == ModelStructure.STACKED_PARALLEL:
            final_out = torch.cat([self.stacked_dnn(cross_out), self.parallel_dnn(embeddings_out)], dim=-1)
        return self.fc(final_out)

    def train_step(self, batch: dict[str, torch.Tensor] | torch.Tensor) -> dict[str, Any]:
        """Model training step.

        Args:
            batch: batch of data

        Returns:
            Loss/metrics after training step
        """
        if not self.is_dict_input:
            batch, target = batch
        else:
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
        if not self.is_dict_input:
            batch, target = batch
        else:
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
        if not self.is_dict_input:
            batch, target = batch
        else:
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
        if not self.is_dict_input:
            batch, _ = batch

        logits = self.forward(batch).squeeze()
        return logits
