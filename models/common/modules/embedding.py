"""Module with Embedding layer implementation."""

import torch
from torch import nn

from common.features.config import FeaturesConfig
from common.features.types import FeatureType


class EmbeddingLayer(nn.Module):
    """Embedding Layer implementation."""
    def __init__(
            self,
            features_config: FeaturesConfig,
            is_dict_input: bool = True,
            flatten_emb: bool = True,
        ):
        """Instantiate EmbeddingLayer module.

        Features are taken in the order, specified in features_config.features
        If is_dict_input is False, feature will be taken as slices from input tensor

        Args:
            features_config: features config instance
            is_dict_input: is input looks like {feature_name: feature_tensor} or just simple tensor
            flatten_emb: do flatten embeddings output into single vector, or try to stack by features
        """
        super().__init__()
        self.config = features_config
        self.is_dict_input = is_dict_input
        self.flatten_emb = flatten_emb

        modules = {}
        for feature in features_config.features:
            if feature.needs_embed:
                if feature.feature_type == FeatureType.CATEGORICAL:
                    modules[feature.name] = nn.Embedding(
                        feature.embedding_vocab_size,
                        feature.embedding_size,
                        padding_idx=feature.embedding_padding_idx,
                    )
                    continue
                modules[feature.name] = nn.Linear(feature.feature_size, feature.embedding_size, bias=False)

        self.embedding_modules = nn.ModuleDict(modules)
        self.dummy_fn = nn.Identity()

    def forward(self, input_: dict[str, torch.Tensor] | torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            input_: input tensor [{feature_name: feature_tensor} or tensor (batch_size, concat_features_len)]

        Returns:
            Tensor after EmbeddingLayer applying
        """
        if not self.is_dict_input and self.config.num_initial_features != input_.size(-1):
            raise RuntimeError(
                f"Number of features in config ({self.config.num_initial_features})"
                f"and in input tensor ({input_.size(-1)}) are not equal"
            )
        outputs = []
        current_idx = 0
        for feature in self.config.features:
            feature_value = (
                input_[feature]
                if self.is_dict_input
                else input_[:, current_idx:current_idx + feature.feature_size]
            )
            if feature.feature_type == FeatureType.CATEGORICAL:
                feature_value = feature_value.long()
                if feature.needs_embed:
                    feature_value = feature_value.squeeze()

            if feature.name in self.embedding_modules:
                output = self.embedding_modules[feature.name](feature_value)
            else:
                output = self.dummy_fn(feature_value)

            current_idx += feature.feature_size
            outputs.append(output)

        if self.flatten_emb:
            # (batch_size, concat_features_len)
            return torch.cat(outputs, dim=-1)

        # (batch_size, feature_num, feature_len) - features should be the same size
        return torch.stack(outputs, dim=1)
