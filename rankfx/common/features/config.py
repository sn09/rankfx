"""Module with features configs implementation."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import field

from pydantic import model_validator
from pydantic.dataclasses import dataclass

from rankfx.common.base.config.base_config import BaseConfig
from rankfx.common.features.types import FeatureType


@dataclass
class Feature(BaseConfig):
    """Config for single feature."""
    name: str
    feature_type: FeatureType
    feature_size: int = field(default=1)

    # embedding parameters
    needs_embed: bool = field(default=False)
    embedding_size: int | None = field(default=None)
    embedding_vocab_size: int | None = field(default=None)
    embedding_padding_idx: int | None = field(default=None)

    @model_validator(mode="after")
    def check_feature_size(self) -> Feature:
        """Validate feature size."""
        if (
            self.feature_size > 1
            and self.feature_type not in [FeatureType.NUMERICAL_SEQUENCE, FeatureType.CATEGORICAL_SEQUENCE]
        ):
            raise ValueError(f"Feature size cannot be > 1 for scalar feature types, got {self.feature_size}")
        return self


@dataclass
class FeaturesConfig(BaseConfig):
    """Config for model features."""
    features: Sequence[Feature] | None = field(default=None)

    @property
    def num_initial_features(self) -> int:
        """Get final number of features."""
        size_ = 0
        for feature in self.features:
            size_ += feature.feature_size

        return size_

    @property
    def num_final_features(self) -> int:
        """Get final number of features."""
        size_ = 0
        for feature in self.features:
            feature_size = feature.embedding_size if feature.needs_embed else feature.feature_size
            size_ += feature_size

        return size_
