"""Module with available features types listing."""

import enum


@enum.unique
class FeatureType(enum.Enum):
    """Available feature types."""
    NUMERICAL = "numerical"
    CATEGORICAL = "categorical"
    SEQUENTIAL = "sequential"
