"""Module with available features types listing."""

import enum


@enum.unique
class FeatureType(enum.Enum):
    """Available feature types."""
    NUMERICAL = "numerical"
    CATEGORICAL = "categorical"
    NUMERICAL_SEQUENCE = "numerical_sequence"
    CATEGORICAL_SEQUENCE = "categorical_sequence"
