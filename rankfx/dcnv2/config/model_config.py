"""Module with DCNv2 model config implementation."""

import enum
from collections.abc import Callable, Sequence
from dataclasses import field
from typing import Any

from pydantic.dataclasses import dataclass
from torch import nn

from rankfx.common.base.config.base_config import BaseConfig


@enum.unique
class ModelStructure(enum.Enum):
    """Possible model configurations."""
    CROSSNET_ONLY = "crossnet_only"
    STACKED = "stacked"
    PARALLEL = "parallel"
    STACKED_PARALLEL = "stacked_parallel"


@dataclass
class DCNv2Config(BaseConfig):
    """DCNv2 config implementation."""

    model_structure: ModelStructure = field(default=ModelStructure.STACKED)

    output_dim: int = field(default=1)

    loss_fn: str = field(default="torch.nn.BCEWithLogitsLoss")
    loss_params: dict[str, Any] = field(default_factory=lambda: {})

    # CrossNet parameters
    use_low_rank_mixture: bool = field(default=True)
    cross_low_rank_dim: int = field(default=32)
    num_cross_layers: int = field(default=4)
    num_cross_experts: int = field(default=4)

    # Parallel DNN parameters
    parallel_hidden_dims: Sequence[int] | None = field(default=None)
    parallel_dropout: float = field(default=0.)
    parallel_use_batch_norm: bool = field(default=True)
    parallel_activation: Callable = field(default=nn.ReLU)

    # Stacked DNN parameters
    stacked_hidden_dims: Sequence[int] | None = field(default=None)
    stacked_dropout: float = field(default=0.)
    stacked_use_batch_norm: bool = field(default=True)
    stacked_activation: Callable = field(default=nn.ReLU)

    def get_backbone_output_dim(self, input_dim: int | None = None) -> int:
        """Get output dim after CrossNet + DNN steps."""
        if self.model_structure == ModelStructure.CROSSNET_ONLY:
            return input_dim
        if self.model_structure == ModelStructure.STACKED:
            return self.stacked_hidden_dims[-1]
        if self.model_structure == ModelStructure.PARALLEL:
            return input_dim + self.parallel_hidden_dims[-1]
        return self.stacked_hidden_dims[-1] + self.parallel_hidden_dims[-1]
