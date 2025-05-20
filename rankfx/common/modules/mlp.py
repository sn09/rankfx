"""Basic MLP net module implementation."""

from collections.abc import Callable, Sequence

import torch
from torch import nn


class MLPBlock(nn.Module):
    """MLP block implementation."""
    def __init__(
        self,
        in_features: int,
        out_features: int | None = None,
        hidden_dims: Sequence[int] | None = None,
        activation_fn: Callable = nn.ReLU,
        dropout: float = 0.,
        use_batch_norm: bool = True,
        add_bias: bool = True,
    ):
        """Instantiate MLPBlock module.

        Args:
            in_features: number of input features
            out_features: number of output features
            hidden_dims: hidden net dimensions
            activation_fn: callable to get activation for hidden layers
            dropout: dropout rate
            use_batch_norm: use batch norm for hidden blocks
            add_bias: add bias for linear layers
        """
        super().__init__()

        if out_features is None and hidden_dims is None:
            raise RuntimeError("both `out_features` and `hidden_dims` are None, set at least one of them")

        hidden_dims = [in_features, *(hidden_dims or [])]

        layers = []
        for in_dim, out_dim in zip(hidden_dims[:-1], hidden_dims[1:], strict=False):
            layers.append(nn.Linear(in_dim, out_dim, bias=add_bias))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(out_dim))
            layers.append(activation_fn())
            if dropout > 0:
                layers.append(nn.Dropout(p=dropout))

        if out_features is not None:
            layers.append(nn.Linear(hidden_dims[-1], out_features, bias=add_bias))
        self.mlp = nn.Sequential(*layers)

    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            input_: input tensor

        Returns:
            Tensor after MLPBlock applying
        """
        return self.mlp(input_)
