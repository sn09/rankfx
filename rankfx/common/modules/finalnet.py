"""Module FinalNet blocks implementation."""

from collections.abc import Callable, Sequence
from typing import Literal

import torch
from torch import nn


class FieldGate(nn.Module):
    """FieldGate block implementation."""
    def __init__(self, num_fields: int):
        """FieldGate block instantiation.

        Args:
            num_fields: number of fields to project
        """
        super().__init__()
        self.proj_field = nn.Linear(num_fields, num_fields)

    def forward(self, feature_emb: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            feature_emb: feature embeddings tensor

        Returns:
            Tensor after FieldGate applying
        """
        gates = self.proj_field(feature_emb.transpose(1, 2)).transpose(1, 2)
        out = torch.cat([feature_emb, feature_emb * gates], dim=1)
        return out


class FinalBlock(nn.Module):
    """FinalNet basic block implementation."""
    def __init__(
        self,
        input_dim: int,
        hidden_dims: Sequence[int],
        hidden_activations: Sequence[Callable] | Callable | None = None,
        dropout_rates: Sequence[float] | float | None = None,
        use_batch_norm: bool = True,
        add_bias: bool = True,
        residual_type: Literal["sum", "concat"] = "sum",
    ):
        """FinalNet basic block instantiation.

        Args:
            input_dim: number of input dimensions
            hidden_dims: hidden layers dimensions
            hidden_activations: hidden layers activations
            dropout_rates: hidden layers dropout rates
            use_batch_norm: use batch norm for hidden layers
            add_bias: add bias to FactorizedInteraction layer
            residual_type: type of residual for FactorizedInteraction, one of `sum` and `concat`
        """
        super().__init__()
        dropout_rates = dropout_rates or 0
        if not isinstance(dropout_rates, list):
            dropout_rates = [dropout_rates] * len(hidden_dims)

        hidden_activations = hidden_activations or nn.ReLU
        if not isinstance(hidden_activations, list):
            hidden_activations = [hidden_activations] * len(hidden_dims)

        hidden_dims = [input_dim, *hidden_dims]
        layers = []
        for idx, (in_dim, out_dim) in enumerate(zip(hidden_dims[:-1], hidden_dims[1:], strict=False)):
            layers.append(
                FactorizedInteraction(
                    input_dim=in_dim,
                    output_dim=out_dim,
                    bias=add_bias,
                    residual_type=residual_type,
                    activation_fn=None,
                )
            )
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(out_dim))
            layers.append(hidden_activations[idx]())
            if dropout_rates[idx] > 0:
                layers.append(nn.Dropout(dropout_rates[idx]))

        self.net = nn.Sequential(*layers)

    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            input_: input tensor

        Returns:
            Tensor after FinalBlock applying
        """
        output = self.net(input_)
        return output


class FactorizedInteraction(nn.Module):
    """Factorized interaction layer implementaion."""
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        bias: bool = True,
        residual_type: Literal["sum", "concat"] = "sum",
        activation_fn: Callable | None = None,
    ):
        """Factorized interaction layer instantiation.

        A replacement of nn.Linear to enhance multiplicative feature interactions.
        `residual_type="concat"` uses the same number of parameters as nn.Linear
        `residual_type="sum"` doubles the number of parameters.

        Args:
            input_dim: number of input dimensions
            output_dim: number of output dimensions
            bias: add bias to linear layer
            residual_type: type of residual, one of `sum` and `concat`
            activation_fn: activation function to use
        """
        super().__init__()
        if residual_type not in ["sum", "concat"]:
            raise RuntimeError(f"Unknown type of residual `{residual_type}`, should be on of [`sum`, `concat`]")

        if residual_type == "concat" and output_dim % 2:
            raise RuntimeError(f"`output_dim` should be divisible by 2, got {output_dim}")

        self.residual_type = residual_type
        if residual_type == "sum":
            output_dim = output_dim * 2
        self.linear = nn.Linear(input_dim, output_dim, bias=bias)

        activation_fn = activation_fn or nn.Identity
        self.activation = activation_fn()

    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            input_: input tensor

        Returns:
            Tensor after FactorizedInteraction applying
        """
        linear_out = self.linear(input_)
        chunk_1, chunk_2 = torch.chunk(linear_out, chunks=2, dim=-1)

        chunk_2 = self.activation(chunk_2)
        if self.residual_type == "concat":
            return torch.cat([chunk_1, chunk_1 * chunk_2], dim=-1)

        return chunk_1 + chunk_2 * chunk_1
