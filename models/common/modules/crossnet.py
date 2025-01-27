"""Module with CrossNet implementaion."""

import torch
from torch import nn


class CrossNetV2(nn.Module):
    """CrossNetv2 implementation."""
    def __init__(self, in_features: int, num_layers: int):
        """Instantiate CrossNetv2 module.

        Args:
            in_features: number of input features
            num_layers: numer of net layers
        """
        super().__init__()
        self.cross_layers = nn.ModuleList(
            nn.Linear(in_features, in_features) for _ in range(num_layers)
        )

    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            input_: input tensor

        Returns:
            Tensor after CrossNet applying
        """
        output_i = input_
        for layer in self.cross_layers:
            output_i = input_ * layer(output_i) + output_i
        return output_i


class CrossNetMix(nn.Module):
    """CrossNetMix implementation.

    Improves CrossNetV2 by:
        1. Add MOE to learn feature interactions in different subspaces
        2. Add nonlinear transformations in low-dimensional space
    """
    def __init__(self, in_features: int, num_layers: int = 2, low_rank_dim: int = 32, num_experts: int = 4):
        """Instantiate CrossNetMix module.

        Args:
            in_features: number of input features
            num_layers: numer of net layers
            low_rank_dim: dimension of inner subspace
            num_experts: number of experts in MoE
        """
        super().__init__()
        self.num_layers = num_layers
        self.num_experts = num_experts

        # U: (in_features, low_rank)
        self.U_layers = nn.ParameterList(
            nn.Parameter(nn.init.xavier_normal_(torch.empty(num_experts, in_features, low_rank_dim)))
            for _ in range(self.num_layers)
        )
        # V: (in_features, low_rank)
        self.V_layers = nn.ParameterList(
            nn.Parameter(nn.init.xavier_normal_(torch.empty(num_experts, in_features, low_rank_dim)))
            for _ in range(self.num_layers)
        )
        # C: (low_rank, low_rank)
        self.C_layers = nn.ParameterList(
            nn.Parameter(nn.init.xavier_normal_(torch.empty(num_experts, low_rank_dim, low_rank_dim)))
            for _ in range(self.num_layers)
        )
        self.gating = nn.ModuleList(nn.Linear(in_features, 1, bias=False) for _ in range(self.num_experts))

        self.bias = nn.ParameterList(
            nn.Parameter(nn.init.zeros_(torch.empty(in_features, 1)))
            for _ in range(self.num_layers)
        )

    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            input_: input tensor

        Returns:
            Tensor after CrossNetMix applying
        """
        # input_: (batch_size, in_features, 1)
        initial_state = input_.unsqueeze(-1)
        interm_state = initial_state
        for layer_idx in range(self.num_layers):
            output_of_experts = []
            gating_score_of_experts = []
            for expert_idx in range(self.num_experts):
                # (1) G(x_l)
                # compute the gating score by x_l
                gating_score_of_experts.append(self.gating[expert_idx](interm_state.squeeze(-1)))

                # (2) E(x_l)
                # project the input x_l to $\mathbb{R}^{r}$
                # layer_output: (batch_size, low_rank_dim, 1)
                layer_output = torch.matmul(self.V_layers[layer_idx][expert_idx].T, interm_state)

                # nonlinear activation in low rank space
                layer_output = torch.tanh(layer_output)
                layer_output = torch.matmul(self.C_layers[layer_idx][expert_idx], layer_output)
                layer_output = torch.tanh(layer_output)

                # project back to $\mathbb{R}^{d}$
                # layer_output: (batch_size, in_features, 1)
                layer_output = torch.matmul(self.U_layers[layer_idx][expert_idx], layer_output)

                # Hadamard-product
                layer_output += self.bias[layer_idx]
                layer_output = initial_state * layer_output

                output_of_experts.append(layer_output.squeeze(-1))

            # (3) mixture of low-rank experts
            # output_of_experts: (batch_size, in_features, num_experts)
            output_of_experts = torch.stack(output_of_experts, -1)

            # gating_score_of_experts: (batch_size, num_experts, 1)
            gating_score_of_experts = torch.stack(gating_score_of_experts, -2)
            moe_out = torch.matmul(output_of_experts, gating_score_of_experts.softmax(-1))

            # interm_state: (batch_size, in_features, 1)
            interm_state = moe_out + interm_state

        # interm_state: (batch_size, in_features)
        interm_state = interm_state.squeeze(-1)
        return interm_state
