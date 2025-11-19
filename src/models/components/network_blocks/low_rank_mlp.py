from typing import List, Optional
import torch
from torch import nn
import math
import torch.nn.functional as F

class LowRankLinear(nn.Module):
    """A linear layer implemented as a low-rank factorization W = U @ V.

    This module always uses a factorized representation. `rank` must be a
    positive integer. U has shape (out_features, rank) and V has shape
    (rank, in_features). The effective weight has shape (out_features, in_features).
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int,
        bias: bool = True,
    ) -> None:
        super().__init__()
        if not isinstance(rank, int) or rank <= 0:
            raise ValueError(f"rank must be a positive int, got {rank}")
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank

        # parameterize weight as U @ V where U: (out, rank), V: (rank, in)
        self.U = nn.Parameter(torch.empty(out_features, rank))
        self.V = nn.Parameter(torch.empty(rank, in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)

        # initialize
        nn.init.kaiming_uniform_(self.U, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.V, a=math.sqrt(5))
        if bias:
            fan_in = in_features
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0.0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # weight has shape (out_features, in_features)
        weight = torch.matmul(self.U, self.V)
        return F.linear(x, weight, self.bias)


class LowRankMLP(nn.Module):
    """A simple fully-connected neural net supporting low-rank linear layers.

    Each linear layer is replaced by a LowRankLinear when `rank` is provided.
    If `rank` is None or large enough, a standard nn.Linear is used.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim_list: Optional[List[int]] = None,
        activation: nn.Module = nn.ReLU,
        bias: bool = True,
        dropout: float = 0.0,
        rank: int = 1,
    ) -> None:
        super().__init__()

        if hidden_dim_list is None:
            hidden_dim_list = []
        # append output dim to keep same behavior as previous implementation
        hidden_dim_list = list(hidden_dim_list) + [output_dim]

        modules: List[nn.Module] = []

        if not isinstance(rank, int) or rank <= 0:
            raise ValueError(f"rank must be a positive int, got {rank}")

        def make_linear(in_f: int, out_f: int) -> nn.Module:
            return LowRankLinear(in_f, out_f, rank=rank, bias=bias)

        # first linear
        modules.append(make_linear(input_dim, hidden_dim_list[0]))

        for i in range(1, len(hidden_dim_list)):
            # activation between linears (keeps previous ordering)
            modules.append(activation())
            modules.append(make_linear(hidden_dim_list[i - 1], hidden_dim_list[i]))
            modules.append(nn.Dropout(dropout))

        # use ModuleList to avoid strict sequential behavior and allow future introspection
        self.model = nn.ModuleList(modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x
        for layer in self.model:
            out = layer(out)
        return out