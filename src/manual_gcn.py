"""Dense adjacency implementation of one GCN layer using only PyTorch primitives."""

from __future__ import annotations

import math
from typing import Callable, Sequence

import torch
from torch import nn

from .graph_utils import SelfLoopPolicy, normalized_adjacency, preprocess_edges


class ManualGCN(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, activation: Callable | None = None):
        super().__init__()
        if in_features <= 0 or out_features <= 0:
            raise ValueError("feature dimensions must be positive")
        self.in_features, self.out_features, self.activation = in_features, out_features, activation
        self.weight = nn.Parameter(torch.empty(in_features, out_features))
        self.bias = nn.Parameter(torch.empty(out_features)) if bias else None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.uniform_(self.weight, -1 / math.sqrt(self.in_features), 1 / math.sqrt(self.in_features))
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, num_nodes: int, edges, features: torch.Tensor, self_loop_policy: SelfLoopPolicy = "add", deduplicate: bool = False):
        if features.ndim != 2 or features.shape != (num_nodes, self.in_features):
            raise ValueError(f"features must have shape ({num_nodes}, {self.in_features})")
        if not features.is_floating_point():
            raise TypeError("features must be floating point")
        graph = preprocess_edges(num_nodes, edges, self_loop_policy, deduplicate, features.device)
        norm_adj = normalized_adjacency(graph, dtype=features.dtype, device=features.device)
        aggregated = norm_adj @ features
        output = aggregated @ self.weight
        if self.bias is not None:
            output = output + self.bias
        return self.activation(output) if self.activation is not None else output

    def matrices(self, num_nodes: int, edges, dtype=torch.float32, self_loop_policy: SelfLoopPolicy = "add"):
        graph = preprocess_edges(num_nodes, edges, self_loop_policy)
        from .graph_utils import adjacency_matrix, degrees
        adjacency = adjacency_matrix(graph, dtype=dtype)
        out_degree, in_degree = degrees(graph, dtype=dtype)
        return adjacency, torch.diag(in_degree), normalized_adjacency(graph, dtype=dtype), out_degree, in_degree
