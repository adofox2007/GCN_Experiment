"""Sparse edge-list/message-passing implementation of one GCN layer."""

from __future__ import annotations

import math
from typing import Callable

import torch
from torch import nn

from .graph_utils import SelfLoopPolicy, degrees, preprocess_edges


class MessagePassingGCN(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, activation: Callable | None = None):
        super().__init__()
        if in_features <= 0 or out_features <= 0:
            raise ValueError("feature dimensions must be positive")
        self.in_features, self.out_features, self.activation = in_features, out_features, activation
        self.weight = nn.Parameter(torch.empty(in_features, out_features))
        self.bias = nn.Parameter(torch.empty(out_features)) if bias else None
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.weight, -1 / math.sqrt(self.in_features), 1 / math.sqrt(self.in_features))
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, num_nodes: int, edges, features: torch.Tensor, self_loop_policy: SelfLoopPolicy = "add", deduplicate: bool = False):
        if features.ndim != 2 or features.shape != (num_nodes, self.in_features):
            raise ValueError(f"features must have shape ({num_nodes}, {self.in_features})")
        if not features.is_floating_point():
            raise TypeError("features must be floating point")
        graph = preprocess_edges(num_nodes, edges, self_loop_policy, deduplicate, features.device)
        out_degree, in_degree = degrees(graph, dtype=features.dtype)
        out_degree, in_degree = out_degree.to(features.device), in_degree.to(features.device)
        out_inv = torch.where(out_degree > 0, out_degree.rsqrt(), torch.zeros_like(out_degree))
        in_inv = torch.where(in_degree > 0, in_degree.rsqrt(), torch.zeros_like(in_degree))
        # Keep an explicit zero-valued path to features so a graph with no edges
        # produces a zero input gradient instead of leaving features.grad as None.
        aggregated = features * 0.0
        if graph.num_edges:
            edge_norm = out_inv[graph.src] * in_inv[graph.dst]
            messages = features[graph.src] * edge_norm[:, None]
            aggregated.index_add_(0, graph.dst, messages)
        output = aggregated @ self.weight
        if self.bias is not None:
            output = output + self.bias
        return self.activation(output) if self.activation is not None else output
