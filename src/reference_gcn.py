"""DGL GraphConv reference wrapper sharing the manual implementations' API."""

from __future__ import annotations

from typing import Callable

import dgl
import torch
from dgl.nn import GraphConv
from torch import nn

from .graph_utils import SelfLoopPolicy, preprocess_edges


class ReferenceGCN(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, activation: Callable | None = None):
        super().__init__()
        self.in_features, self.out_features = in_features, out_features
        self.conv = GraphConv(in_features, out_features, norm="both", weight=True, bias=bias, activation=activation, allow_zero_in_degree=True)

    @property
    def weight(self):
        return self.conv.weight

    @property
    def bias(self):
        return self.conv.bias

    def forward(self, num_nodes: int, edges, features: torch.Tensor, self_loop_policy: SelfLoopPolicy = "add", deduplicate: bool = False):
        if features.ndim != 2 or features.shape != (num_nodes, self.in_features):
            raise ValueError(f"features must have shape ({num_nodes}, {self.in_features})")
        graph = preprocess_edges(num_nodes, edges, self_loop_policy, deduplicate, features.device)
        if num_nodes == 0:
            output = features @ self.weight
            if self.bias is not None:
                output = output + self.bias
            return self.conv._activation(output) if self.conv._activation is not None else output
        dgl_graph = dgl.graph((graph.src, graph.dst), num_nodes=num_nodes, device=features.device)
        return self.conv(dgl_graph, features)


def copy_parameters(source: nn.Module, target: nn.Module) -> None:
    with torch.no_grad():
        target.weight.copy_(source.weight)
        if source.bias is not None and target.bias is not None:
            target.bias.copy_(source.bias)
