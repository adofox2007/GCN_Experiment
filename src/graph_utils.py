"""Graph validation, preprocessing, degree and random graph utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import torch

SelfLoopPolicy = Literal["none", "add", "exactly_one"]


@dataclass(frozen=True)
class ProcessedGraph:
    num_nodes: int
    src: torch.Tensor
    dst: torch.Tensor

    @property
    def num_edges(self) -> int:
        return int(self.src.numel())


def _as_index_tensor(values: Sequence[int] | torch.Tensor, device=None) -> torch.Tensor:
    try:
        tensor = torch.as_tensor(values, device=device)
    except (TypeError, ValueError, RuntimeError) as exc:
        raise TypeError("edge indices must be a one-dimensional integer sequence") from exc
    if tensor.ndim != 1:
        raise ValueError("edge indices must be one-dimensional")
    if tensor.numel() == 0:
        return tensor.to(dtype=torch.long)
    integer_dtypes = {torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64}
    if tensor.dtype not in integer_dtypes:
        raise TypeError("edge indices must contain integers")
    return tensor.to(dtype=torch.long)


def validate_graph(num_nodes: int, src: torch.Tensor, dst: torch.Tensor) -> None:
    if not isinstance(num_nodes, int) or isinstance(num_nodes, bool) or num_nodes < 0:
        raise ValueError("num_nodes must be a non-negative integer")
    if src.ndim != 1 or dst.ndim != 1 or src.numel() != dst.numel():
        raise ValueError("src and dst must be one-dimensional arrays with equal length")
    if src.dtype != torch.long or dst.dtype != torch.long:
        raise TypeError("edge indices must have dtype torch.long")
    if src.numel() and ((src < 0).any() or (dst < 0).any() or (src >= num_nodes).any() or (dst >= num_nodes).any()):
        raise ValueError("edge index is outside [0, num_nodes)")


def remove_self_loops(src: torch.Tensor, dst: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    keep = src != dst
    return src[keep], dst[keep]


def deduplicate_edges(src: torch.Tensor, dst: torch.Tensor, num_nodes: int) -> tuple[torch.Tensor, torch.Tensor]:
    if src.numel() == 0:
        return src, dst
    codes = src * max(num_nodes, 1) + dst
    unique_codes = torch.unique(codes, sorted=True)
    return unique_codes // max(num_nodes, 1), unique_codes % max(num_nodes, 1)


def preprocess_edges(
    num_nodes: int,
    edges: tuple[Sequence[int] | torch.Tensor, Sequence[int] | torch.Tensor],
    self_loop_policy: SelfLoopPolicy = "add",
    deduplicate: bool = False,
    device=None,
) -> ProcessedGraph:
    """Validate edges and apply one shared preprocessing policy.

    ``add`` strictly performs A + I: existing loops remain and one extra loop is
    appended for every node. ``exactly_one`` removes all loops then adds one per
    node. Ordinary parallel edges are retained unless ``deduplicate`` is true.
    """
    if self_loop_policy not in {"none", "add", "exactly_one"}:
        raise ValueError(f"unknown self-loop policy: {self_loop_policy}")
    src = _as_index_tensor(edges[0], device=device)
    dst = _as_index_tensor(edges[1], device=device)
    validate_graph(num_nodes, src, dst)
    if deduplicate:
        src, dst = deduplicate_edges(src, dst, num_nodes)
    if self_loop_policy == "exactly_one":
        src, dst = remove_self_loops(src, dst)
    if self_loop_policy in {"add", "exactly_one"} and num_nodes:
        nodes = torch.arange(num_nodes, dtype=torch.long, device=src.device)
        src, dst = torch.cat((src, nodes)), torch.cat((dst, nodes))
    return ProcessedGraph(num_nodes, src, dst)


def degrees(graph: ProcessedGraph, dtype=torch.float32) -> tuple[torch.Tensor, torch.Tensor]:
    out_degree = torch.bincount(graph.src, minlength=graph.num_nodes).to(dtype=dtype)
    in_degree = torch.bincount(graph.dst, minlength=graph.num_nodes).to(dtype=dtype)
    return out_degree, in_degree


def adjacency_matrix(graph: ProcessedGraph, dtype=torch.float32, device=None) -> torch.Tensor:
    """Return A with A[dst, src], so A @ H aggregates source features at destinations."""
    device = graph.src.device if device is None else device
    adj = torch.zeros((graph.num_nodes, graph.num_nodes), dtype=dtype, device=device)
    if graph.num_edges:
        ones = torch.ones(graph.num_edges, dtype=dtype, device=device)
        adj.index_put_((graph.dst.to(device), graph.src.to(device)), ones, accumulate=True)
    return adj


def normalized_adjacency(graph: ProcessedGraph, dtype=torch.float32, device=None) -> torch.Tensor:
    adj = adjacency_matrix(graph, dtype=dtype, device=device)
    out_degree, in_degree = degrees(graph, dtype=dtype)
    out_degree, in_degree = out_degree.to(adj.device), in_degree.to(adj.device)
    out_inv = torch.where(out_degree > 0, out_degree.rsqrt(), torch.zeros_like(out_degree))
    in_inv = torch.where(in_degree > 0, in_degree.rsqrt(), torch.zeros_like(in_degree))
    return in_inv[:, None] * adj * out_inv[None, :]


def random_edges(num_nodes: int, num_edges: int, directed: bool, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    if num_nodes < 0 or num_edges < 0:
        raise ValueError("num_nodes and num_edges must be non-negative")
    if num_nodes == 0:
        if num_edges:
            raise ValueError("a zero-node graph cannot contain edges")
        empty = torch.empty(0, dtype=torch.long)
        return empty, empty
    generator = torch.Generator().manual_seed(seed)
    if directed:
        return (
            torch.randint(num_nodes, (num_edges,), generator=generator),
            torch.randint(num_nodes, (num_edges,), generator=generator),
        )
    # A non-self edge needs its reverse, so an odd edge budget cannot consist
    # only of reciprocal pairs. Use floor(E/2) pairs plus one self-loop. This
    # preserves the previous even-E behavior and keeps the edge multiset symmetric.
    pair_count = num_edges // 2
    u = torch.randint(num_nodes, (pair_count,), generator=generator)
    v = torch.randint(num_nodes, (pair_count,), generator=generator)
    src = torch.cat((u, v))
    dst = torch.cat((v, u))
    if num_edges % 2:
        loop = torch.randint(num_nodes, (1,), generator=generator)
        src = torch.cat((src, loop))
        dst = torch.cat((dst, loop))
    return src, dst
