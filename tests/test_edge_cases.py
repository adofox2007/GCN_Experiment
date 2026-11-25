import pytest
import torch

from src import ManualGCN, MessagePassingGCN, ReferenceGCN
from src.graph_utils import adjacency_matrix, preprocess_edges
from tests.helpers import assert_equivalent


def test_t02_isolated_node():
    outputs = assert_equivalent(3, ([0, 1], [1, 0]), torch.tensor([[1.], [3.], [5.]]), 1, bias=False)
    assert all(torch.isfinite(y).all() for y in outputs)


def test_t03_existing_self_loop_strict_a_plus_i():
    graph = preprocess_edges(2, ([0, 0], [0, 1]), "add")
    adj = adjacency_matrix(graph)
    assert adj[0, 0] == 2  # original loop plus the added identity edge
    assert_equivalent(2, ([0, 0], [0, 1]), torch.randn(2, 3), 2)


def test_t04_parallel_edges_are_preserved():
    graph = preprocess_edges(2, ([0, 0, 1], [1, 1, 0]), "add")
    assert adjacency_matrix(graph)[1, 0] == 2
    outputs = assert_equivalent(2, ([0, 0, 1], [1, 1, 0]), torch.randn(2, 3), 2)
    dedup = assert_equivalent(2, ([0, 0, 1], [1, 1, 0]), torch.randn(2, 3), 2, deduplicate=True)
    assert outputs[0].shape == dedup[0].shape


def test_t05_nodes_without_edges():
    x = torch.randn(4, 3)
    outputs = assert_equivalent(4, ([], []), x, 2, bias=False)
    assert all(torch.isfinite(y).all() for y in outputs)


def test_t06_zero_node_zero_edge_graph():
    x = torch.empty((0, 3))
    outputs = assert_equivalent(0, ([], []), x, 2)
    assert all(y.numel() == 0 for y in outputs)


def test_t07_single_node_graph():
    assert_equivalent(1, ([], []), torch.tensor([[2., -1.]]), 3)


def test_t14_illegal_node_index():
    for cls in (ManualGCN, MessagePassingGCN, ReferenceGCN):
        with pytest.raises(ValueError, match="outside"):
            cls(2, 2)(2, ([0, 2], [1, 0]), torch.randn(2, 2))


def test_t15_illegal_feature_shape():
    for cls in (ManualGCN, MessagePassingGCN, ReferenceGCN):
        with pytest.raises(ValueError, match="shape"):
            cls(3, 2)(2, ([], []), torch.randn(3, 3))


def test_t16_zero_degree_without_self_loops():
    x = torch.tensor([[1.], [2.], [3.]])
    outputs = assert_equivalent(3, ([0], [1]), x, 1, bias=False, policy="none")
    assert all(torch.isfinite(y).all() for y in outputs)
    assert all(torch.allclose(y[2], torch.zeros(1)) for y in outputs)


def test_exactly_one_self_loop_policy():
    graph = preprocess_edges(3, ([0, 0, 1], [0, 0, 2]), "exactly_one")
    adj = adjacency_matrix(graph)
    assert torch.equal(torch.diag(adj), torch.ones(3))
    assert_equivalent(3, ([0, 0, 1], [0, 0, 2]), torch.randn(3, 2), 2, policy="exactly_one")
