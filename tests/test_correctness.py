import torch

from src.graph_utils import random_edges
from tests.helpers import assert_equivalent


def test_t01_ordinary_undirected_graph():
    edges = ([0, 1, 1, 2], [1, 0, 2, 1])
    assert_equivalent(3, edges, torch.randn(3, 4), 2)


def test_t08_scalar_features_and_output():
    outputs = assert_equivalent(3, ([0, 1], [1, 0]), torch.tensor([[1.], [3.], [5.]]), 1, bias=False)
    expected = torch.tensor([[2.], [2.], [5.]])
    for output in outputs:
        # Set weights after the generic comparison in a dedicated independent check below.
        assert output.shape == expected.shape


def test_t08_independent_manual_expected_result():
    from src import ManualGCN, MessagePassingGCN, ReferenceGCN
    expected = torch.tensor([[2.], [2.], [5.]])
    for cls in (ManualGCN, MessagePassingGCN, ReferenceGCN):
        model = cls(1, 1, bias=False)
        with torch.no_grad():
            model.weight.fill_(1)
        actual = model(3, ([0, 1], [1, 0]), torch.tensor([[1.], [3.], [5.]]))
        assert torch.allclose(actual, expected, atol=1e-6)


def test_t09_feature_expansion():
    assert_equivalent(5, ([0, 1, 2, 3], [1, 2, 3, 4]), torch.randn(5, 2), 8)


def test_t10_feature_compression():
    assert_equivalent(5, ([0, 1, 2, 3], [1, 2, 3, 4]), torch.randn(5, 8), 2)


def test_t11_equal_feature_dimensions():
    assert_equivalent(4, ([0, 1, 2], [1, 2, 3]), torch.randn(4, 4), 4)


def test_t12_random_undirected_graph():
    edges = random_edges(30, 100, directed=False, seed=12)
    assert_equivalent(30, edges, torch.randn(30, 16), 8)


def test_t13_random_directed_graph():
    edges = random_edges(30, 91, directed=True, seed=13)
    assert_equivalent(30, edges, torch.randn(30, 7), 5)


def test_t17_bias_on_and_off():
    x, edges = torch.randn(4, 3), ([0, 1, 2], [1, 2, 3])
    assert_equivalent(4, edges, x, 2, bias=True)
    assert_equivalent(4, edges, x, 2, bias=False)


def test_t18_activation_on_and_off():
    x, edges = torch.randn(4, 3), ([0, 1, 2], [1, 2, 3])
    assert_equivalent(4, edges, x, 2, activation=torch.relu)
    assert_equivalent(4, edges, x, 2, activation=None)
