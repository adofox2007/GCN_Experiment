import torch

from tests.helpers import ATOL, RTOL, make_models


def _gradients():
    edges = ([0, 0, 1, 2, 3], [1, 2, 2, 3, 0])
    base = torch.randn(4, 3)
    models = make_models(3, 2, bias=True, activation=torch.tanh, seed=20)
    results = []
    for model in models:
        x = base.clone().requires_grad_(True)
        output = model(4, edges, x)
        loss = (output.square() * torch.arange(1, 9).reshape(4, 2)).sum()
        loss.backward()
        results.append((x.grad.detach().clone(), model.weight.grad.detach().clone(), model.bias.grad.detach().clone()))
    return results


def test_t19_input_feature_gradient_consistency():
    grads = _gradients()
    assert torch.allclose(grads[0][0], grads[1][0], atol=ATOL, rtol=RTOL)
    assert torch.allclose(grads[0][0], grads[2][0], atol=ATOL, rtol=RTOL)


def test_t20_weight_and_bias_gradient_consistency():
    grads = _gradients()
    for index in (1, 2):
        assert torch.allclose(grads[0][index], grads[1][index], atol=ATOL, rtol=RTOL)
        assert torch.allclose(grads[0][index], grads[2][index], atol=ATOL, rtol=RTOL)


def test_t21_no_edge_without_self_loops_has_zero_input_gradient():
    from src import ManualGCN, MessagePassingGCN, ReferenceGCN

    base = torch.randn(3, 2)
    for cls in (ManualGCN, MessagePassingGCN, ReferenceGCN):
        model = cls(2, 1, bias=False)
        x = base.clone().requires_grad_(True)
        model(3, ([], []), x, self_loop_policy="none").sum().backward()
        assert x.grad is not None
        assert torch.equal(x.grad, torch.zeros_like(x))
