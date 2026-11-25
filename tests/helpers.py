import torch

from src.manual_gcn import ManualGCN
from src.message_passing import MessagePassingGCN
from src.reference_gcn import ReferenceGCN, copy_parameters

ATOL, RTOL = 1e-6, 1e-5


def make_models(fin, fout, bias=True, activation=None, seed=7):
    torch.manual_seed(seed)
    reference = ReferenceGCN(fin, fout, bias=bias, activation=activation)
    dense = ManualGCN(fin, fout, bias=bias, activation=activation)
    sparse = MessagePassingGCN(fin, fout, bias=bias, activation=activation)
    copy_parameters(reference, dense)
    copy_parameters(reference, sparse)
    return dense, sparse, reference


def assert_equivalent(num_nodes, edges, x, fout, bias=True, activation=None, policy="add", deduplicate=False):
    outputs = []
    for model in make_models(x.shape[1], fout, bias, activation):
        y = model(num_nodes, edges, x, policy, deduplicate)
        assert y.shape == (num_nodes, fout)
        assert torch.isfinite(y).all()
        outputs.append(y)
    assert torch.allclose(outputs[0], outputs[1], atol=ATOL, rtol=RTOL)
    assert torch.allclose(outputs[0], outputs[2], atol=ATOL, rtol=RTOL)
    return outputs
