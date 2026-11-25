from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import dgl
import torch

from experiments.common import RESULTS, setup, tensor_list, write_json
from src.reference_gcn import ReferenceGCN


def main():
    setup()
    features = torch.tensor([[1.0], [3.0], [5.0]])
    model = ReferenceGCN(1, 1, bias=True)
    with torch.no_grad():
        model.weight.fill_(1.0)
        model.bias.zero_()
    output = model(3, ([0, 1], [1, 0]), features)
    payload = {
        "implementation": "dgl.nn.GraphConv",
        "dgl_version": dgl.__version__,
        "norm": "both",
        "allow_zero_in_degree": True,
        "self_loop_policy": "strict A + I",
        "input": tensor_list(features),
        "weight": tensor_list(model.weight),
        "bias": tensor_list(model.bias),
        "output": tensor_list(output),
        "expected": [[2.0], [2.0], [5.0]],
        "max_abs_error": float((output - torch.tensor([[2.0], [2.0], [5.0]])).abs().max()),
    }
    write_json(RESULTS / "baseline_result.json", payload)
    print(payload)
    return payload


if __name__ == "__main__":
    main()
