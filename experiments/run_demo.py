from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from experiments.common import RESULTS, setup, tensor_list, write_json
from src import ManualGCN, MessagePassingGCN, ReferenceGCN


def main():
    setup()
    edges = ([0, 1], [1, 0])
    features = torch.tensor([[1.0], [3.0], [5.0]])
    expected = torch.tensor([[2.0], [2.0], [5.0]])
    outputs = {}
    dense = ManualGCN(1, 1, bias=True)
    with torch.no_grad():
        dense.weight.fill_(1.0)
        dense.bias.zero_()
    adjacency, degree, normalized, out_degree, in_degree = dense.matrices(3, edges)
    for name, model in {
        "manual_dense": dense,
        "message_passing": MessagePassingGCN(1, 1, bias=True),
        "dgl_reference": ReferenceGCN(1, 1, bias=True),
    }.items():
        with torch.no_grad():
            model.weight.fill_(1.0)
            model.bias.zero_()
        outputs[name] = model(3, edges, features)
    payload = {
        "raw_edges": [[0, 1], [1, 0]],
        "shape_convention": {"A": [3, 3], "H": [3, 1], "W": [1, 1], "b": [1], "M": [3, 1], "H_out": [3, 1]},
        "features": tensor_list(features),
        "weight": [[1.0]], "bias": [0.0], "activation": None,
        "adjacency_after_strict_a_plus_i": tensor_list(adjacency),
        "degree_matrix": tensor_list(degree),
        "out_degree": tensor_list(out_degree), "in_degree": tensor_list(in_degree),
        "normalized_adjacency": tensor_list(normalized),
        "aggregated_features": tensor_list(normalized @ features),
        "outputs": {key: tensor_list(value) for key, value in outputs.items()},
        "expected_output": tensor_list(expected),
        "max_abs_errors": {key: float((value - expected).abs().max()) for key, value in outputs.items()},
    }
    write_json(RESULTS / "manual_example.json", payload)
    log_lines = [
        "Minimal hand-check example", f"A_tilde [3,3] =\n{adjacency}", f"D_tilde [3,3] =\n{degree}",
        f"S [3,3] =\n{normalized}", f"H [3,1] =\n{features}", f"M=S@H [3,1] =\n{normalized @ features}",
        *[f"{name} output [3,1] =\n{value}\nmax error={payload['max_abs_errors'][name]:.3e}" for name, value in outputs.items()],
    ]
    (RESULTS / "logs" / "manual_example.log").write_text("\n\n".join(log_lines), encoding="utf-8")
    print("\n".join(log_lines))
    return payload


if __name__ == "__main__":
    main()
