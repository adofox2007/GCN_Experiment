from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
import torch

from experiments.common import RESULTS, SEED, setup
from src import ManualGCN, MessagePassingGCN, ReferenceGCN
from src.graph_utils import adjacency_matrix, preprocess_edges, random_edges
from src.reference_gcn import copy_parameters


def compare_case(name, n, e, fin, fout, directed):
    edges = random_edges(n, e, directed, SEED + n + e + fin)
    x0 = torch.randn(n, fin)
    reference = ReferenceGCN(fin, fout, bias=True, activation=torch.tanh)
    dense = ManualGCN(fin, fout, bias=True, activation=torch.tanh)
    sparse = MessagePassingGCN(fin, fout, bias=True, activation=torch.tanh)
    copy_parameters(reference, dense); copy_parameters(reference, sparse)
    records, values = [], {}
    for model_name, model in (("ManualGCN", dense), ("MessagePassingGCN", sparse), ("DGL GraphConv", reference)):
        x = x0.clone().requires_grad_(True)
        output = model(n, edges, x)
        # A mean-scaled probe keeps gradient magnitude comparable across graph sizes,
        # so the FP32 tolerance measures implementation differences rather than N.
        loss = output.square().mean(); loss.backward()
        values[model_name] = (output.detach(), x.grad.detach(), model.weight.grad.detach())
    ref_out, ref_x_grad, ref_w_grad = values["DGL GraphConv"]
    for model_name, (output, x_grad, w_grad) in values.items():
        diff = (output - ref_out).abs()
        xdiff, wdiff = (x_grad-ref_x_grad).abs(), (w_grad-ref_w_grad).abs()
        records.append({
            "case": name, "implementation": model_name, "nodes": n, "raw_edges": e,
            "processed_edges": e+n, "directed": directed, "in_features": fin, "out_features": fout,
            "max_abs_error": float(diff.max()) if diff.numel() else 0.0,
            "mean_abs_error": float(diff.mean()) if diff.numel() else 0.0,
            "max_input_grad_error": float(xdiff.max()) if xdiff.numel() else 0.0,
            "max_weight_grad_error": float(wdiff.max()) if wdiff.numel() else 0.0,
            "nan_count": int(torch.isnan(output).sum()), "inf_count": int(torch.isinf(output).sum()),
            "passed": bool(torch.allclose(output, ref_out, atol=1e-6, rtol=1e-5) and torch.allclose(x_grad, ref_x_grad, atol=1e-6, rtol=1e-5) and torch.allclose(w_grad, ref_w_grad, atol=1e-6, rtol=1e-5)),
        })
    return records


def strategy_rows():
    n, edges = 3, ([0, 0, 0, 1, 1], [0, 1, 1, 0, 2])
    x = torch.tensor([[1.0], [2.0], [4.0]])
    rows = []
    for policy, dedup in (("none", False), ("add", False), ("exactly_one", False), ("add", True)):
        model = ManualGCN(1, 1, bias=False)
        with torch.no_grad(): model.weight.fill_(1.0)
        graph = preprocess_edges(n, edges, policy, dedup)
        adj = adjacency_matrix(graph)
        output = model(n, edges, x, policy, dedup)
        rows.append({"self_loop_policy": policy, "deduplicate": dedup, "processed_edges": graph.num_edges,
                     "adjacency": str(adj.tolist()), "in_degree": str(adj.sum(1).tolist()),
                     "out_degree": str(adj.sum(0).tolist()), "output": str(output.detach().flatten().tolist())})
    return rows


def main():
    setup()
    cases = [("small_scalar", 10, 24, 1, 1, False), ("directed_expand", 50, 180, 4, 8, True),
             ("medium_equal", 100, 500, 16, 16, False), ("large_compress", 500, 2500, 64, 32, True)]
    rows = []
    for case in cases: rows.extend(compare_case(*case))
    frame = pd.DataFrame(rows)
    frame.to_csv(RESULTS / "tables" / "numerical_comparison.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(strategy_rows()).to_csv(RESULTS / "tables" / "graph_strategy_comparison.csv", index=False, encoding="utf-8-sig")
    print(frame.to_string(index=False))
    return frame


if __name__ == "__main__": main()
