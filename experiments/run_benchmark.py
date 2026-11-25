from __future__ import annotations

import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import dgl
import pandas as pd
import torch

from experiments.common import RESULTS, SEED, setup
from src import ManualGCN, MessagePassingGCN, ReferenceGCN
from src.graph_utils import degrees, normalized_adjacency, preprocess_edges, random_edges
from src.reference_gcn import copy_parameters


def measure(fn, warmup, repeats):
    for _ in range(warmup): fn()
    samples = []
    for _ in range(repeats):
        start = time.perf_counter_ns(); fn(); samples.append((time.perf_counter_ns()-start)/1e6)
    ordered = sorted(samples)
    q1, q3 = ordered[len(ordered)//4], ordered[(3*len(ordered))//4]
    return statistics.median(samples), q3-q1


def benchmark_size(n, raw_edges, fin=16, fout=16, repeats=30):
    edges = random_edges(n, raw_edges, True, SEED+n)
    x = torch.randn(n, fin)
    ref, dense, sparse = ReferenceGCN(fin, fout), ManualGCN(fin, fout), MessagePassingGCN(fin, fout)
    copy_parameters(ref, dense); copy_parameters(ref, sparse)
    processed = preprocess_edges(n, edges, "add")
    norm_adj = normalized_adjacency(processed)
    out_degree, in_degree = degrees(processed)
    edge_norm = out_degree.rsqrt()[processed.src] * in_degree.rsqrt()[processed.dst]
    dg = dgl.graph((processed.src, processed.dst), num_nodes=n)
    def dense_core(): return (norm_adj @ x) @ dense.weight + dense.bias
    def sparse_core():
        agg = x.new_zeros((n, fin)); agg.index_add_(0, processed.dst, x[processed.src]*edge_norm[:,None]); return agg@sparse.weight+sparse.bias
    def dgl_core(): return ref.conv(dg, x)
    functions = {("ManualGCN", "core_excludes_preprocessing"): dense_core,
                 ("MessagePassingGCN", "core_excludes_preprocessing"): sparse_core,
                 ("DGL GraphConv", "core_excludes_preprocessing"): dgl_core,
                 ("ManualGCN", "full_includes_preprocessing"): lambda: dense(n, edges, x),
                 ("MessagePassingGCN", "full_includes_preprocessing"): lambda: sparse(n, edges, x),
                 ("DGL GraphConv", "full_includes_preprocessing"): lambda: ref(n, edges, x)}
    rows=[]
    for (name, scope), fn in functions.items():
        median, iqr = measure(fn, warmup=5, repeats=repeats)
        rows.append({"implementation":name,"scope":scope,"nodes":n,"raw_edges":raw_edges,"processed_edges":raw_edges+n,
                     "in_features":fin,"out_features":fout,"threads":torch.get_num_threads(),"warmup":5,"repeats":repeats,
                     "median_ms":median,"iqr_ms":iqr})
    return rows


def main():
    setup(); rows=[]
    for n, e, r in ((10, 30, 50), (100, 500, 40), (500, 3000, 25)):
        rows.extend(benchmark_size(n,e,repeats=r))
    frame=pd.DataFrame(rows); frame.to_csv(RESULTS/"tables"/"benchmark.csv",index=False,encoding="utf-8-sig")
    print(frame.to_string(index=False)); return frame


if __name__ == "__main__": main()
