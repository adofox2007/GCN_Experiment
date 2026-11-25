from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

from experiments.common import RESULTS, setup

COLORS = ["#2F5597", "#70AD47", "#ED7D31"]


def save(name):
    plt.tight_layout(); plt.savefig(RESULTS/"figures"/name, dpi=220, bbox_inches="tight"); plt.close()


def draw_graph(edges, loops, name, title):
    graph=nx.MultiDiGraph(); graph.add_nodes_from(range(3)); graph.add_edges_from(zip(*edges))
    if loops: graph.add_edges_from((i,i) for i in range(3))
    pos={0:(0,1),1:(1,1),2:(.5,0)}
    plt.figure(figsize=(5.2,3.8)); nx.draw_networkx_nodes(graph,pos,node_color="#D9EAF7",edgecolors="#2F5597",node_size=1100)
    nx.draw_networkx_labels(graph,pos,font_size=12); nx.draw_networkx_edges(graph,pos,edge_color="#666666",arrows=True,
        connectionstyle="arc3,rad=0.08",min_source_margin=18,min_target_margin=18)
    plt.title(title); plt.axis("off"); save(name)


def heatmap(matrix, name, title, vmax=None):
    plt.figure(figsize=(4.8,4)); image=plt.imshow(matrix,cmap="Blues",vmin=0,vmax=vmax)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]): plt.text(j,i,f"{matrix[i,j]:.2g}",ha="center",va="center")
    plt.xticks(range(matrix.shape[1])); plt.yticks(range(matrix.shape[0])); plt.xlabel("Source node"); plt.ylabel("Destination node")
    plt.title(title); plt.colorbar(image,fraction=.046); save(name)


def main():
    setup(); demo=json.loads((RESULTS/"manual_example.json").read_text(encoding="utf-8"))
    edges=([0,1],[1,0]); draw_graph(edges,False,"01_raw_graph.png","Raw graph with isolated node 2")
    draw_graph(edges,True,"02_graph_with_self_loops.png","Graph after strict A + I")
    adjacency=np.array(demo["adjacency_after_strict_a_plus_i"]); normalized=np.array(demo["normalized_adjacency"])
    heatmap(adjacency,"03_adjacency_heatmap.png","Adjacency matrix after A + I",vmax=1)
    heatmap(normalized,"04_normalized_adjacency_heatmap.png","Normalized adjacency matrix",vmax=1)
    names=list(demo["outputs"]); values=[np.array(demo["outputs"][n]).flatten() for n in names]
    x=np.arange(3); width=.24; plt.figure(figsize=(7,4))
    for i,(name,value) in enumerate(zip(names,values)): plt.bar(x+(i-1)*width,value,width,label=name,color=COLORS[i])
    plt.xticks(x,["Node 0","Node 1","Node 2"]); plt.ylabel("Output"); plt.title("Three implementations on hand-check example"); plt.legend(fontsize=8); save("05_output_comparison.png")
    comparison=pd.read_csv(RESULTS/"tables"/"numerical_comparison.csv"); manual=comparison[comparison.implementation!="DGL GraphConv"].copy()
    manual["display_error"]=manual.max_abs_error.clip(lower=1e-12)
    plt.figure(figsize=(8,4));
    for i,(impl,group) in enumerate(manual.groupby("implementation")): plt.semilogy(group["case"],group["display_error"],marker="o",label=impl,color=COLORS[i])
    plt.ylabel("Max absolute error (log scale)"); plt.xlabel("Case"); plt.xticks(rotation=15); plt.title("Numerical error versus DGL reference"); plt.legend(); save("06_numerical_error.png")
    bench=pd.read_csv(RESULTS/"tables"/"benchmark.csv"); core=bench[bench.scope=="core_excludes_preprocessing"]
    plt.figure(figsize=(7.5,4.5));
    for i,(impl,group) in enumerate(core.groupby("implementation")): plt.plot(group.nodes,group.median_ms,marker="o",label=impl,color=COLORS[i])
    plt.xlabel("Nodes"); plt.ylabel("Median forward computation time (ms)"); plt.title("CPU forward computation time (prebuilt inputs)"); plt.legend(); plt.grid(alpha=.2); save("07_runtime_comparison.png")
    print(f"generated 7 figures in {RESULTS/'figures'}")


if __name__ == "__main__": main()
