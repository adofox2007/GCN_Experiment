from __future__ import annotations

import json, subprocess, sys
from pathlib import Path
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from experiments.common import RESULTS, write_json


def main():
    demo=json.loads((RESULTS/"manual_example.json").read_text(encoding="utf-8"))
    comp=pd.read_csv(RESULTS/"tables"/"numerical_comparison.csv"); bench=pd.read_csv(RESULTS/"tables"/"benchmark.csv")
    test=pd.read_csv(RESULTS/"tables"/"test_results.csv"); summary={
        "tests":{"total":int(len(test)),"passed":int((test.outcome=="passed").sum()),"failed":int((test.outcome=="failed").sum()),"skipped":int((test.outcome=="skipped").sum())},
        "manual_example_outputs":demo["outputs"],"manual_example_max_error":max(demo["max_abs_errors"].values()),
        "max_numerical_error":float(comp.max_abs_error.max()),
        "max_input_gradient_error":float(comp.max_input_grad_error.max()),"max_weight_gradient_error":float(comp.max_weight_grad_error.max()),
        "all_comparisons_passed":bool(comp.passed.all()),"figures":len(list((RESULTS/"figures").glob("*.png"))),
        "benchmark_fastest_by_size":[],
    }
    core=bench[bench.scope=="core_excludes_preprocessing"]
    for nodes,group in core.groupby("nodes"):
        row=group.loc[group.median_ms.idxmin()]; summary["benchmark_fastest_by_size"].append({"nodes":int(nodes),"implementation":row.implementation,"median_ms":float(row.median_ms)})
    try: summary["git_head"]=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip()
    except Exception: summary["git_head"]="not committed yet"
    write_json(RESULTS/"experiment_summary.json",summary); print(json.dumps(summary,ensure_ascii=False,indent=2)); return summary


if __name__=="__main__": main()
