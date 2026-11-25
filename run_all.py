from __future__ import annotations

import json, os, subprocess, sys
from pathlib import Path

ROOT=Path(__file__).resolve().parent


def run(label, args, log=None):
    print(f"\n{'='*72}\n{label}\n{'='*72}")
    env=os.environ.copy(); env["DGLBACKEND"]="pytorch"; env["CUDA_VISIBLE_DEVICES"]=""
    result=subprocess.run(args,cwd=ROOT,env=env,text=True,encoding="utf-8",errors="replace",capture_output=bool(log))
    if log:
        Path(log).parent.mkdir(parents=True,exist_ok=True); Path(log).write_text(result.stdout+result.stderr,encoding="utf-8")
        print(result.stdout); print(result.stderr,file=sys.stderr)
    if result.returncode: raise SystemExit(f"{label} failed with exit code {result.returncode}")


def main():
    py=sys.executable
    run("1. Environment check",[py,"experiments/collect_environment.py"])
    run("2. DGL reference baseline",[py,"experiments/run_baseline.py"])
    run("3. Hand-check example",[py,"experiments/run_demo.py"])
    report=ROOT/"results"/"pytest_report.xml"; log=ROOT/"results"/"logs"/"pytest.log"
    run("4. Automated tests",[py,"-m","pytest","-q",f"--junitxml={report}"],log)
    run("4b. Test result table",[py,"tools/pytest_to_csv.py",str(report),str(ROOT/"results"/"tables"/"test_results.csv")])
    run("5. Numerical comparison",[py,"experiments/run_comparison.py"])
    run("6. CPU benchmark",[py,"experiments/run_benchmark.py"])
    run("7. Visualizations",[py,"experiments/visualize.py"])
    run("8. Result summary",[py,"experiments/summarize.py"])
    run("9. Word report",[py,"experiments/generate_report.py"])
    summary=json.loads((ROOT/"results"/"experiment_summary.json").read_text(encoding="utf-8"))
    print("\nFINAL ACCEPTANCE SUMMARY")
    print(json.dumps(summary,ensure_ascii=False,indent=2))
    return 0


if __name__=="__main__": raise SystemExit(main())
