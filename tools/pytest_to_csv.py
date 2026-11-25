from __future__ import annotations

import csv, sys
from pathlib import Path
from xml.etree import ElementTree


def main(report_path, output_path):
    root=ElementTree.parse(report_path).getroot(); suite=root if root.tag=="testsuite" else root.find("testsuite"); rows=[]
    for test in suite.findall("testcase"):
        if test.find("failure") is not None or test.find("error") is not None: outcome="failed"
        elif test.find("skipped") is not None: outcome="skipped"
        else: outcome="passed"
        rows.append({"test":f"{test.get('classname')}::{test.get('name')}","outcome":outcome,"duration_seconds":float(test.get("time","0"))})
    with Path(output_path).open("w",newline="",encoding="utf-8-sig") as handle:
        writer=csv.DictWriter(handle,fieldnames=["test","outcome","duration_seconds"]); writer.writeheader(); writer.writerows(rows)
    return 0 if all(row["outcome"]!="failed" for row in rows) else 1


if __name__=="__main__": raise SystemExit(main(sys.argv[1],sys.argv[2]))
