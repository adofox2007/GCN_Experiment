from __future__ import annotations

import os
import platform
import sys
import winreg
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
import dgl, matplotlib, networkx, numpy, pandas, pytest, torch
import docx
from experiments.common import RESULTS, SEED, setup


def cpu_name():
    try:
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as key:
            return winreg.QueryValueEx(key,"ProcessorNameString")[0].strip()
    except OSError: return platform.processor() or "Unavailable"


def main():
    setup(); versions={
        "Operating system":platform.platform(),"CPU":cpu_name(),"Logical CPU count":os.cpu_count(),
        "Python":sys.version.replace("\n"," "),"Python executable":sys.executable,"PyTorch":torch.__version__,
        "DGL":dgl.__version__,"NumPy":numpy.__version__,"Pandas":pandas.__version__,"Matplotlib":matplotlib.__version__,
        "NetworkX":networkx.__version__,"Pytest":pytest.__version__,"python-docx":docx.__version__,
        "Experiment device":"CPU (CUDA_VISIBLE_DEVICES is empty; all tensors created on CPU)","CUDA available in base build":torch.cuda.is_available(),
        "Random seed":SEED,"PyTorch intra-op threads":torch.get_num_threads(),"DGL backend":dgl.backend.backend_name,
    }
    text="\n".join(f"{k}: {v}" for k,v in versions.items())+"\n"
    (RESULTS/"environment.txt").write_text(text,encoding="utf-8"); print(text); return versions


if __name__=="__main__": main()
