from __future__ import annotations

import json
import os
import random
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
SEED = 20261125


def setup(seed: int = SEED) -> None:
    os.environ.setdefault("DGLBACKEND", "pytorch")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def tensor_list(tensor: torch.Tensor):
    return tensor.detach().cpu().tolist()
