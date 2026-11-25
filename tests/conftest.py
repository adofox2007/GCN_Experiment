from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def pytest_configure():
    random.seed(20261125)
    np.random.seed(20261125)
    torch.manual_seed(20261125)
    torch.set_num_threads(1)
