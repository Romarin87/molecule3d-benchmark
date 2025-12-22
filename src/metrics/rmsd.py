from __future__ import annotations

from typing import Iterable

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdMolAlign


def best_rmsd(pred: Chem.Mol, gt: Chem.Mol) -> float:
    return float(rdMolAlign.GetBestRMS(pred, gt))


def rmsd_stats(values: Iterable[float]) -> dict[str, float]:
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.size == 0:
        return {"count": 0.0, "mean": float("nan"), "median": float("nan")}

    mean = float(np.mean(arr))
    median = float(np.median(arr))
    p90 = float(np.percentile(arr, 90))
    p95 = float(np.percentile(arr, 95))
    min_v = float(np.min(arr))
    max_v = float(np.max(arr))
    frac_1 = float(np.mean(arr < 1.0))
    frac_2 = float(np.mean(arr < 2.0))
    return {
        "count": float(arr.size),
        "mean": mean,
        "median": median,
        "p90": p90,
        "p95": p95,
        "min": min_v,
        "max": max_v,
        "rmsd_lt_1": frac_1,
        "rmsd_lt_2": frac_2,
    }
