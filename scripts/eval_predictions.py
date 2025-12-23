#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from rdkit import Chem

try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover - optional dependency
    plt = None

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.datasets.molecule3d import iter_manifest_records
from src.metrics.rmsd import best_rmsd, rmsd_stats


def _coerce_str(value: object) -> str | None:
    if isinstance(value, str):
        return value
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="ignore")
    return None


def _parse_sdf(sdf: str) -> Chem.Mol | None:
    mol = Chem.MolFromMolBlock("Molecule\n" + sdf, removeHs=False)
    if mol is not None:
        return mol
    mol = Chem.MolFromMolBlock(sdf, removeHs=False)
    return mol


def _plot_rmsd(rmsds: list[float], out_path: Path) -> None:
    if plt is None or not rmsds:
        return
    arr = np.asarray(rmsds, dtype=np.float64)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].hist(arr, bins=50, color="#4C78A8", alpha=0.85)
    axes[0].set_xlabel("RMSD (A)")
    axes[0].set_ylabel("Count")
    axes[0].set_title("RMSD Histogram")

    sorted_arr = np.sort(arr)
    cdf = np.arange(1, len(sorted_arr) + 1) / float(len(sorted_arr))
    axes[1].plot(sorted_arr, cdf, color="#F58518")
    axes[1].set_xlabel("RMSD (A)")
    axes[1].set_ylabel("CDF")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].grid(alpha=0.3)
    axes[1].set_title("RMSD CDF")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _iter_records(
    manifest: str,
    atom_count: int | None,
    max_samples: int,
):
    records = iter_manifest_records(manifest, atom_count=atom_count, max_samples=max_samples)
    for rec in records:
        yield rec


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate predicted structures with RDKit symmetry-aware RMSD.")
    parser.add_argument("--predictions", type=str, required=True, help="Path to predictions .npz.")
    parser.add_argument("--manifest", type=str, required=True, help="Path to *_manifest.json from prepare_data.py.")
    parser.add_argument("--atom-count", type=int, default=None, help="Optional atom count filter.")
    parser.add_argument("--max-eval", type=int, default=0, help="Max eval samples (0 means all).")
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON path for metrics (default: <predictions>_metrics.json).",
    )
    parser.add_argument(
        "--plot",
        type=str,
        default=None,
        help="Output path for RMSD plot (default: <predictions>_rmsd.png).",
    )
    parser.add_argument("--no-plot", action="store_true", help="Disable RMSD plotting.")
    args = parser.parse_args()

    pred_path = Path(args.predictions).expanduser().resolve()
    data = np.load(pred_path, allow_pickle=True)
    pred_smiles = data["smiles"]
    pred_sdf = data["sdf_pred"]

    if pred_smiles.shape[0] != pred_sdf.shape[0]:
        raise RuntimeError("Prediction arrays have mismatched lengths.")

    total_pred = int(pred_smiles.shape[0])
    max_eval = args.max_eval if args.max_eval > 0 else total_pred
    max_eval = min(max_eval, total_pred)

    rmsds: list[float] = []
    evaluated = 0
    skipped = 0
    mismatched = 0
    start = time.time()

    records = _iter_records(
        manifest=args.manifest,
        atom_count=args.atom_count,
        max_samples=max_eval,
    )

    for idx, rec in enumerate(records):
        if idx >= max_eval:
            break
        smi_pred = _coerce_str(pred_smiles[idx])
        sdf_pred = _coerce_str(pred_sdf[idx])
        if not smi_pred or not sdf_pred:
            skipped += 1
            continue
        if smi_pred != rec.smiles:
            mismatched += 1
            skipped += 1
            continue

        pred_mol = _parse_sdf(sdf_pred)
        if pred_mol is None:
            skipped += 1
            continue

        gt = Chem.RemoveHs(rec.mol3d)
        pred = Chem.RemoveHs(pred_mol)
        if pred.GetNumAtoms() != gt.GetNumAtoms():
            skipped += 1
            continue

        rmsds.append(best_rmsd(pred, gt))
        evaluated += 1

    elapsed = time.time() - start
    plot_path = None
    if not args.no_plot and plt is not None and rmsds:
        plot_path = args.plot
        if plot_path is None:
            plot_path = str(pred_path.with_suffix("").with_name(pred_path.stem + "_rmsd.png"))
        _plot_rmsd(rmsds, Path(plot_path))

    output_path = args.output
    if output_path is None:
        output_path = str(pred_path.with_suffix("").with_name(pred_path.stem + "_metrics.json"))

    out = {
        "predictions": str(pred_path),
        "manifest": str(Path(args.manifest).expanduser().resolve()),
        "pred_count": total_pred,
        "evaluated": evaluated,
        "skipped": skipped,
        "mismatched": mismatched,
        "elapsed_sec": round(elapsed, 3),
        "plot": plot_path,
        "metrics": rmsd_stats(rmsds),
    }
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
