#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from rdkit import Chem
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.metrics.rmsd import best_rmsd, rmsd_stats
from src.models.etkdg import ETKDGModel


def _parse_sdf(sdf: str) -> Chem.Mol | None:
    mol = Chem.MolFromMolBlock("Molecule\n" + sdf, removeHs=False)
    if mol is not None:
        return mol
    mol = Chem.MolFromMolBlock(sdf, removeHs=False)
    return mol


def _plot_rmsd(rmsds: list[float], out_path: Path) -> None:
    if not rmsds:
        return
    arr = np.asarray(rmsds, dtype=np.float64)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].hist(arr, bins=50, color="#4C78A8", alpha=0.85)
    axes[0].set_xlabel("RMSD (A)")
    axes[0].set_ylabel("Count")
    axes[0].set_title("ETKDG RMSD Histogram")

    sorted_arr = np.sort(arr)
    cdf = np.arange(1, len(sorted_arr) + 1) / float(len(sorted_arr))
    axes[1].plot(sorted_arr, cdf, color="#F58518")
    axes[1].set_xlabel("RMSD (A)")
    axes[1].set_ylabel("CDF")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].grid(alpha=0.3)
    axes[1].set_title("ETKDG RMSD CDF")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _default_output_path(manifest_path: Path, suffix: str) -> str:
    stem = manifest_path.name.replace("_manifest.json", "")
    return str(manifest_path.with_name(f"{stem}{suffix}"))


def _write_log_header(log_fh, manifest_path: Path, args: argparse.Namespace) -> None:
    payload = {
        "type": "meta",
        "manifest": str(manifest_path),
        "max_samples": int(args.max_samples),
        "atom_count_filter": args.atom_count,
        "num_confs": args.num_confs,
        "seed": args.seed,
        "use_mmff": bool(args.use_mmff),
        "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    log_fh.write(json.dumps(payload, ensure_ascii=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate RDKit ETKDG baseline on SMILES+SDF shards.")
    parser.add_argument("--manifest", type=str, required=True, help="Path to *_manifest.json.")
    parser.add_argument("--max-samples", type=int, default=1000, help="Max samples to evaluate (0 means all).")
    parser.add_argument("--atom-count", type=int, default=None, help="Filter by atom count (optional).")
    parser.add_argument("--num-confs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--use-mmff", action="store_true", help="Enable MMFF post-optimization.")
    parser.add_argument(
        "--plot",
        type=str,
        default=None,
        help="Output path for RMSD plot (default: <manifest>_etkdg_rmsd.png).",
    )
    parser.add_argument(
        "--log",
        type=str,
        default=None,
        help="Output JSONL log path (default: <manifest>_etkdg_rmsd.jsonl).",
    )
    parser.add_argument("--no-log", action="store_true", help="Disable writing the JSONL log.")
    parser.add_argument("--log-smiles", action="store_true", help="Include SMILES strings in log entries.")
    args = parser.parse_args()

    manifest_path = Path(args.manifest).expanduser().resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    shards = manifest.get("shards", [])

    log_path = None
    log_fh = None
    if not args.no_log:
        log_path = args.log
        if log_path is None:
            log_path = _default_output_path(manifest_path, "_etkdg_rmsd.jsonl")
        log_path = str(Path(log_path).expanduser().resolve())
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        log_fh = open(log_path, "w", encoding="utf-8")
        _write_log_header(log_fh, manifest_path, args)

    model = ETKDGModel(num_confs=args.num_confs, random_seed=args.seed, use_mmff=args.use_mmff)

    rmsds: list[float] = []
    skipped = 0
    processed = 0
    start = time.time()
    total = None
    if shards and all("count" in shard for shard in shards):
        total = int(sum(int(shard.get("count", 0)) for shard in shards))
        if args.max_samples > 0:
            total = min(total, args.max_samples)

    pbar = tqdm(total=total, unit="mol")

    for shard_idx, shard in enumerate(shards):
        data = np.load(shard["path"], allow_pickle=True)
        smiles_list = data["smiles"]
        sdf_list = data["sdf"]
        for smiles, sdf in zip(smiles_list, sdf_list):
            processed += 1
            pbar.update(1)
            if processed % 200 == 0:
                pbar.set_postfix(evaluated=len(rmsds), skipped=skipped)
            if args.max_samples > 0 and len(rmsds) >= args.max_samples:
                break
            if not smiles or not sdf:
                skipped += 1
            else:
                mol_gt = _parse_sdf(str(sdf))
                if mol_gt is None:
                    skipped += 1
                elif args.atom_count is not None and mol_gt.GetNumAtoms() != args.atom_count:
                    skipped += 1
                else:
                    try:
                        pred = model.predict(str(smiles))
                    except Exception:
                        skipped += 1
                    else:
                        gt = Chem.RemoveHs(mol_gt)
                        if pred.GetNumAtoms() != gt.GetNumAtoms():
                            skipped += 1
                        else:
                            try:
                                rmsd = best_rmsd(pred, gt)
                                rmsds.append(rmsd)
                                if log_fh is not None:
                                    entry = {
                                        "type": "sample",
                                        "index": int(len(rmsds)),
                                        "processed": int(processed),
                                        "shard": int(shard_idx),
                                        "rmsd": float(rmsd),
                                        "atom_count": int(gt.GetNumAtoms()),
                                    }
                                    if args.log_smiles:
                                        entry["smiles"] = str(smiles)
                                    log_fh.write(json.dumps(entry, ensure_ascii=True) + "\n")
                            except Exception:
                                skipped += 1
        if args.max_samples > 0 and len(rmsds) >= args.max_samples:
            break

    pbar.set_postfix(evaluated=len(rmsds), skipped=skipped)
    pbar.close()
    elapsed = time.time() - start
    metrics = rmsd_stats(rmsds)
    plot_path = args.plot
    if plot_path is None:
        plot_path = _default_output_path(manifest_path, "_etkdg_rmsd.png")

    if rmsds:
        _plot_rmsd(rmsds, Path(plot_path))

    if log_fh is not None:
        log_fh.close()

    out = {
        "manifest": str(manifest_path),
        "evaluated": len(rmsds),
        "skipped": skipped,
        "elapsed_sec": round(elapsed, 3),
        "num_confs": args.num_confs,
        "seed": args.seed,
        "use_mmff": bool(args.use_mmff),
        "log": str(log_path) if log_fh is not None else None,
        "plot": str(plot_path) if rmsds else None,
        "metrics": metrics,
    }
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
