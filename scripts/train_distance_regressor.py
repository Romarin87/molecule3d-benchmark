#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

from rdkit import Chem

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.datasets.molecule3d import iter_manifest_records
from src.models.distance_regressor import FeatureConfig
from src.training.baselines import train_distance_regressor


DEFAULT_ELEMENTS = (6, 1, 8, 7)


def _parse_elements(raw: str | None) -> tuple[int, ...]:
    if raw is None:
        return DEFAULT_ELEMENTS
    raw = raw.strip()
    if not raw:
        return DEFAULT_ELEMENTS
    if "," not in raw and " " not in raw and raw.isalpha() and raw.upper() == raw:
        tokens = list(raw)
    else:
        tokens = re.split(r"[,\s]+", raw)

    pt = Chem.GetPeriodicTable()
    nums: list[int] = []
    for token in tokens:
        token = token.strip()
        if not token:
            continue
        if token.isdigit():
            z = int(token)
        else:
            symbol = token[0].upper() + token[1:].lower()
            z = int(pt.GetAtomicNumber(symbol))
        if z <= 0:
            raise ValueError(f"Unknown element token: {token}")
        if z not in nums:
            nums.append(z)
    return tuple(nums) if nums else DEFAULT_ELEMENTS


def _load_records(
    manifest: str,
    atom_count: int | None,
    max_samples: int,
):
    records = list(iter_manifest_records(manifest, atom_count=atom_count, max_samples=max_samples))
    if not records:
        raise RuntimeError("No records found for manifest.")
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a distance regressor and save a checkpoint.")
    parser.add_argument("--manifest", type=str, required=True, help="Path to *_manifest.json from prepare_data.py.")
    parser.add_argument("--atom-count", type=int, required=True, help="Required atom count filter.")
    parser.add_argument("--elements", type=str, default=None, help="Element symbols or atomic numbers.")
    parser.add_argument("--max-degree", type=int, default=6)
    parser.add_argument("--max-train", type=int, default=0, help="Max training samples (0 means all).")
    parser.add_argument("--model", type=str, default="rf", choices=["ridge", "mlp", "rf", "xgb"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=str, default="checkpoints/distance_regressor.joblib")
    args = parser.parse_args()

    elements = _parse_elements(args.elements)
    cfg = FeatureConfig(atom_count=int(args.atom_count), elements=elements, max_degree=int(args.max_degree))
    records = _load_records(
        manifest=args.manifest,
        atom_count=args.atom_count,
        max_samples=args.max_train,
    )

    train_start = time.time()
    model = train_distance_regressor(records, cfg=cfg, model_name=args.model, seed=args.seed)
    train_elapsed = time.time() - train_start

    ckpt_path = Path(args.output).expanduser().resolve()
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(ckpt_path))

    out = {
        "manifest": str(Path(args.manifest).expanduser().resolve()),
        "train_samples": len(records),
        "atom_count": args.atom_count,
        "elements": list(elements),
        "max_degree": args.max_degree,
        "model": args.model,
        "seed": args.seed,
        "train_elapsed_sec": round(train_elapsed, 3),
        "checkpoint": str(ckpt_path),
    }
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
