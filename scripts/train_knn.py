#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.datasets.molecule3d import iter_manifest_records
from src.models.knn_template import KNNTemplateModel


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
    parser = argparse.ArgumentParser(description="Train a k-NN template model and save a checkpoint.")
    parser.add_argument("--manifest", type=str, required=True, help="Path to *_manifest.json from prepare_data.py.")
    parser.add_argument("--atom-count", type=int, default=None, help="Optional atom count filter.")
    parser.add_argument("--max-train", type=int, default=0, help="Max training samples (0 means all).")
    parser.add_argument("--k", type=int, default=1)
    parser.add_argument("--fp-radius", type=int, default=2)
    parser.add_argument("--fp-bits", type=int, default=2048)
    parser.add_argument("--use-chirality", action="store_true")
    parser.add_argument("--output", type=str, default="checkpoints/knn.pkl", help="Path to save checkpoint.")
    args = parser.parse_args()

    train_start = time.time()
    records = _load_records(
        manifest=args.manifest,
        atom_count=args.atom_count,
        max_samples=args.max_train,
    )
    model = KNNTemplateModel(
        k=args.k,
        fp_radius=args.fp_radius,
        fp_bits=args.fp_bits,
        use_chirality=bool(args.use_chirality),
    ).fit(records)
    train_elapsed = time.time() - train_start

    ckpt_path = Path(args.output).expanduser().resolve()
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model,
        "config": {
            "model": "knn",
            "k": int(args.k),
            "fp_radius": int(args.fp_radius),
            "fp_bits": int(args.fp_bits),
            "use_chirality": bool(args.use_chirality),
            "atom_count": args.atom_count,
        },
    }
    with ckpt_path.open("wb") as fh:
        pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)

    out = {
        "manifest": str(Path(args.manifest).expanduser().resolve()),
        "train_samples": len(records),
        "atom_count": args.atom_count,
        "k": args.k,
        "fp_radius": args.fp_radius,
        "fp_bits": args.fp_bits,
        "use_chirality": bool(args.use_chirality),
        "train_elapsed_sec": round(train_elapsed, 3),
        "checkpoint": str(ckpt_path),
    }
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
