#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description="Read a SMILES+SDF npz shard.")
    parser.add_argument("--npz", type=str, required=True, help="Path to the .npz shard.")
    parser.add_argument("--limit", type=int, default=3, help="Number of samples to print.")
    args = parser.parse_args()

    path = Path(args.npz).expanduser().resolve()
    data = np.load(path, allow_pickle=True)

    smiles = data["smiles"]
    sdf = data["sdf"]
    n = min(len(smiles), len(sdf), max(args.limit, 0))

    print(f"path: {path}")
    print(f"count: {len(smiles)}")
    print(f"keys: {list(data.keys())}")
    print("---")
    for i in range(n):
        smi = smiles[i]
        block = sdf[i]
        smi_preview = smi if len(smi) <= 120 else smi[:120] + "..."
        lines = block.splitlines()
        head = "\n".join(lines[:8])
        print(f"[{i}] SMILES: {smi_preview}")
        print(head)
        print("---")


if __name__ == "__main__":
    main()
