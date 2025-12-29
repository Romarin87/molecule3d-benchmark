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

from src.datasets.graph import iter_graph_samples
from src.datasets.molecule3d import iter_manifest_records
from src.training.gnn import train_egnn_transformer

try:
    import torch
except Exception:  # pragma: no cover - optional dependency
    torch = None


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


def _node_feat_dim(elements: tuple[int, ...], max_degree: int) -> int:
    return (len(elements) + 1) + (max_degree + 1) + 3


def _load_samples(
    manifest: str,
    atom_count: int | None,
    elements: tuple[int, ...],
    max_degree: int,
    max_samples: int,
):
    records = iter_manifest_records(manifest, atom_count=atom_count, max_samples=0)
    samples = list(
        iter_graph_samples(
            records,
            elements=elements,
            max_degree=max_degree,
            atom_count=atom_count,
            max_samples=max_samples,
        )
    )
    if not samples:
        raise RuntimeError("No graph samples found for manifest.")
    return samples


def main() -> None:
    parser = argparse.ArgumentParser(description="Train an EGNN+Transformer on Molecule3D and save a checkpoint.")
    parser.add_argument("--manifest", type=str, required=True, help="Path to *_manifest.json from prepare_data.py.")
    parser.add_argument("--atom-count", type=int, default=None, help="Optional atom count filter.")
    parser.add_argument("--elements", type=str, default=None, help="Element symbols or atomic numbers.")
    parser.add_argument("--max-degree", type=int, default=6)
    parser.add_argument("--max-train", type=int, default=0, help="Max training samples (0 means all).")
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--rbf-bins", type=int, default=16)
    parser.add_argument("--rbf-cutoff", type=float, default=5.0)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=1, help="Gradient accumulation batch size.")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output",
        type=str,
        default="checkpoints/egnn_transformer.pt",
        help="Path to save checkpoint.",
    )
    args = parser.parse_args()

    if torch is None:
        raise ImportError("PyTorch is required for EGNN+Transformer training.")

    elements = _parse_elements(args.elements)
    node_feat_dim = _node_feat_dim(elements, args.max_degree)
    device = args.device
    if device != "cpu" and not torch.cuda.is_available():
        print("CUDA not available; falling back to CPU.", file=sys.stderr)
        device = "cpu"

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    train_samples = _load_samples(
        manifest=args.manifest,
        atom_count=args.atom_count,
        elements=elements,
        max_degree=args.max_degree,
        max_samples=args.max_train,
    )

    train_start = time.time()
    model, loss_history = train_egnn_transformer(
        samples=train_samples,
        node_feat_dim=node_feat_dim,
        edge_feat_dim=1,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        dropout=args.dropout,
        rbf_bins=args.rbf_bins,
        rbf_cutoff=args.rbf_cutoff,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        device=device,
        return_loss_history=True,
    )
    train_elapsed = time.time() - train_start

    ckpt_path = Path(args.output).expanduser().resolve()
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    payload = {
        "model_state": model.state_dict(),
        "config": {
            "model": "egnn_transformer",
            "node_feat_dim": int(node_feat_dim),
            "edge_feat_dim": 1,
            "hidden_dim": int(args.hidden_dim),
            "num_layers": int(args.num_layers),
            "num_heads": int(args.num_heads),
            "dropout": float(args.dropout),
            "rbf_bins": int(args.rbf_bins),
            "rbf_cutoff": float(args.rbf_cutoff),
            "elements": list(elements),
            "max_degree": int(args.max_degree),
            "atom_count": args.atom_count,
        },
    }
    torch.save(payload, ckpt_path)

    out = {
        "manifest": str(Path(args.manifest).expanduser().resolve()),
        "train_samples": len(train_samples),
        "atom_count": args.atom_count,
        "elements": list(elements),
        "max_degree": args.max_degree,
        "epochs": args.epochs,
        "hidden_dim": args.hidden_dim,
        "num_layers": args.num_layers,
        "num_heads": args.num_heads,
        "dropout": args.dropout,
        "rbf_bins": args.rbf_bins,
        "rbf_cutoff": args.rbf_cutoff,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "param_count": int(param_count),
        "loss_history": loss_history,
        "device": device,
        "seed": args.seed,
        "train_elapsed_sec": round(train_elapsed, 3),
        "checkpoint": str(ckpt_path),
    }
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
