#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
from rdkit import Chem

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.datasets.graph import record_to_graph
from src.datasets.molecule3d import iter_manifest_records
from src.models.chem_utils import init_coords_from_smiles, mol_from_smiles_coords
from src.models.distance_regressor import DistanceRegressorModel
from src.models.egnn import EGNN
from src.models.etkdg import ETKDGModel
from src.models.knn_template import KNNTemplateModel
from src.models.mpnn import MPNN

try:
    import torch
except Exception:  # pragma: no cover - optional dependency
    torch = None


def _iter_records(
    manifest: str,
    atom_count: int | None,
    max_samples: int,
):
    records = iter_manifest_records(manifest, atom_count=atom_count, max_samples=max_samples)
    for rec in records:
        yield rec


def _load_knn(path: Path) -> KNNTemplateModel:
    with path.open("rb") as fh:
        payload = pickle.load(fh)
    return payload.get("model", payload)


def _load_mpnn(path: Path, device: str) -> tuple[MPNN, dict]:
    payload = torch.load(path, map_location=device)
    config = payload.get("config")
    if not isinstance(config, dict) or config.get("model") != "mpnn":
        raise RuntimeError("Checkpoint model is not MPNN.")
    model = MPNN(
        node_feat_dim=int(config["node_feat_dim"]),
        edge_feat_dim=int(config.get("edge_feat_dim", 1)),
        hidden_dim=int(config["hidden_dim"]),
        num_layers=int(config["num_layers"]),
        pair_hidden_dim=int(config.get("pair_hidden_dim", 128)),
    ).to(device)
    model.load_state_dict(payload["model_state"])
    model.eval()
    return model, config


def _load_egnn(path: Path, device: str) -> tuple[EGNN, dict]:
    payload = torch.load(path, map_location=device)
    config = payload.get("config")
    if not isinstance(config, dict) or config.get("model") != "egnn":
        raise RuntimeError("Checkpoint model is not EGNN.")
    model = EGNN(
        node_feat_dim=int(config["node_feat_dim"]),
        edge_feat_dim=int(config.get("edge_feat_dim", 1)),
        hidden_dim=int(config["hidden_dim"]),
        num_layers=int(config["num_layers"]),
    ).to(device)
    model.load_state_dict(payload["model_state"])
    model.eval()
    return model, config


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict structures for any baseline/GNN model.")
    parser.add_argument(
        "--method",
        type=str,
        required=True,
        choices=["etkdg", "knn", "distance_regressor", "mpnn", "egnn"],
    )
    parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint path (not needed for ETKDG).")
    parser.add_argument("--manifest", type=str, required=True, help="Path to *_manifest.json from prepare_data.py.")
    parser.add_argument("--atom-count", type=int, default=None, help="Optional atom count filter.")
    parser.add_argument("--max-samples", type=int, default=0, help="Max samples to predict (0 means all).")
    parser.add_argument("--output", type=str, default=None, help="Output .npz path for predictions.")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--use-mmff", action="store_true", help="Enable MMFF for ETKDG/dist regressor.")
    parser.add_argument("--num-confs", type=int, default=10, help="ETKDG num conformers.")
    parser.add_argument("--seed", type=int, default=0, help="ETKDG random seed.")
    parser.add_argument("--max-iters", type=int, default=200, help="ETKDG MMFF max iterations.")
    args = parser.parse_args()

    method = args.method
    if method != "etkdg" and not args.checkpoint:
        raise ValueError("--checkpoint is required for non-ETKDG methods.")

    device = args.device
    if method in {"mpnn", "egnn"}:
        if torch is None:
            raise ImportError("PyTorch is required for GNN prediction.")
        if device != "cpu" and not torch.cuda.is_available():
            print("CUDA not available; falling back to CPU.", file=sys.stderr)
            device = "cpu"

    ckpt_path = Path(args.checkpoint).expanduser().resolve() if args.checkpoint else None
    atom_count = args.atom_count
    if method == "etkdg":
        model = ETKDGModel(
            num_confs=args.num_confs,
            random_seed=args.seed,
            use_mmff=bool(args.use_mmff),
            max_iters=args.max_iters,
        )
        config = {
            "model": "etkdg",
            "num_confs": int(args.num_confs),
            "seed": int(args.seed),
            "use_mmff": bool(args.use_mmff),
            "max_iters": int(args.max_iters),
        }
    elif method == "knn":
        model = _load_knn(ckpt_path)
        config = {"model": "knn"}
    elif method == "distance_regressor":
        model = DistanceRegressorModel.load(str(ckpt_path))
        config = {
            "model": "distance_regressor",
            "atom_count": model.cfg.atom_count,
            "elements": list(model.cfg.elements),
            "max_degree": model.cfg.max_degree,
        }
        if atom_count is None:
            atom_count = model.cfg.atom_count
    elif method == "mpnn":
        model, config = _load_mpnn(ckpt_path, device=device)
        if atom_count is None:
            atom_count = config.get("atom_count")
    elif method == "egnn":
        model, config = _load_egnn(ckpt_path, device=device)
        if atom_count is None:
            atom_count = config.get("atom_count")
    else:
        raise ValueError(f"Unknown method: {method}")

    out_path = args.output
    if out_path is None:
        tag = Path(args.manifest).stem.replace("_manifest", "")
        out_path = f"pred_{method}_{tag}.npz"
    out_path = str(Path(out_path).expanduser().resolve())

    pred_smiles: list[str] = []
    pred_sdf: list[str] = []
    processed = 0
    predicted = 0
    failed = 0
    start = time.time()

    records = _iter_records(
        manifest=args.manifest,
        atom_count=atom_count,
        max_samples=args.max_samples,
    )

    with torch.no_grad() if method in {"mpnn", "egnn"} else _nullcontext():
        for rec in records:
            processed += 1
            pred_smiles.append(rec.smiles)
            sdf_block = ""
            try:
                if method == "etkdg":
                    mol = model.predict(rec.smiles)
                elif method == "knn":
                    mol = model.predict(rec.smiles)
                elif method == "distance_regressor":
                    mol = model.predict(rec.smiles, mmff=bool(args.use_mmff))
                elif method == "mpnn":
                    sample = record_to_graph(
                        rec,
                        elements=tuple(config["elements"]),
                        max_degree=int(config["max_degree"]),
                        atom_count=config.get("atom_count"),
                    )
                    if sample is None:
                        raise RuntimeError("Failed to featurize record.")
                    node_feats, edge_index, edge_attr, _, _ = sample.to_torch(device=device)
                    pred_coords = model.predict_coords(node_feats, edge_index, edge_attr)
                    mol = mol_from_smiles_coords(sample.smiles, pred_coords)
                elif method == "egnn":
                    sample = record_to_graph(
                        rec,
                        elements=tuple(config["elements"]),
                        max_degree=int(config["max_degree"]),
                        atom_count=config.get("atom_count"),
                    )
                    if sample is None:
                        raise RuntimeError("Failed to featurize record.")
                    node_feats, edge_index, edge_attr, _, _ = sample.to_torch(device=device)
                    coords0 = torch.from_numpy(init_coords_from_smiles(sample.smiles)).to(
                        device=device, dtype=node_feats.dtype
                    )
                    coords = model(node_feats, edge_index, edge_attr=edge_attr, coords=coords0)
                    mol = mol_from_smiles_coords(sample.smiles, coords.cpu().numpy())
                else:
                    raise ValueError(f"Unknown method: {method}")
                sdf_block = Chem.MolToMolBlock(mol)
                predicted += 1
            except Exception:
                failed += 1
            pred_sdf.append(sdf_block)

    arr_smiles = np.array(pred_smiles, dtype=object)
    arr_sdf = np.array(pred_sdf, dtype=object)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, smiles=arr_smiles, sdf_pred=arr_sdf)

    elapsed = time.time() - start
    out = {
        "method": method,
        "checkpoint": str(ckpt_path) if ckpt_path is not None else None,
        "manifest": str(Path(args.manifest).expanduser().resolve()),
        "atom_count": atom_count,
        "processed": processed,
        "predicted": predicted,
        "failed": failed,
        "elapsed_sec": round(elapsed, 3),
        "output": out_path,
        "config": config,
    }
    print(json.dumps(out, indent=2))


class _nullcontext:
    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc, tb):
        return False


if __name__ == "__main__":
    main()
