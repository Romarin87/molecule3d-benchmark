#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
from rdkit import Chem

from src.datasets.molecule3d import find_molecule3d_dir, iter_rows, parse_dataset_sdf


def _download_molecule3d_snapshot() -> Path:
    try:
        from huggingface_hub import snapshot_download
    except Exception as exc:  # pragma: no cover - optional dependency
        raise ImportError("huggingface_hub is required to auto-download Molecule3D.") from exc

    snapshot_path = snapshot_download(repo_id="maomlab/Molecule3D", repo_type="dataset")
    root = Path(snapshot_path)
    target = root / "Molecule3D_random_split"
    if target.exists():
        return target.resolve()

    matches = list(root.rglob("Molecule3D_random_split"))
    if matches:
        return matches[0].resolve()

    raise FileNotFoundError("Downloaded dataset but no Molecule3D_random_split folder found.")


def _iter_smiles_sdf_from_parquet(
    data_dir: Path,
    split: str,
    batch_size: int,
    atom_count: int | None,
    allowed_elements: set[str] | None,
):
    for row in iter_rows(data_dir, split=split, columns=["SMILES", "sdf"], batch_size=batch_size):
        smiles = row["SMILES"]
        sdf = row["sdf"]
        if not isinstance(smiles, str) or not isinstance(sdf, str):
            continue
        if atom_count is not None or allowed_elements is not None:
            mol = parse_dataset_sdf(sdf)
            if mol is None:
                continue
            if atom_count is not None and mol.GetNumAtoms() != atom_count:
                continue
            if allowed_elements is not None and not _mol_has_only_elements(mol, allowed_elements):
                continue
        yield smiles, sdf


def _iter_smiles_sdf_from_sdf(
    sdf_path: str,
    smiles_prop: str,
    atom_count: int | None,
    allowed_elements: set[str] | None,
):
    supplier = Chem.SDMolSupplier(str(Path(sdf_path).expanduser()), removeHs=False)
    for mol in supplier:
        if mol is None:
            continue
        if atom_count is not None and mol.GetNumAtoms() != atom_count:
            continue
        if allowed_elements is not None and not _mol_has_only_elements(mol, allowed_elements):
            continue
        if mol.HasProp(smiles_prop):
            smiles = mol.GetProp(smiles_prop)
        else:
            smiles = Chem.MolToSmiles(Chem.RemoveHs(mol))
        if not smiles:
            continue
        sdf = Chem.MolToMolBlock(mol)
        yield smiles, sdf


def _resolve_records(args: argparse.Namespace):
    if args.sdf_path:
        return _iter_smiles_sdf_from_sdf(
            sdf_path=args.sdf_path,
            smiles_prop=args.smiles_prop,
            atom_count=args.atom_count,
            allowed_elements=args.allowed_elements,
        )

    try:
        data_dir = find_molecule3d_dir(args.data_dir)
    except FileNotFoundError:
        if args.no_download:
            raise
        print("Molecule3D cache not found. Downloading from HuggingFace...")
        data_dir = _download_molecule3d_snapshot()

    return _iter_smiles_sdf_from_parquet(
        data_dir=data_dir,
        split=args.split,
        batch_size=args.batch_size,
        atom_count=args.atom_count,
        allowed_elements=args.allowed_elements,
    )


def _parse_allowed_elements(raw: str | None) -> list[str] | None:
    if raw is None:
        return None
    raw = raw.strip()
    if not raw:
        return None
    if "," not in raw and " " not in raw and raw.isalpha() and raw.upper() == raw:
        tokens = list(raw)
    else:
        tokens = re.split(r"[,\s]+", raw)
    elements: list[str] = []
    for token in tokens:
        token = token.strip()
        if not token:
            continue
        symbol = token[0].upper() + token[1:].lower()
        if symbol not in elements:
            elements.append(symbol)
    return elements or None


def _mol_has_only_elements(mol: Chem.Mol, allowed: set[str]) -> bool:
    for atom in mol.GetAtoms():
        if atom.GetSymbol() not in allowed:
            return False
    return True


def _write_shard(
    out_dir: Path,
    prefix: str,
    shard_idx: int,
    smiles_batch: list[str],
    sdf_batch: list[str],
) -> dict[str, object]:
    arr_smiles = np.array(smiles_batch, dtype=object)
    arr_sdf = np.array(sdf_batch, dtype=object)
    out_path = out_dir / f"{prefix}_shard{shard_idx:03d}.npz"
    np.savez_compressed(out_path, smiles=arr_smiles, sdf=arr_sdf)
    return {"path": str(out_path), "count": int(arr_smiles.shape[0])}


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare Molecule3D SMILES+SDF into numpy shards.")
    parser.add_argument("--data-dir", type=str, default=None, help="Path to Molecule3D_random_split dir.")
    parser.add_argument("--split", choices=["train", "validation", "test"], default="train")
    parser.add_argument("--sdf-path", type=str, default=None, help="Optional SDF path (bypasses parquet).")
    parser.add_argument("--smiles-prop", type=str, default="SMILES", help="SDF property name for SMILES.")
    parser.add_argument("--atom-count", type=int, default=None, help="Only keep molecules with this atom count.")
    parser.add_argument(
        "--allowed-elements",
        type=str,
        default=None,
        help="Comma/space-separated element symbols to keep (e.g., C,H,O,N).",
    )
    parser.add_argument("--max-samples", type=int, default=0, help="Max samples to keep (0 means all).")
    parser.add_argument("--batch-size", type=int, default=2048, help="Parquet batch size.")
    parser.add_argument("--out-dir", type=str, default="data/processed")
    parser.add_argument("--prefix", type=str, default=None, help="Output file prefix.")
    parser.add_argument("--shard-size", type=int, default=50000, help="Samples per shard (0 means no sharding).")
    parser.add_argument("--no-download", action="store_true", help="Disable auto-download when cache is missing.")
    args = parser.parse_args()

    allowed_elements = _parse_allowed_elements(args.allowed_elements)
    allowed_set = set(allowed_elements) if allowed_elements else None
    args.allowed_elements = allowed_set

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.prefix
    if prefix is None:
        base = "sdf" if args.sdf_path else args.split
        if args.atom_count is not None:
            base = f"{base}_atoms{args.atom_count}"
        if allowed_elements is not None:
            elem_tag = "".join(sorted(allowed_elements))
            base = f"{base}_elems{elem_tag}"
        prefix = f"{base}_smiles_sdf"

    shard_size = args.shard_size if args.shard_size > 0 else 1_000_000_000

    records = _resolve_records(args)
    smiles_batch: list[str] = []
    sdf_batch: list[str] = []
    kept = 0
    skipped = 0
    shard_idx = 0
    manifest = {
        "fields": ["smiles", "sdf"],
        "atom_count": args.atom_count,
        "allowed_elements": allowed_elements,
        "source": "sdf" if args.sdf_path else "molecule3d",
        "split": None if args.sdf_path else args.split,
        "total_kept": 0,
        "total_skipped": 0,
        "shards": [],
    }

    for smiles, sdf in records:
        if args.max_samples > 0 and kept >= args.max_samples:
            break
        if not smiles or not sdf:
            skipped += 1
            continue

        smiles_batch.append(smiles)
        sdf_batch.append(sdf)
        kept += 1

        if len(smiles_batch) >= shard_size:
            shard_info = _write_shard(out_dir, prefix, shard_idx, smiles_batch, sdf_batch)
            manifest["shards"].append(shard_info)
            shard_idx += 1
            smiles_batch = []
            sdf_batch = []

    if smiles_batch:
        shard_info = _write_shard(out_dir, prefix, shard_idx, smiles_batch, sdf_batch)
        manifest["shards"].append(shard_info)

    manifest["total_kept"] = kept
    manifest["total_skipped"] = skipped
    manifest_path = out_dir / f"{prefix}_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(json.dumps({"out_dir": str(out_dir), "kept": kept, "skipped": skipped, "manifest": str(manifest_path)}, indent=2))


if __name__ == "__main__":
    main()
