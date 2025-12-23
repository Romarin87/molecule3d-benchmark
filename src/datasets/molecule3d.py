from __future__ import annotations

import glob
import json
from pathlib import Path
from typing import Iterator

import numpy as np
import pyarrow.parquet as pq
from rdkit import Chem

from ..models.types import MoleculeRecord


HF_MOLECULE3D_SNAPSHOT_GLOB = (
    "~/.cache/huggingface/hub/datasets--maomlab--Molecule3D/snapshots/*/Molecule3D_random_split"
)


def find_molecule3d_dir(explicit_dir: str | None) -> Path:
    if explicit_dir is not None:
        p = Path(explicit_dir).expanduser().resolve()
        if not p.exists():
            raise FileNotFoundError(f"--data-dir not found: {p}")
        return p

    candidates = sorted(Path(p).resolve() for p in glob.glob(str(Path(HF_MOLECULE3D_SNAPSHOT_GLOB).expanduser())))
    if not candidates:
        raise FileNotFoundError(
            "Could not find cached Molecule3D parquet files.\n"
            f"Tried: {HF_MOLECULE3D_SNAPSHOT_GLOB}\n"
            "Fix: run the HuggingFace download once, or pass --data-dir pointing to a directory that contains "
            "train-*.parquet / validation-*.parquet / test-*.parquet."
        )
    return candidates[-1].resolve()


def list_parquet_files(data_dir: Path, split: str) -> list[Path]:
    files = sorted(data_dir.glob(f"{split}-*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet shards for split={split} under {data_dir}")
    return files


def iter_rows(
    data_dir: Path, split: str, columns: list[str], batch_size: int = 2048
) -> Iterator[dict[str, object]]:
    for parquet_path in list_parquet_files(data_dir, split):
        pf = pq.ParquetFile(parquet_path)
        for batch in pf.iter_batches(batch_size=batch_size, columns=columns, use_threads=True):
            table = batch.to_pydict()
            n = len(table[columns[0]])
            for i in range(n):
                yield {col: table[col][i] for col in columns}


def parse_dataset_sdf(sdf_str: str) -> Chem.Mol | None:
    mol = Chem.MolFromMolBlock("Molecule\n" + sdf_str, removeHs=False)
    if mol is not None:
        return mol

    mol = Chem.MolFromMolBlock(sdf_str, removeHs=False)
    if mol is not None:
        return mol
    return None


def iter_records(
    data_dir: Path,
    split: str,
    atom_count: int | None = None,
    max_samples: int = 0,
    batch_size: int = 2048,
) -> Iterator[MoleculeRecord]:
    count = 0
    for row in iter_rows(data_dir, split=split, columns=["SMILES", "sdf"], batch_size=batch_size):
        if max_samples > 0 and count >= max_samples:
            break
        smiles = row["SMILES"]
        sdf = row["sdf"]
        if not isinstance(smiles, str) or not isinstance(sdf, str):
            continue

        mol2d = Chem.MolFromSmiles(smiles)
        if mol2d is None:
            continue
        mol3d = parse_dataset_sdf(sdf)
        if mol3d is None:
            continue

        if atom_count is not None:
            if mol2d.GetNumAtoms() != atom_count or mol3d.GetNumAtoms() != atom_count:
                continue

        count += 1
        yield MoleculeRecord(smiles=smiles, mol3d=mol3d)


def _coerce_str(value: object) -> str | None:
    if isinstance(value, str):
        return value
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="ignore")
    return None


def iter_manifest_records(
    manifest_path: str | Path,
    atom_count: int | None = None,
    max_samples: int = 0,
) -> Iterator[MoleculeRecord]:
    manifest_path = Path(manifest_path).expanduser().resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    shards = manifest.get("shards", [])
    count = 0
    for shard in shards:
        shard_path = Path(shard.get("path", ""))
        if not shard_path.is_absolute():
            shard_path = (manifest_path.parent / shard_path).resolve()
        with np.load(shard_path, allow_pickle=True) as data:
            smiles_list = data["smiles"]
            sdf_list = data["sdf"]
            for smiles, sdf in zip(smiles_list, sdf_list):
                if max_samples > 0 and count >= max_samples:
                    return
                smi = _coerce_str(smiles)
                block = _coerce_str(sdf)
                if not smi or not block:
                    continue

                mol2d = Chem.MolFromSmiles(smi)
                if mol2d is None:
                    continue
                mol3d = parse_dataset_sdf(block)
                if mol3d is None:
                    continue

                if atom_count is not None:
                    if mol2d.GetNumAtoms() != atom_count or mol3d.GetNumAtoms() != atom_count:
                        continue

                count += 1
                yield MoleculeRecord(smiles=smi, mol3d=mol3d)


def load_records(
    data_dir: str | Path | None,
    split: str,
    atom_count: int | None = None,
    max_samples: int = 0,
    batch_size: int = 2048,
) -> list[MoleculeRecord]:
    resolved = find_molecule3d_dir(str(data_dir)) if data_dir is not None else find_molecule3d_dir(None)
    return list(
        iter_records(
            data_dir=resolved,
            split=split,
            atom_count=atom_count,
            max_samples=max_samples,
            batch_size=batch_size,
        )
    )


def iter_sdf_records(
    sdf_path: str | Path,
    smiles_prop: str = "SMILES",
    atom_count: int | None = None,
    max_samples: int = 0,
) -> Iterator[MoleculeRecord]:
    supplier = Chem.SDMolSupplier(str(Path(sdf_path).expanduser()), removeHs=False)
    count = 0
    for mol in supplier:
        if mol is None:
            continue
        if max_samples > 0 and count >= max_samples:
            break

        if atom_count is not None and mol.GetNumAtoms() != atom_count:
            continue

        if mol.HasProp(smiles_prop):
            smiles = mol.GetProp(smiles_prop)
        else:
            smiles = Chem.MolToSmiles(Chem.RemoveHs(mol))
        if not smiles:
            continue
        if mol.GetNumConformers() == 0:
            continue

        count += 1
        yield MoleculeRecord(smiles=smiles, mol3d=mol)
