from __future__ import annotations

from typing import Iterable

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdDepictor
from rdkit.Geometry import Point3D


def canonical_atom_order(mol: Chem.Mol) -> list[int]:
    ranks = list(Chem.CanonicalRankAtoms(mol, breakTies=True))
    return [idx for idx, _ in sorted(enumerate(ranks), key=lambda kv: (kv[1], kv[0]))]


def one_hot(index: int, size: int) -> np.ndarray:
    vec = np.zeros(size, dtype=np.float32)
    if 0 <= index < size:
        vec[index] = 1.0
    return vec


def atom_features(atom: Chem.Atom, elements: Iterable[int], max_degree: int) -> np.ndarray:
    elements = tuple(elements)
    z = atom.GetAtomicNum()
    try:
        elem_idx = elements.index(z)
        elem = one_hot(elem_idx, len(elements) + 1)
    except ValueError:
        elem = one_hot(len(elements), len(elements) + 1)

    deg = int(atom.GetTotalDegree())
    deg = min(max(deg, 0), max_degree)
    deg_oh = one_hot(deg, max_degree + 1)

    aromatic = np.array([1.0 if atom.GetIsAromatic() else 0.0], dtype=np.float32)
    in_ring = np.array([1.0 if atom.IsInRing() else 0.0], dtype=np.float32)
    charge = np.array([float(atom.GetFormalCharge())], dtype=np.float32)
    return np.concatenate([elem, deg_oh, aromatic, in_ring, charge], axis=0)


def featurize_mol(
    mol2d: Chem.Mol, atom_count: int, elements: Iterable[int], max_degree: int
) -> tuple[np.ndarray, list[int]]:
    n = mol2d.GetNumAtoms()
    if n != atom_count:
        raise ValueError(f"expected {atom_count} atoms, got {n}")

    order = canonical_atom_order(mol2d)
    idx_to_pos = {atom_idx: pos for pos, atom_idx in enumerate(order)}

    atom_f = np.stack(
        [atom_features(mol2d.GetAtomWithIdx(atom_idx), elements, max_degree) for atom_idx in order], axis=0
    )

    adj = np.zeros((n, n), dtype=np.float32)
    for bond in mol2d.GetBonds():
        i = idx_to_pos[bond.GetBeginAtomIdx()]
        j = idx_to_pos[bond.GetEndAtomIdx()]
        bt = float(bond.GetBondTypeAsDouble())
        adj[i, j] = bt
        adj[j, i] = bt

    x = np.concatenate([atom_f.reshape(-1), adj.reshape(-1)], axis=0).astype(np.float32, copy=False)
    return x, order


def flatten_upper_triangle(dist: np.ndarray) -> np.ndarray:
    n = dist.shape[0]
    out = []
    for i in range(n):
        for j in range(i + 1, n):
            out.append(dist[i, j])
    return np.asarray(out, dtype=np.float32)


def vector_to_distance_matrix(vec: np.ndarray, n: int) -> np.ndarray:
    dist = np.zeros((n, n), dtype=np.float32)
    k = 0
    for i in range(n):
        for j in range(i + 1, n):
            d = float(vec[k])
            d = max(d, 0.0)
            dist[i, j] = d
            dist[j, i] = d
            k += 1
    return dist


def classical_mds(dist: np.ndarray, n_components: int = 3) -> np.ndarray:
    n = dist.shape[0]
    d2 = dist.astype(np.float64) ** 2
    j = np.eye(n, dtype=np.float64) - np.ones((n, n), dtype=np.float64) / float(n)
    b = -0.5 * j @ d2 @ j
    eigvals, eigvecs = np.linalg.eigh(b)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    eigvals = np.maximum(eigvals[:n_components], 0.0)
    coords = eigvecs[:, :n_components] * np.sqrt(eigvals[None, :])
    return coords.astype(np.float32)


def coords_from_match(mol3d: Chem.Mol, order_2d: list[int], match_2d_to_3d: tuple[int, ...]) -> np.ndarray:
    conf = mol3d.GetConformer()
    n = len(order_2d)
    xyz = np.zeros((n, 3), dtype=np.float32)
    for pos, atom_idx_2d in enumerate(order_2d):
        atom_idx_3d = match_2d_to_3d[atom_idx_2d]
        pt = conf.GetAtomPosition(int(atom_idx_3d))
        xyz[pos] = (pt.x, pt.y, pt.z)
    return xyz


def mol_from_smiles_coords(smiles: str, coords: np.ndarray) -> Chem.Mol:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    if coords.shape[0] != mol.GetNumAtoms():
        raise ValueError(f"Atom count mismatch for SMILES={smiles}: {coords.shape[0]} vs {mol.GetNumAtoms()}")

    order = canonical_atom_order(mol)
    conf = Chem.Conformer(mol.GetNumAtoms())
    for pos, atom_idx in enumerate(order):
        x0, y0, z0 = map(float, coords[pos])
        conf.SetAtomPosition(int(atom_idx), Point3D(x0, y0, z0))
    mol.RemoveAllConformers()
    mol.AddConformer(conf, assignId=True)
    return mol


def init_coords_from_smiles(smiles: str) -> np.ndarray:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")

    try:
        rdDepictor.Compute2DCoords(mol)
    except Exception:
        try:
            from rdkit.Chem import AllChem
        except Exception:  # pragma: no cover - optional dependency
            AllChem = None
        if AllChem is not None:
            AllChem.Compute2DCoords(mol)

    n = mol.GetNumAtoms()
    coords = np.zeros((n, 3), dtype=np.float32)
    if mol.GetNumConformers() > 0:
        conf = mol.GetConformer()
        order = canonical_atom_order(mol)
        for pos, atom_idx in enumerate(order):
            pt = conf.GetAtomPosition(int(atom_idx))
            coords[pos] = (float(pt.x), float(pt.y), 0.0)

    if not coords.any():
        if n == 1:
            coords[0, 0] = 1.0
        else:
            angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False, dtype=np.float32)
            coords[:, 0] = np.cos(angles)
            coords[:, 1] = np.sin(angles)
    return coords
