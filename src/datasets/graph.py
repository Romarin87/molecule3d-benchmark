from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Iterator

import numpy as np
from rdkit import Chem

from ..models.chem_utils import atom_features, canonical_atom_order, coords_from_match, flatten_upper_triangle
from ..models.types import MoleculeRecord


@dataclass(frozen=True)
class GraphSample:
    smiles: str
    node_feats: np.ndarray
    edge_index: np.ndarray
    edge_attr: np.ndarray
    dist_vec: np.ndarray
    xyz: np.ndarray

    def to_torch(self, device: str | None = None):
        try:
            import torch
        except Exception as exc:  # pragma: no cover - optional dependency
            raise ImportError("PyTorch is required for to_torch().") from exc

        dev = torch.device(device) if device is not None else None
        node_feats = torch.from_numpy(self.node_feats)
        edge_index = torch.from_numpy(self.edge_index).long()
        edge_attr = torch.from_numpy(self.edge_attr)
        dist_vec = torch.from_numpy(self.dist_vec)
        xyz = torch.from_numpy(self.xyz)
        if dev is not None:
            node_feats = node_feats.to(dev)
            edge_index = edge_index.to(dev)
            edge_attr = edge_attr.to(dev)
            dist_vec = dist_vec.to(dev)
            xyz = xyz.to(dev)
        return node_feats, edge_index, edge_attr, dist_vec, xyz


def _mol_to_graph(
    mol2d: Chem.Mol, order: list[int], elements: Iterable[int], max_degree: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = mol2d.GetNumAtoms()
    idx_to_pos = {atom_idx: pos for pos, atom_idx in enumerate(order)}
    node_feats = np.stack(
        [atom_features(mol2d.GetAtomWithIdx(atom_idx), elements, max_degree) for atom_idx in order], axis=0
    )

    edges = []
    edge_attrs = []
    for bond in mol2d.GetBonds():
        i = idx_to_pos[bond.GetBeginAtomIdx()]
        j = idx_to_pos[bond.GetEndAtomIdx()]
        bt = float(bond.GetBondTypeAsDouble())
        edges.append((i, j))
        edges.append((j, i))
        edge_attrs.append([bt])
        edge_attrs.append([bt])

    if edges:
        edge_index = np.array(edges, dtype=np.int64).T
        edge_attr = np.array(edge_attrs, dtype=np.float32)
    else:
        edge_index = np.zeros((2, 0), dtype=np.int64)
        edge_attr = np.zeros((0, 1), dtype=np.float32)

    return node_feats.astype(np.float32), edge_index, edge_attr


def record_to_graph(
    record: MoleculeRecord,
    elements: Iterable[int],
    max_degree: int,
    atom_count: int | None = None,
) -> GraphSample | None:
    mol2d = Chem.MolFromSmiles(record.smiles)
    if mol2d is None or record.mol3d is None:
        return None
    if atom_count is not None:
        if mol2d.GetNumAtoms() != atom_count or record.mol3d.GetNumAtoms() != atom_count:
            return None

    match = record.mol3d.GetSubstructMatch(mol2d, useChirality=False)
    if not match or len(match) != mol2d.GetNumAtoms():
        return None

    order = canonical_atom_order(mol2d)
    node_feats, edge_index, edge_attr = _mol_to_graph(mol2d, order, elements, max_degree)
    xyz = coords_from_match(record.mol3d, order, match)
    diff = xyz[:, None, :] - xyz[None, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=-1, dtype=np.float64)).astype(np.float32)
    dist_vec = flatten_upper_triangle(dist)
    return GraphSample(
        smiles=record.smiles,
        node_feats=node_feats,
        edge_index=edge_index,
        edge_attr=edge_attr,
        dist_vec=dist_vec,
        xyz=xyz,
    )


def iter_graph_samples(
    records: Iterable[MoleculeRecord],
    elements: Iterable[int],
    max_degree: int,
    atom_count: int | None = None,
    max_samples: int = 0,
) -> Iterator[GraphSample]:
    count = 0
    for rec in records:
        if max_samples > 0 and count >= max_samples:
            break
        sample = record_to_graph(rec, elements=elements, max_degree=max_degree, atom_count=atom_count)
        if sample is None:
            continue
        count += 1
        yield sample
