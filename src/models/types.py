from __future__ import annotations

from dataclasses import dataclass

from rdkit import Chem


@dataclass(frozen=True)
class MoleculeRecord:
    smiles: str
    mol3d: Chem.Mol
