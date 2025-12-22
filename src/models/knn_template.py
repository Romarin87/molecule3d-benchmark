from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem

from .types import MoleculeRecord


@dataclass
class KNNTemplateModel:
    k: int = 1
    fp_radius: int = 2
    fp_bits: int = 2048
    use_chirality: bool = False
    records: list[MoleculeRecord] = field(default_factory=list, init=False)
    fps: list[DataStructs.cDataStructs.ExplicitBitVect] = field(default_factory=list, init=False)

    def fit(self, records: list[MoleculeRecord]) -> "KNNTemplateModel":
        if self.k < 1:
            raise ValueError("k must be >= 1")
        self.records = list(records)
        self.fps = []
        for rec in self.records:
            mol = Chem.MolFromSmiles(rec.smiles)
            if mol is None:
                raise ValueError(f"Invalid SMILES in training data: {rec.smiles}")
            fp = AllChem.GetMorganFingerprintAsBitVect(
                mol, radius=self.fp_radius, nBits=self.fp_bits, useChirality=self.use_chirality
            )
            self.fps.append(fp)
        return self

    def predict(self, smiles: str) -> Chem.Mol:
        if not self.records:
            raise RuntimeError("Model has not been fit.")
        query = Chem.MolFromSmiles(smiles)
        if query is None:
            raise ValueError(f"Invalid SMILES: {smiles}")

        qfp = AllChem.GetMorganFingerprintAsBitVect(
            query, radius=self.fp_radius, nBits=self.fp_bits, useChirality=self.use_chirality
        )
        sims = np.array([DataStructs.TanimotoSimilarity(qfp, fp) for fp in self.fps], dtype=np.float32)
        if sims.size == 0:
            raise RuntimeError("No templates available.")

        order = np.argsort(-sims)
        last_error = None
        for idx in order[: self.k]:
            template = Chem.RemoveHs(self.records[int(idx)].mol3d)
            if template.GetNumAtoms() != query.GetNumAtoms():
                last_error = "Atom count mismatch."
                continue
            match = template.GetSubstructMatch(query, useChirality=self.use_chirality)
            if not match or len(match) != query.GetNumAtoms():
                last_error = "Substructure match failed."
                continue

            conf = Chem.Conformer(query.GetNumAtoms())
            t_conf = template.GetConformer()
            for q_idx, t_idx in enumerate(match):
                pt = t_conf.GetAtomPosition(int(t_idx))
                conf.SetAtomPosition(int(q_idx), pt)
            query.RemoveAllConformers()
            query.AddConformer(conf, assignId=True)
            return query

        raise RuntimeError(f"Failed to map query onto templates. Last error: {last_error}")
