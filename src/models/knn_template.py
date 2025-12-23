from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
try:
    from rdkit.Chem import rdFingerprintGenerator
except Exception:  # pragma: no cover - optional dependency
    rdFingerprintGenerator = None

from .types import MoleculeRecord


@dataclass
class KNNTemplateModel:
    k: int = 1
    fp_radius: int = 2
    fp_bits: int = 2048
    use_chirality: bool = False
    records: list[MoleculeRecord] = field(default_factory=list, init=False)
    fps: list[DataStructs.cDataStructs.ExplicitBitVect] = field(default_factory=list, init=False)

    def _make_morgan_generator(self):
        if rdFingerprintGenerator is None:
            return None
        try:
            return rdFingerprintGenerator.GetMorganGenerator(
                radius=int(self.fp_radius),
                fpSize=int(self.fp_bits),
                includeChirality=bool(self.use_chirality),
            )
        except TypeError:
            return rdFingerprintGenerator.GetMorganGenerator(
                radius=int(self.fp_radius),
                fpSize=int(self.fp_bits),
                useChirality=bool(self.use_chirality),
            )

    def fit(self, records: list[MoleculeRecord]) -> "KNNTemplateModel":
        if self.k < 1:
            raise ValueError("k must be >= 1")
        self.records = list(records)
        self.fps = []
        generator = self._make_morgan_generator()
        for rec in self.records:
            mol = Chem.MolFromSmiles(rec.smiles)
            if mol is None:
                raise ValueError(f"Invalid SMILES in training data: {rec.smiles}")
            if generator is not None:
                fp = generator.GetFingerprint(mol)
            else:
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

        generator = self._make_morgan_generator()
        if generator is not None:
            qfp = generator.GetFingerprint(query)
        else:
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
