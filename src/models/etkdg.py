from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem


@dataclass
class ETKDGModel:
    num_confs: int = 10
    random_seed: int = 0
    use_mmff: bool = False
    max_iters: int = 200

    def fit(self, records=None) -> "ETKDGModel":
        return self

    def predict(self, smiles: str) -> Chem.Mol:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES: {smiles}")

        mol = Chem.AddHs(mol)
        params = AllChem.ETKDGv3()
        params.randomSeed = int(self.random_seed)
        conf_ids = list(AllChem.EmbedMultipleConfs(mol, numConfs=self.num_confs, params=params))
        if not conf_ids:
            raise RuntimeError("ETKDG failed to embed conformers.")

        energies = None
        if self.use_mmff:
            results = AllChem.MMFFOptimizeMoleculeConfs(mol, maxIters=self.max_iters)
            energies = [float(item[1]) for item in results]

        best_id = conf_ids[0]
        if energies is not None:
            best_id = conf_ids[int(np.argmin(energies))]

        best_conf = Chem.Conformer(mol.GetConformer(best_id))
        mol.RemoveAllConformers()
        mol.AddConformer(best_conf, assignId=True)
        mol = Chem.RemoveHs(mol)
        return mol
