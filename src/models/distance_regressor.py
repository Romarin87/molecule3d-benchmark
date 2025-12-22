from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

import joblib
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Geometry import Point3D
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .chem_utils import (
    classical_mds,
    coords_from_match,
    featurize_mol,
    flatten_upper_triangle,
    vector_to_distance_matrix,
)
from .types import MoleculeRecord

try:
    import xgboost as xgb
except Exception:  # pragma: no cover - optional dependency
    xgb = None


@dataclass(frozen=True)
class FeatureConfig:
    atom_count: int
    elements: tuple[int, ...] = (6, 7, 8, 9, 15, 16, 17, 35, 53)
    max_degree: int = 6

    @property
    def atom_feature_dim(self) -> int:
        return (len(self.elements) + 1) + (self.max_degree + 1) + 3

    @property
    def feature_dim(self) -> int:
        n = self.atom_count
        return (n * self.atom_feature_dim) + (n * n)

    @property
    def distance_dim(self) -> int:
        n = self.atom_count
        return (n * (n - 1)) // 2


def build_regressor(model_name: str, seed: int) -> Pipeline:
    if model_name == "ridge":
        return Pipeline([("scaler", StandardScaler()), ("model", Ridge(alpha=1.0, random_state=seed))])
    if model_name == "mlp":
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "model",
                    MLPRegressor(
                        hidden_layer_sizes=(512, 256),
                        activation="relu",
                        solver="adam",
                        learning_rate_init=1e-3,
                        max_iter=200,
                        early_stopping=True,
                        n_iter_no_change=10,
                        random_state=seed,
                    ),
                ),
            ]
        )
    if model_name == "rf":
        return Pipeline(
            [
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=200, n_jobs=-1, random_state=seed, min_samples_leaf=1
                    ),
                )
            ]
        )
    if model_name == "xgb":
        if xgb is None:
            raise ImportError("xgboost is not installed.")
        return Pipeline(
            [
                (
                    "model",
                    xgb.XGBRegressor(
                        n_estimators=400,
                        max_depth=8,
                        learning_rate=0.05,
                        subsample=0.8,
                        colsample_bytree=0.9,
                        objective="reg:squarederror",
                        n_jobs=-1,
                        random_state=seed,
                    ),
                )
            ]
        )
    raise ValueError(f"Unknown model: {model_name}")


def _distance_vector_from_xyz(xyz: np.ndarray) -> np.ndarray:
    diff = xyz[:, None, :] - xyz[None, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=-1, dtype=np.float64)).astype(np.float32)
    return flatten_upper_triangle(dist)


def _record_to_xy(record: MoleculeRecord, cfg: FeatureConfig) -> tuple[np.ndarray, np.ndarray] | None:
    mol2d = Chem.MolFromSmiles(record.smiles)
    if mol2d is None or record.mol3d is None:
        return None
    if mol2d.GetNumAtoms() != cfg.atom_count or record.mol3d.GetNumAtoms() != cfg.atom_count:
        return None

    match = record.mol3d.GetSubstructMatch(mol2d, useChirality=False)
    if not match or len(match) != cfg.atom_count:
        return None

    x, order = featurize_mol(mol2d, cfg.atom_count, cfg.elements, cfg.max_degree)
    xyz = coords_from_match(record.mol3d, order, match)
    y = _distance_vector_from_xyz(xyz)
    if y.shape[0] != cfg.distance_dim:
        return None
    return x, y


@dataclass
class DistanceRegressorModel:
    model_name: str
    cfg: FeatureConfig
    seed: int = 0
    model: Pipeline | None = None

    def fit(self, records: Iterable[MoleculeRecord]) -> "DistanceRegressorModel":
        x_list: list[np.ndarray] = []
        y_list: list[np.ndarray] = []
        for rec in records:
            item = _record_to_xy(rec, self.cfg)
            if item is None:
                continue
            x, y = item
            x_list.append(x)
            y_list.append(y)

        if not x_list:
            raise RuntimeError("No valid training samples for distance regressor.")

        x_arr = np.stack(x_list, axis=0)
        y_arr = np.stack(y_list, axis=0)
        self.model = build_regressor(self.model_name, seed=self.seed)
        self.model.fit(x_arr, y_arr)
        return self

    def predict(self, smiles: str, mmff: bool = False) -> Chem.Mol:
        if self.model is None:
            raise RuntimeError("Model has not been fit.")
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES: {smiles}")
        if mol.GetNumAtoms() != self.cfg.atom_count:
            raise ValueError(
                f"Model expects {self.cfg.atom_count} atoms, got {mol.GetNumAtoms()} atoms for SMILES={smiles}"
            )

        x, order = featurize_mol(mol, self.cfg.atom_count, self.cfg.elements, self.cfg.max_degree)
        y_pred = self.model.predict(x.reshape(1, -1))[0].astype(np.float32, copy=False)
        y_pred = np.clip(y_pred, 0.0, None)

        dist = vector_to_distance_matrix(y_pred, self.cfg.atom_count)
        xyz = classical_mds(dist, n_components=3)

        conf = Chem.Conformer(self.cfg.atom_count)
        for pos, atom_idx in enumerate(order):
            x0, y0, z0 = map(float, xyz[pos])
            conf.SetAtomPosition(int(atom_idx), Point3D(x0, y0, z0))
        mol.RemoveAllConformers()
        mol.AddConformer(conf, assignId=True)

        if mmff:
            mol_h = Chem.AddHs(mol, addCoords=True)
            AllChem.MMFFOptimizeMolecule(mol_h, maxIters=200)
            mol = Chem.RemoveHs(mol_h)

        return mol

    def save(self, path: str) -> None:
        if self.model is None:
            raise RuntimeError("Model has not been fit.")
        payload = {
            "model": self.model,
            "feature_config": asdict(self.cfg),
            "model_name": self.model_name,
            "seed": self.seed,
        }
        joblib.dump(payload, path)

    @classmethod
    def load(cls, path: str) -> "DistanceRegressorModel":
        payload = joblib.load(path)
        cfg = FeatureConfig(**payload["feature_config"])
        instance = cls(model_name=payload.get("model_name", "rf"), cfg=cfg, seed=int(payload.get("seed", 0)))
        instance.model = payload["model"]
        return instance
