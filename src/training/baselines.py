from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from rdkit import Chem

from ..metrics.rmsd import best_rmsd, rmsd_stats
from ..models import DistanceRegressorModel, ETKDGModel, FeatureConfig, KNNTemplateModel
from ..models.types import MoleculeRecord


@dataclass
class EvalSummary:
    metrics: dict[str, float]
    evaluated: int
    skipped: int


def train_knn(
    records: Iterable[MoleculeRecord],
    k: int = 1,
    fp_radius: int = 2,
    fp_bits: int = 2048,
    use_chirality: bool = False,
) -> KNNTemplateModel:
    model = KNNTemplateModel(k=k, fp_radius=fp_radius, fp_bits=fp_bits, use_chirality=use_chirality)
    model.fit(list(records))
    return model


def train_distance_regressor(
    records: Iterable[MoleculeRecord],
    cfg: FeatureConfig,
    model_name: str = "rf",
    seed: int = 0,
) -> DistanceRegressorModel:
    model = DistanceRegressorModel(model_name=model_name, cfg=cfg, seed=seed)
    model.fit(records)
    return model


def build_etkdg(num_confs: int = 10, random_seed: int = 0, use_mmff: bool = False) -> ETKDGModel:
    return ETKDGModel(num_confs=num_confs, random_seed=random_seed, use_mmff=use_mmff)


def evaluate_model(
    model,
    records: Iterable[MoleculeRecord],
    max_samples: int = 0,
) -> EvalSummary:
    rmsds: list[float] = []
    skipped = 0
    evaluated = 0

    for rec in records:
        if max_samples > 0 and evaluated >= max_samples:
            break
        try:
            pred = model.predict(rec.smiles)
            gt = Chem.RemoveHs(rec.mol3d)
            if pred.GetNumAtoms() != gt.GetNumAtoms():
                skipped += 1
                continue
            rmsds.append(best_rmsd(pred, gt))
            evaluated += 1
        except Exception:
            skipped += 1

    metrics = rmsd_stats(rmsds)
    return EvalSummary(metrics=metrics, evaluated=evaluated, skipped=skipped)
