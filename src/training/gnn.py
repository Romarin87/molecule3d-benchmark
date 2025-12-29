from __future__ import annotations

import sys
from typing import Iterable

import numpy as np

from ..datasets.graph import GraphSample
from ..models.egnn import EGNN
from ..models.egnn_transformer import EGNNTransformer
from ..models.mpnn import MPNN
from ..models.chem_utils import init_coords_from_smiles

try:
    import torch
    from torch import nn
    from torch.nn import functional as F
except Exception:  # pragma: no cover - optional dependency
    torch = None
    nn = None
    F = None

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover - optional dependency
    tqdm = None


class _SimpleProgress:
    def __init__(self, total: int, desc: str) -> None:
        self.total = total
        self.desc = desc
        self.count = 0
        self.width = 30
        self._last_len = 0

    def update(self, n: int = 1) -> None:
        if self.total <= 0:
            return
        self.count += n
        filled = int(self.width * self.count / self.total)
        bar = "#" * filled + "-" * (self.width - filled)
        pct = (self.count / self.total) * 100
        msg = f"{self.desc} [{bar}] {self.count}/{self.total} ({pct:5.1f}%)"
        if self._last_len > len(msg):
            msg = msg + " " * (self._last_len - len(msg))
        sys.stderr.write("\r" + msg)
        sys.stderr.flush()
        self._last_len = len(msg)

    def close(self) -> None:
        if self.total <= 0:
            return
        sys.stderr.write("\n")
        sys.stderr.flush()

    def __enter__(self) -> "_SimpleProgress":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


def _progress(total: int, desc: str):
    if tqdm is not None:
        return tqdm(total=total, desc=desc, ascii=True)
    return _SimpleProgress(total=total, desc=desc)


def _ensure_torch() -> None:
    if torch is None or nn is None or F is None:
        raise ImportError("PyTorch is required for training.")


def _dist_vector_from_coords(x: "torch.Tensor") -> "torch.Tensor":
    n = x.size(0)
    diff = x[:, None, :] - x[None, :, :]
    dist = torch.sqrt((diff * diff).sum(dim=-1) + 1e-8)
    idx = torch.triu_indices(n, n, offset=1, device=x.device)
    return dist[idx[0], idx[1]]


def _to_torch(sample: GraphSample, device: str) -> tuple["torch.Tensor", ...]:
    node_feats, edge_index, edge_attr, dist_vec, xyz = sample.to_torch(device=device)
    return node_feats, edge_index, edge_attr, dist_vec, xyz


def train_mpnn(
    samples: Iterable[GraphSample],
    node_feat_dim: int,
    edge_feat_dim: int = 1,
    hidden_dim: int = 128,
    num_layers: int = 4,
    epochs: int = 10,
    lr: float = 1e-3,
    device: str = "cpu",
    return_loss_history: bool = False,
) -> MPNN | tuple[MPNN, list[float]]:
    _ensure_torch()
    model = MPNN(
        node_feat_dim=node_feat_dim,
        edge_feat_dim=edge_feat_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    dataset = list(samples)
    total_steps = len(dataset) * epochs
    loss_history: list[float] = []
    with _progress(total=total_steps, desc="train_mpnn") as pbar:
        for _ in range(epochs):
            model.train()
            epoch_loss = 0.0
            steps = 0
            for sample in dataset:
                node_feats, edge_index, edge_attr, dist_vec, _ = _to_torch(sample, device=device)
                pred = model(node_feats, edge_index, edge_attr)
                loss = F.mse_loss(pred, dist_vec)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += float(loss.item())
                steps += 1
                pbar.update(1)
            if steps > 0:
                loss_history.append(epoch_loss / steps)
            else:
                loss_history.append(float("nan"))

    if return_loss_history:
        return model, loss_history
    return model


def train_egnn(
    samples: Iterable[GraphSample],
    node_feat_dim: int,
    edge_feat_dim: int = 1,
    hidden_dim: int = 128,
    num_layers: int = 4,
    epochs: int = 10,
    lr: float = 1e-3,
    device: str = "cpu",
    return_loss_history: bool = False,
) -> EGNN | tuple[EGNN, list[float]]:
    _ensure_torch()
    model = EGNN(
        node_feat_dim=node_feat_dim,
        edge_feat_dim=edge_feat_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    dataset = list(samples)
    init_coords = [torch.from_numpy(init_coords_from_smiles(sample.smiles)) for sample in dataset]
    total_steps = len(dataset) * epochs
    loss_history: list[float] = []
    with _progress(total=total_steps, desc="train_egnn") as pbar:
        for _ in range(epochs):
            model.train()
            epoch_loss = 0.0
            steps = 0
            for sample, init_xyz in zip(dataset, init_coords):
                node_feats, edge_index, edge_attr, dist_vec, _ = _to_torch(sample, device=device)
                coords0 = init_xyz.to(device=device, dtype=node_feats.dtype)
                coords = model(node_feats, edge_index, edge_attr=edge_attr, coords=coords0)
                pred_dist = _dist_vector_from_coords(coords)
                loss = F.mse_loss(pred_dist, dist_vec)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += float(loss.item())
                steps += 1
                pbar.update(1)
            if steps > 0:
                loss_history.append(epoch_loss / steps)
            else:
                loss_history.append(float("nan"))

    if return_loss_history:
        return model, loss_history
    return model


def train_egnn_transformer(
    samples: Iterable[GraphSample],
    node_feat_dim: int,
    edge_feat_dim: int = 1,
    hidden_dim: int = 128,
    num_layers: int = 4,
    num_heads: int = 4,
    dropout: float = 0.0,
    rbf_bins: int = 16,
    rbf_cutoff: float = 5.0,
    epochs: int = 10,
    lr: float = 1e-3,
    device: str = "cpu",
    return_loss_history: bool = False,
) -> EGNNTransformer | tuple[EGNNTransformer, list[float]]:
    _ensure_torch()
    model = EGNNTransformer(
        node_feat_dim=node_feat_dim,
        edge_feat_dim=edge_feat_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        dropout=dropout,
        rbf_bins=rbf_bins,
        rbf_cutoff=rbf_cutoff,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    dataset = list(samples)
    init_coords = [torch.from_numpy(init_coords_from_smiles(sample.smiles)) for sample in dataset]
    total_steps = len(dataset) * epochs
    loss_history: list[float] = []
    with _progress(total=total_steps, desc="train_egnn_transformer") as pbar:
        for _ in range(epochs):
            model.train()
            epoch_loss = 0.0
            steps = 0
            for sample, init_xyz in zip(dataset, init_coords):
                node_feats, edge_index, edge_attr, dist_vec, _ = _to_torch(sample, device=device)
                coords0 = init_xyz.to(device=device, dtype=node_feats.dtype)
                coords = model(node_feats, edge_index, edge_attr=edge_attr, coords=coords0)
                pred_dist = _dist_vector_from_coords(coords)
                loss = F.mse_loss(pred_dist, dist_vec)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += float(loss.item())
                steps += 1
                pbar.update(1)
            if steps > 0:
                loss_history.append(epoch_loss / steps)
            else:
                loss_history.append(float("nan"))

    if return_loss_history:
        return model, loss_history
    return model
