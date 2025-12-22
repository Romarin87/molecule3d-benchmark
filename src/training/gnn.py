from __future__ import annotations

from typing import Iterable

import numpy as np

from ..datasets.graph import GraphSample
from ..models.egnn import EGNN
from ..models.mpnn import MPNN

try:
    import torch
    from torch import nn
    from torch.nn import functional as F
except Exception:  # pragma: no cover - optional dependency
    torch = None
    nn = None
    F = None


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
) -> MPNN:
    _ensure_torch()
    model = MPNN(
        node_feat_dim=node_feat_dim,
        edge_feat_dim=edge_feat_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    dataset = list(samples)
    for _ in range(epochs):
        model.train()
        for sample in dataset:
            node_feats, edge_index, edge_attr, dist_vec, _ = _to_torch(sample, device=device)
            pred = model(node_feats, edge_index, edge_attr)
            loss = F.mse_loss(pred, dist_vec)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

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
) -> EGNN:
    _ensure_torch()
    model = EGNN(
        node_feat_dim=node_feat_dim,
        edge_feat_dim=edge_feat_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    dataset = list(samples)
    for _ in range(epochs):
        model.train()
        for sample in dataset:
            node_feats, edge_index, edge_attr, dist_vec, _ = _to_torch(sample, device=device)
            coords = model(node_feats, edge_index, edge_attr=edge_attr)
            pred_dist = _dist_vector_from_coords(coords)
            loss = F.mse_loss(pred_dist, dist_vec)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return model
