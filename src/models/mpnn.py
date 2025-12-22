from __future__ import annotations

from typing import Optional

import numpy as np

from .chem_utils import classical_mds, vector_to_distance_matrix

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
        raise ImportError("PyTorch is required for MPNN.")


def _scatter_sum(src: "torch.Tensor", index: "torch.Tensor", dim_size: int) -> "torch.Tensor":
    out = torch.zeros((dim_size, src.size(-1)), device=src.device, dtype=src.dtype)
    out.index_add_(0, index, src)
    return out


class _MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, out_dim))

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        return self.net(x)


class MPNNLayer(nn.Module):
    def __init__(self, node_dim: int, edge_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.msg = _MLP(node_dim * 2 + edge_dim, hidden_dim, hidden_dim)
        self.upd = _MLP(node_dim + hidden_dim, hidden_dim, node_dim)

    def forward(
        self, h: "torch.Tensor", edge_index: "torch.Tensor", edge_attr: "torch.Tensor"
    ) -> "torch.Tensor":
        row, col = edge_index
        m = self.msg(torch.cat([h[row], h[col], edge_attr], dim=-1))
        agg = _scatter_sum(m, row, h.size(0))
        h = self.upd(torch.cat([h, agg], dim=-1))
        return h


class MPNN(nn.Module):
    def __init__(
        self,
        node_feat_dim: int,
        edge_feat_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 4,
        pair_hidden_dim: int = 128,
    ) -> None:
        _ensure_torch()
        super().__init__()
        self.node_proj = nn.Linear(node_feat_dim, hidden_dim)
        self.layers = nn.ModuleList(
            [MPNNLayer(hidden_dim, edge_feat_dim, hidden_dim) for _ in range(num_layers)]
        )
        self.pair_mlp = _MLP(hidden_dim * 2, pair_hidden_dim, 1)

    def forward(
        self, node_feats: "torch.Tensor", edge_index: "torch.Tensor", edge_attr: "torch.Tensor"
    ) -> "torch.Tensor":
        h = self.node_proj(node_feats)
        for layer in self.layers:
            h = layer(h, edge_index, edge_attr)

        n = h.size(0)
        pair_idx = torch.triu_indices(n, n, offset=1, device=h.device)
        h_i = h[pair_idx[0]]
        h_j = h[pair_idx[1]]
        dist = F.softplus(self.pair_mlp(torch.cat([h_i, h_j], dim=-1))).squeeze(-1)
        return dist

    def predict_coords(
        self, node_feats: "torch.Tensor", edge_index: "torch.Tensor", edge_attr: "torch.Tensor"
    ) -> np.ndarray:
        dist_vec = self.forward(node_feats, edge_index, edge_attr)
        n = node_feats.size(0)
        dist = vector_to_distance_matrix(dist_vec.detach().cpu().numpy(), n)
        return classical_mds(dist, n_components=3)
