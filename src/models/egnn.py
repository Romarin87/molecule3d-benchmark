from __future__ import annotations

from typing import Optional

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
        raise ImportError("PyTorch is required for EGNN.")


def _scatter_sum(src: "torch.Tensor", index: "torch.Tensor", dim_size: int) -> "torch.Tensor":
    out = torch.zeros((dim_size, src.size(-1)), device=src.device, dtype=src.dtype)
    out.index_add_(0, index, src)
    return out


class _MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, out_dim))

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        return self.net(x)


class EGNNLayer(nn.Module):
    def __init__(self, node_dim: int, edge_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.edge_mlp = _MLP(node_dim * 2 + edge_dim + 1, hidden_dim, hidden_dim)
        self.coord_mlp = _MLP(hidden_dim, hidden_dim, 1)
        self.node_mlp = _MLP(node_dim + hidden_dim, hidden_dim, node_dim)

    def forward(
        self, h: "torch.Tensor", x: "torch.Tensor", edge_index: "torch.Tensor", edge_attr: "torch.Tensor"
    ) -> tuple["torch.Tensor", "torch.Tensor"]:
        row, col = edge_index
        diff = x[row] - x[col]
        dist2 = (diff * diff).sum(dim=-1, keepdim=True)
        m_ij = self.edge_mlp(torch.cat([h[row], h[col], edge_attr, dist2], dim=-1))
        coord_scale = self.coord_mlp(m_ij)
        trans = diff * coord_scale
        agg = _scatter_sum(m_ij, row, h.size(0))
        delta = _scatter_sum(trans, row, h.size(0))
        x = x + delta
        h = self.node_mlp(torch.cat([h, agg], dim=-1))
        return h, x


class EGNN(nn.Module):
    def __init__(
        self,
        node_feat_dim: int,
        edge_feat_dim: int = 0,
        hidden_dim: int = 128,
        num_layers: int = 4,
    ) -> None:
        _ensure_torch()
        super().__init__()
        self.edge_feat_dim = edge_feat_dim
        self.node_proj = nn.Linear(node_feat_dim, hidden_dim)
        self.layers = nn.ModuleList(
            [EGNNLayer(hidden_dim, edge_feat_dim, hidden_dim) for _ in range(num_layers)]
        )

    def forward(
        self,
        node_feats: "torch.Tensor",
        edge_index: "torch.Tensor",
        edge_attr: Optional["torch.Tensor"] = None,
        coords: Optional["torch.Tensor"] = None,
    ) -> "torch.Tensor":
        n = node_feats.size(0)
        if coords is None:
            coords = torch.zeros((n, 3), device=node_feats.device, dtype=node_feats.dtype)
        if edge_attr is None:
            edge_attr = torch.zeros((edge_index.size(1), self.edge_feat_dim), device=node_feats.device)

        h = self.node_proj(node_feats)
        x = coords
        for layer in self.layers:
            h, x = layer(h, x, edge_index, edge_attr)
        return x
