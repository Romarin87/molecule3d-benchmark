from __future__ import annotations

import math
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
        raise ImportError("PyTorch is required for EGNNTransformer.")


def _scatter_sum(src: "torch.Tensor", index: "torch.Tensor", dim_size: int) -> "torch.Tensor":
    out = torch.zeros((dim_size, src.size(-1)), device=src.device, dtype=src.dtype)
    out.index_add_(0, index, src)
    return out


def _rbf_features(dist: "torch.Tensor", num_bins: int, cutoff: float) -> "torch.Tensor":
    if num_bins <= 1:
        return torch.exp(-dist.unsqueeze(-1) ** 2)
    centers = torch.linspace(0.0, cutoff, num_bins, device=dist.device, dtype=dist.dtype)
    step = float(centers[1] - centers[0])
    gamma = 1.0 / (step * step + 1e-8)
    diff = dist.unsqueeze(-1) - centers
    return torch.exp(-gamma * diff * diff)


class _MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        return self.net(x)


class EGNNLayer(nn.Module):
    def __init__(self, node_dim: int, edge_dim: int, hidden_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.edge_mlp = _MLP(node_dim * 2 + edge_dim + 1, hidden_dim, hidden_dim, dropout=dropout)
        self.coord_mlp = _MLP(hidden_dim, hidden_dim, 1, dropout=dropout)
        self.node_mlp = _MLP(node_dim + hidden_dim, hidden_dim, node_dim, dropout=dropout)

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


class GraphTransformerLayer(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        rbf_bins: int,
        rbf_cutoff: float,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.rbf_bins = rbf_bins
        self.rbf_cutoff = rbf_cutoff

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.dist_mlp = _MLP(rbf_bins, dim, num_heads, dropout=dropout)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, h: "torch.Tensor", coords: "torch.Tensor") -> "torch.Tensor":
        n = h.size(0)
        h_norm = self.norm1(h)
        qkv = self.qkv(h_norm).view(n, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(1, 2, 0, 3)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        dist = torch.cdist(coords, coords)
        rbf = _rbf_features(dist, self.rbf_bins, self.rbf_cutoff)
        dist_bias = self.dist_mlp(rbf).permute(2, 0, 1)
        attn = attn + dist_bias
        attn = attn - attn.max(dim=-1, keepdim=True).values
        attn = torch.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = out.permute(1, 0, 2).reshape(n, self.dim)
        out = self.proj(out)
        h = h + self.dropout(out)
        h = h + self.dropout(self.ff(self.norm2(h)))
        return h


class EGNNTransformer(nn.Module):
    def __init__(
        self,
        node_feat_dim: int,
        edge_feat_dim: int = 0,
        hidden_dim: int = 128,
        num_layers: int = 4,
        num_heads: int = 4,
        dropout: float = 0.0,
        rbf_bins: int = 16,
        rbf_cutoff: float = 5.0,
    ) -> None:
        _ensure_torch()
        super().__init__()
        self.edge_feat_dim = edge_feat_dim
        self.node_proj = nn.Linear(node_feat_dim, hidden_dim)
        self.layers = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        "egnn": EGNNLayer(hidden_dim, edge_feat_dim, hidden_dim, dropout=dropout),
                        "trans": GraphTransformerLayer(
                            hidden_dim,
                            num_heads=num_heads,
                            rbf_bins=rbf_bins,
                            rbf_cutoff=rbf_cutoff,
                            dropout=dropout,
                        ),
                    }
                )
                for _ in range(num_layers)
            ]
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
        for block in self.layers:
            h, x = block["egnn"](h, x, edge_index, edge_attr)
            h = block["trans"](h, x)
        return x
