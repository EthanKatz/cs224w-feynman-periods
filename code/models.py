"""GNN architectures for graph-level regression of log Feynman periods.

All three models share the same skeleton so that only the message-passing
operator differs:

    node features --Linear--> h^0 --[conv -> BatchNorm -> ReLU (+res)] x K-->
    [sum-pool || mean-pool] --MLP--> scalar  (standardised log period)

Sum pooling is important here: with mean pooling alone a GNN fed constant
features on a 4-regular graph could not even recover the graph size (and
hence the loop number L, which explains most of the variance in log P).
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, GINConv, SAGEConv, global_add_pool, global_mean_pool


def make_norm(norm: str, dim: int) -> nn.Module:
    """BatchNorm by default.  'none' is needed for constant input features:
    every node embedding is then identical, BatchNorm's running variance
    collapses to ~0 and eval-mode outputs explode."""
    return nn.BatchNorm1d(dim) if norm == "batch" else nn.Identity()


def make_conv(kind: str, dim: int, norm: str = "batch") -> nn.Module:
    """Return one message-passing layer of the requested type."""
    if kind == "gcn":    # Kipf & Welling 2017: symmetric-normalised neighbourhood mean
        return GCNConv(dim, dim)
    if kind == "sage":   # Hamilton et al. 2017: [h_v || mean_{u in N(v)} h_u] W
        return SAGEConv(dim, dim, aggr="mean")
    if kind == "gin":    # Xu et al. 2019: MLP((1+eps) h_v + sum_{u in N(v)} h_u)
        mlp = nn.Sequential(nn.Linear(dim, dim), make_norm(norm, dim), nn.ReLU(), nn.Linear(dim, dim))
        return GINConv(mlp, train_eps=True)
    raise ValueError(kind)


class PeriodGNN(nn.Module):
    """Graph-level regressor: predicts standardised log(period) of a graph.

    Args:
        in_dim: node-feature dimension.
        kind: 'gcn' | 'sage' | 'gin'.
        hidden: hidden width.
        layers: number of message-passing layers.
        dropout: dropout in the readout MLP.
        norm: 'batch' (default) or 'none'.
    """

    def __init__(self, in_dim: int, kind: str = "gin", hidden: int = 64,
                 layers: int = 4, dropout: float = 0.0, norm: str = "batch"):
        super().__init__()
        self.encoder = nn.Linear(in_dim, hidden)
        self.convs = nn.ModuleList(make_conv(kind, hidden, norm) for _ in range(layers))
        self.norms = nn.ModuleList(make_norm(norm, hidden) for _ in range(layers))
        self.head = nn.Sequential(
            nn.Linear(2 * hidden, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, 1))

    def forward(self, x, edge_index, batch):
        h = self.encoder(x)
        for conv, norm in zip(self.convs, self.norms):
            h = h + torch.relu(norm(conv(h, edge_index)))   # residual connection
        g = torch.cat([global_add_pool(h, batch), global_mean_pool(h, batch)], dim=-1)
        return self.head(g).squeeze(-1)
