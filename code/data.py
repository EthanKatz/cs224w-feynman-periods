"""Data pipeline for predicting Feynman periods in phi^4 theory.

Source: Balduf & Shaban, "Feynman periods in phi4-theory", Borealis Dataverse,
doi:10.5683/SP3/NLEDGH (CC-BY 4.0).  Each file ``periods_LL.csv(.bz2)`` holds
one *completed* primitive graph per line: a 4-regular simple graph with L+2
vertices and 2(L+2) edges, where L is the loop number of the (decompleted)
Feynman graph.  All decompletions of one completion share the same period, so
the completed graph is the natural input object.

This module
  1. parses the semicolon-separated files into edge lists + periods,
  2. computes node features (constant, arbitrary index, per-node cycle
     participation, Laplacian eigenvector PE, random-walk SE),
  3. builds a list of ``torch_geometric.data.Data`` objects with
     ``y = log(period)`` and caches it to ``data/processed``,
  4. creates train/val/test splits that are *grouped by period value* so that
     graphs related by known period symmetries (which have bit-identical
     periods in the dataset) never straddle the split.

Run ``python code/data.py`` once to build the cache.
"""
from __future__ import annotations

import bz2
import os
import re
import zlib
from dataclasses import dataclass

import numpy as np
import torch
from numba import njit
from torch_geometric.data import Data

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(ROOT, "data", "raw")
PROC_DIR = os.path.join(ROOT, "data", "processed")

# Hyper-parameters of the structural encodings.
MAX_CYCLE_LEN = 8        # per-node counts of simple cycles of length 3..8
N_LAPPE = 8              # number of non-trivial Laplacian eigenvectors
N_RWSE = 12              # random-walk return probabilities for k = 1..12
MAX_NODES = 14           # L <= 12  ->  at most 14 vertices in the completion

EDGE_RE = re.compile(r"\{(\d+),(\d+)\}")


# --------------------------------------------------------------------------
# 1. Parsing
# --------------------------------------------------------------------------
@dataclass
class RawGraph:
    """One row of the dataset (only the columns we use)."""
    nickel: str
    edges: np.ndarray        # (E, 2) int array, 0-indexed vertex labels
    num_nodes: int
    loops: int
    period: float
    period_unc: float
    hepp: float              # Hepp bound (an *expensive* reference feature)
    period_str: str          # exact string, used to group symmetric graphs


def raw_path(loops: int) -> str:
    base = os.path.join(RAW_DIR, f"periods_{loops:02d}.csv")
    return base if os.path.exists(base) else base + ".bz2"


def iter_rows(loops: int):
    """Yield ``RawGraph`` objects for every graph with the given loop number."""
    path = raw_path(loops)
    opener = bz2.open if path.endswith(".bz2") else open
    with opener(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            if line.startswith("#") or not line.strip():
                continue
            cols = [c.strip() for c in line.split(";")]
            edges = np.array([(int(a), int(b)) for a, b in EDGE_RE.findall(cols[4])],
                             dtype=np.int64)
            n = int(edges.max()) + 1
            L = int(cols[5])
            assert L == loops and n == L + 2 and len(edges) == 2 * n, cols[0]
            yield RawGraph(nickel=cols[0], edges=edges, num_nodes=n, loops=L,
                           period=float(cols[2]), period_unc=float(cols[3]),
                           hepp=float(cols[1]), period_str=cols[2])


# --------------------------------------------------------------------------
# 2. Node features
# --------------------------------------------------------------------------
@njit(cache=True)
def _cycle_participation(adj: np.ndarray, max_len: int) -> np.ndarray:
    """Count, for every vertex, the simple cycles of each length through it.

    Enumerates each simple cycle exactly once: the DFS starts at the cycle's
    smallest vertex ``s``, only visits vertices > s, and a closed cycle is
    accepted only if its second vertex is smaller than its last vertex
    (removes the two orientations).

    Returns an (n, max_len - 2) array; column k holds cycles of length k + 3.
    """
    n = adj.shape[0]
    out = np.zeros((n, max_len - 2), dtype=np.int64)
    path = np.zeros(max_len, dtype=np.int64)
    on_path = np.zeros(n, dtype=np.bool_)
    nbr_idx = np.zeros(max_len, dtype=np.int64)  # iterator position per depth
    for s in range(n):
        path[0] = s
        on_path[s] = True
        depth = 0
        nbr_idx[0] = 0
        while depth >= 0:
            v = path[depth]
            advanced = False
            while nbr_idx[depth] < n:
                u = nbr_idx[depth]
                nbr_idx[depth] += 1
                if not adj[v, u]:
                    continue
                if u == s and depth >= 2 and path[1] < path[depth]:
                    length = depth + 1
                    for i in range(length):
                        out[path[i], length - 3] += 1
                    continue
                if u > s and not on_path[u] and depth + 1 < max_len:
                    depth += 1
                    path[depth] = u
                    on_path[u] = True
                    nbr_idx[depth] = 0
                    advanced = True
                    break
            if not advanced:
                on_path[path[depth]] = False
                depth -= 1
        on_path[s] = False
    return out


def cycle_features(adj: np.ndarray) -> np.ndarray:
    """log1p of per-node simple-cycle counts, lengths 3..MAX_CYCLE_LEN."""
    return np.log1p(_cycle_participation(adj.astype(np.bool_), MAX_CYCLE_LEN)).astype(np.float32)


def laplacian_pe(adj: np.ndarray, k: int = N_LAPPE) -> np.ndarray:
    """First ``k`` non-trivial eigenvectors of the normalized Laplacian.

    The graphs are 4-regular, so L_sym = I - A/4 and the eigenvectors are
    those of A.  Signs are arbitrary; we randomly flip them at train time.
    Zero-padded if the graph has fewer than k+1 vertices.
    """
    n = adj.shape[0]
    lap = np.eye(n) - adj / adj.sum(1, keepdims=True)
    _, vecs = np.linalg.eigh(lap)
    pe = np.zeros((n, k), dtype=np.float32)
    m = min(k, n - 1)
    pe[:, :m] = vecs[:, 1:1 + m]
    return pe


def rwse(adj: np.ndarray, k: int = N_RWSE) -> np.ndarray:
    """Random-walk structural encoding: return probabilities (P^t)_vv, t=1..k.

    Equivalent (up to the constant 4^t) to closed-walk counts, which mix
    cycle counts with backtracking walks; a cheap cousin of cycle features.
    """
    P = adj / adj.sum(1, keepdims=True)
    out = np.zeros((adj.shape[0], k), dtype=np.float64)
    M = np.eye(adj.shape[0])
    for t in range(k):
        M = M @ P
        out[:, t] = np.diag(M)
    # Standardise roughly: scale by n so values are O(1) across sizes.
    return (out * adj.shape[0]).astype(np.float32)


def build_data(g: RawGraph) -> Data:
    """Convert one ``RawGraph`` into a PyG ``Data`` object with all features."""
    n = g.num_nodes
    adj = np.zeros((n, n), dtype=np.float64)
    adj[g.edges[:, 0], g.edges[:, 1]] = 1
    adj[g.edges[:, 1], g.edges[:, 0]] = 1
    assert (adj.sum(1) == 4).all(), "completed phi^4 graphs are 4-regular"
    ei = torch.from_numpy(np.concatenate([g.edges, g.edges[:, ::-1]], 0).T.copy())
    idx_onehot = np.eye(MAX_NODES, dtype=np.float32)[:n]  # arbitrary nauty labels
    cyc = cycle_features(adj)
    return Data(
        edge_index=ei,
        num_nodes=n,
        x_label=torch.from_numpy(idx_onehot),
        x_cycle=torch.from_numpy(cyc),
        x_lap=torch.from_numpy(laplacian_pe(adj)),
        x_rwse=torch.from_numpy(rwse(adj)),
        # graph-level summaries used by non-GNN baselines
        g_cycles=torch.from_numpy(cyc).expm1().sum(0, keepdim=True) /
                 torch.arange(3, MAX_CYCLE_LEN + 1).float(),  # cycles counted once
        y=torch.tensor([np.log(g.period)], dtype=torch.float32),
        loops=torch.tensor([g.loops]),
        log_hepp=torch.tensor([np.log(g.hepp)], dtype=torch.float32),
        rel_unc=torch.tensor([g.period_unc / g.period], dtype=torch.float32),
        group=zlib.crc32(g.period_str.encode()),  # deterministic group id
    )


# --------------------------------------------------------------------------
# 3. Dataset construction + caching
# --------------------------------------------------------------------------
def load_dataset(max_loops: int = 12, min_loops: int = 3) -> list[Data]:
    """Return the list of ``Data`` objects, building the cache if needed."""
    os.makedirs(PROC_DIR, exist_ok=True)
    path = os.path.join(PROC_DIR, f"phi4_L{min_loops}-{max_loops}.pt")
    if os.path.exists(path):
        return torch.load(path, weights_only=False)
    data = []
    for L in range(min_loops, max_loops + 1):
        rows = list(iter_rows(L))
        data.extend(build_data(g) for g in rows)
        print(f"L={L:2d}: {len(rows):6d} graphs")
    torch.save(data, path)
    return data


def grouped_split(data: list[Data], frac=(0.8, 0.1, 0.1), seed: int = 0):
    """Split indices into train/val/test, keeping each period-group together.

    Graphs known to be related by period symmetries (product, twist,
    Fourier, ...) carry identical period values in the dataset; a random split
    would leak their labels.  We therefore shuffle *groups* of identical
    periods (within each loop order, so every L is represented in every split).
    """
    rng = np.random.default_rng(seed)
    loops = np.array([int(d.loops) for d in data])
    groups = np.array([d.group for d in data])
    split = np.zeros(len(data), dtype=np.int64)  # 0 train, 1 val, 2 test
    for L in np.unique(loops):
        idx = np.where(loops == L)[0]
        ug = np.unique(groups[idx])
        rng.shuffle(ug)
        cut1, cut2 = int(frac[0] * len(ug)), int((frac[0] + frac[1]) * len(ug))
        if len(ug) < 3:          # L=3,4: single graph -> train only
            cut1 = cut2 = len(ug)
        val_g, test_g = set(ug[cut1:cut2]), set(ug[cut2:])
        for i in idx:
            split[i] = 1 if groups[i] in val_g else 2 if groups[i] in test_g else 0
    return [np.where(split == s)[0] for s in range(3)]


def extrapolation_split(data: list[Data], test_loops: int = 12, val_frac: float = 0.1,
                        seed: int = 0):
    """Train/val on all graphs with L < test_loops (grouped by period value),
    test on *every* graph with L == test_loops.  Measures whether a model can
    predict periods at a loop order it has never seen."""
    loops = np.array([int(d.loops) for d in data])
    groups = np.array([d.group for d in data])
    rng = np.random.default_rng(seed)
    low = np.where(loops < test_loops)[0]
    ug = np.unique(groups[low])
    rng.shuffle(ug)
    val_g = set(ug[: int(val_frac * len(ug))])
    is_val = np.array([groups[i] in val_g for i in low])
    return low[~is_val], low[is_val], np.where(loops == test_loops)[0]


if __name__ == "__main__":
    ds = load_dataset()
    tr, va, te = grouped_split(ds)
    print(f"{len(ds)} graphs; train/val/test = {len(tr)}/{len(va)}/{len(te)}")
    print(ds[-1])
