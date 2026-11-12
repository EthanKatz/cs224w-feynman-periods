"""Non-GNN baselines on the same grouped split as the GNNs.

  per_loop_mean    : predict the training-set mean log P of the graph's loop order
  linear_L         : log P ~ a + b L   (Balduf & Shaban's NNs never beat linear regression)
  ridge_cycles     : ridge on [one-hot(L), log1p(graph-level cycle counts 3..8)]
                     -> the same information as our per-node cycle features, but
                        pooled *before* learning (tests whether message passing helps)
  ridge_hepp       : ridge on [one-hot(L), log Hepp bound]  -- an *expensive*
                     feature (Hepp bound needs a sum over all edge orderings);
                     reported as an upper reference, not a fair competitor.
  ridge_cycles_hepp: both of the above.

Writes results/baselines.json.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
from sklearn.linear_model import LinearRegression, Ridge

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import ROOT, extrapolation_split, grouped_split, load_dataset  # noqa: E402
from train import metrics  # noqa: E402


def design(data, idx, kind, numeric_L=False):
    """Build the design matrix for baseline ``kind`` over graph indices ``idx``."""
    L = np.array([int(data[i].loops) for i in idx])
    onehot = np.eye(13)[L]
    cols = []
    if kind == "linear_L" or numeric_L:  # numeric L is needed to extrapolate in L
        cols.append(L[:, None].astype(float))
    else:
        cols.append(onehot)
    if "cycles" in kind:
        cols.append(np.log1p(np.stack([data[i].g_cycles.numpy()[0] for i in idx])))
    if "hepp" in kind:
        cols.append(np.array([float(data[i].log_hepp) for i in idx])[:, None])
    return np.concatenate(cols, 1), L


def main(seed: int = 0, split: str = "grouped"):
    data = load_dataset()
    extrap = split == "extrap"
    tr, va, te = (extrapolation_split if extrap else grouped_split)(data, seed=seed)
    y = np.array([float(d.y) for d in data])
    out = {}
    kinds = ["linear_L", "ridge_cycles", "ridge_hepp", "ridge_cycles_hepp"]
    for kind in (kinds if extrap else ["per_loop_mean"] + kinds):
        Xtr, _ = design(data, tr, kind, numeric_L=extrap)
        Xte, Lte = design(data, te, kind, numeric_L=extrap)
        model = LinearRegression() if kind in ("per_loop_mean", "linear_L") else Ridge(alpha=1e-3)
        model.fit(Xtr, y[tr])
        pred = model.predict(Xte)
        out[kind] = {"test": metrics(pred, y[te], Lte),
                     "test_pred": {"pred": pred.tolist(), "y": y[te].tolist(), "loops": Lte.tolist()}}
        t = out[kind]["test"]
        print(f"{kind:18s} mae_log={t['mae_log']:.4f} mrel={t['mrel']:.4f} "
              f"r2={t['r2']:.4f} r2_within={t['r2_within']:.3f}")
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    name = "baselines_extrap.json" if extrap else "baselines.json"
    with open(os.path.join(ROOT, "results", name), "w") as fh:
        json.dump(out, fh)


if __name__ == "__main__":
    main()
    print("--- extrapolation split: train L<=11, test L=12 ---")
    main(split="extrap")
