"""Train / evaluate one (architecture, node-feature) configuration.

Example:
    python code/train.py --model gin --features cycle+lap --seed 0

Writes ``results/runs/<model>_<features>_s<seed>.json`` with metrics and the
test-set predictions (used by ``make_figures.py``).

Node-feature options (combine with '+'):
    const  - a constant 1 on every node (what a 'structure-free' GNN sees)
    label  - one-hot of the dataset's vertex label.  NOTE: these are nauty
             *canonical* labels, so label + edges determine the isomorphism
             class: a strong (non-equivariant) structural signal, not noise.
    perm   - one-hot of a uniformly random vertex permutation, resampled every
             forward pass: truly arbitrary vertex labels
    rni    - random node initialisation, resampled every forward pass
    cycle  - per-node counts of simple cycles of length 3..8 (log1p)  [ours]
    lap    - 8 Laplacian eigenvector PEs, random sign flips in training [ours]
    rwse   - random-walk return probabilities, 12 steps               [ours]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import ROOT, extrapolation_split, grouped_split, load_dataset  # noqa: E402
from models import PeriodGNN  # noqa: E402

FEATURE_DIMS = {"const": 1, "label": 14, "perm": 14, "rni": 8, "cycle": 6, "lap": 8, "rwse": 12}


def node_features(batch, features: list[str], train: bool) -> torch.Tensor:
    """Assemble the node-feature matrix for a mini-batch."""
    parts = []
    n = batch.num_nodes
    for f in features:
        if f == "const":
            parts.append(torch.ones(n, 1))
        elif f == "rni":
            parts.append(torch.randn(n, FEATURE_DIMS["rni"]))
        elif f == "perm":
            # random rank of each node within its graph -> one-hot label
            order = torch.argsort(batch.batch.double() + torch.rand(n, dtype=torch.float64))
            pos = torch.empty(n, dtype=torch.long)
            pos[order] = torch.arange(n) - batch.ptr[batch.batch[order]]
            parts.append(torch.nn.functional.one_hot(pos, FEATURE_DIMS["perm"]).float())
        elif f == "lap":
            pe = batch.x_lap
            if train:  # eigenvector sign ambiguity: random flip per graph & vector
                sign = torch.randint(0, 2, (batch.num_graphs, pe.size(1))) * 2 - 1
                pe = pe * sign[batch.batch]
            parts.append(pe)
        else:
            parts.append(batch[f"x_{f}"])
    return torch.cat(parts, dim=-1)


def metrics(pred_log: np.ndarray, y_log: np.ndarray, loops: np.ndarray) -> dict:
    """Regression metrics on log-period predictions.

    mae_log/rmse_log : errors of log P (mae_log ~ typical relative error)
    mrel             : mean |P_hat - P| / P, the metric reported by Balduf & Shaban
    r2               : ordinary R^2 of log P (dominated by the loop number)
    r2_within        : R^2 after removing the per-loop mean of the *true* values,
                       i.e. how much of the within-loop spread is explained.
    """
    err = pred_log - y_log
    resid = y_log.copy()
    for L in np.unique(loops):
        m = loops == L
        resid[m] -= y_log[m].mean()
    out = {
        "mae_log": float(np.abs(err).mean()),
        "rmse_log": float(np.sqrt((err ** 2).mean())),
        "mrel": float(np.abs(np.expm1(err)).mean()),
        "r2": float(1 - (err ** 2).sum() / ((y_log - y_log.mean()) ** 2).sum()),
        "r2_within": float(1 - (err ** 2).sum() / (resid ** 2).sum()),
    }
    out["per_loop_mrel"] = {int(L): float(np.abs(np.expm1(err[loops == L])).mean())
                            for L in np.unique(loops)}
    return out


@torch.no_grad()
def predict(model, loader, features, mu, sd, lin=(0.0, 0.0)):
    model.eval()
    preds, ys, loops = [], [], []
    for b in loader:
        out = model(node_features(b, features, False), b.edge_index, b.batch)
        preds.append(out * sd + mu + lin[0] * b.loops + lin[1])
        ys.append(b.y)
        loops.append(b.loops)
    return (torch.cat(preds).numpy(), torch.cat(ys).numpy(), torch.cat(loops).numpy())


def run(args) -> dict:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(args.threads)
    features = args.features.split("+")
    data = load_dataset(max_loops=args.max_loops)
    if args.split == "extrap":  # train L<=11, test on all L=12 graphs
        tr, va, te = extrapolation_split(data, seed=args.split_seed)
    else:
        tr, va, te = grouped_split(data, seed=args.split_seed)
    if args.train_frac < 1.0:  # learning-curve experiments
        rng = np.random.default_rng(args.seed)
        tr = rng.choice(tr, int(len(tr) * args.train_frac), replace=False)
    y_tr = torch.cat([data[i].y for i in tr])
    # Optional: learn only the residual of a linear-in-L fit, so the network does
    # not have to extrapolate the (trivial) exponential growth of P with L itself.
    lin = (0.0, 0.0)
    if args.resid_L:
        L_tr = np.array([int(data[i].loops) for i in tr])
        lin = tuple(float(c) for c in np.polyfit(L_tr, y_tr.numpy(), 1))
        y_tr = y_tr - (lin[0] * torch.from_numpy(L_tr).float() + lin[1])
    mu, sd = float(y_tr.mean()), float(y_tr.std())

    loader = lambda idx, shuffle: DataLoader([data[i] for i in idx], batch_size=args.batch,
                                             shuffle=shuffle)
    tr_loader, va_loader, te_loader = loader(tr, True), loader(va, False), loader(te, False)

    in_dim = sum(FEATURE_DIMS[f] for f in features)
    model = PeriodGNN(in_dim, args.model, args.hidden, args.layers, args.dropout, args.norm)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    best, best_state, history = float("inf"), None, []
    t0 = time.time()
    for epoch in range(args.epochs):
        model.train()
        tot = 0.0
        for b in tr_loader:
            opt.zero_grad()
            out = model(node_features(b, features, True), b.edge_index, b.batch)
            target = b.y - (lin[0] * b.loops + lin[1])
            loss = F.mse_loss(out, (target - mu) / sd)
            loss.backward()
            opt.step()
            tot += loss.item() * b.num_graphs
        sched.step()
        p, y, l = predict(model, va_loader, features, mu, sd, lin)
        val_mae = float(np.abs(p - y).mean())
        history.append({"epoch": epoch, "train_mse_std": tot / len(tr), "val_mae_log": val_mae})
        if val_mae < best:  # model selection on validation MAE of log P
            best = val_mae
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        if args.verbose:
            print(f"ep {epoch:3d} train {tot/len(tr):.4f} val_mae {val_mae:.4f} "
                  f"({time.time()-t0:.0f}s)", flush=True)

    model.load_state_dict(best_state)
    res = {"config": vars(args), "n_params": sum(p.numel() for p in model.parameters()),
           "train_time_s": time.time() - t0, "history": history}
    for name, ld in [("val", va_loader), ("test", te_loader)]:
        p, y, l = predict(model, ld, features, mu, sd, lin)
        res[name] = metrics(p, y, l)
        if name == "test":
            res["test_pred"] = {"pred": p.tolist(), "y": y.tolist(), "loops": l.tolist()}
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gin", choices=["gcn", "sage", "gin"])
    ap.add_argument("--features", default="cycle")
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--norm", default="batch", choices=["batch", "none"])
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--split_seed", type=int, default=0)
    ap.add_argument("--max_loops", type=int, default=12)
    ap.add_argument("--split", default="grouped", choices=["grouped", "extrap"])
    ap.add_argument("--resid_L", action="store_true",
                    help="predict residual of a linear fit of log P on L")
    ap.add_argument("--train_frac", type=float, default=1.0)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--tag", default="")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    res = run(args)
    out_dir = os.path.join(ROOT, "results", "runs")
    os.makedirs(out_dir, exist_ok=True)
    name = f"{args.model}_{args.features}_s{args.seed}{args.tag}.json"
    with open(os.path.join(out_dir, name), "w") as fh:
        json.dump(res, fh)
    t = res["test"]
    print(f"{name}: test mae_log={t['mae_log']:.4f} mrel={t['mrel']:.4f} "
          f"r2_within={t['r2_within']:.3f} ({res['train_time_s']:.0f}s)")


if __name__ == "__main__":
    main()
