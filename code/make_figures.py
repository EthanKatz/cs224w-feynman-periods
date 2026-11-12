"""Aggregate run JSONs into tables and figures for the milestone report.

Outputs:
  results/summary.csv            mean/std over seeds of every (model, features)
  milestone/table_main.tex       LaTeX rows for the main results table
  figures/*.pdf                  figures used in the report
"""
from __future__ import annotations

import glob
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import ROOT, load_dataset  # noqa: E402

FIG = os.path.join(ROOT, "figures")
os.makedirs(FIG, exist_ok=True)
# Validated categorical slots 1-3 (blue, orange, aqua) for the three architectures.
COLORS = {"gcn": "#2a78d6", "sage": "#eb6834", "gin": "#1baf7a"}
NAMES = {"gcn": "GCN", "sage": "GraphSAGE", "gin": "GIN"}
FEATS = ["const", "perm", "rni", "label", "rwse", "lap", "cycle", "cycle+lap", "cycle+lap+rwse"]
FEAT_NAMES = {"const": "constant", "label": "canonical label", "perm": "random label", "rni": "random (RNI)",
              "rwse": "RWSE", "lap": "LapPE", "cycle": "cycle", "cycle+lap": "cycle+LapPE",
              "cycle+lap+rwse": "cycle+LapPE+RWSE"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
                     "axes.spines.right": False, "font.family": "serif"})


def load_runs() -> pd.DataFrame:
    rows = []
    for p in glob.glob(os.path.join(ROOT, "results", "runs", "*.json")):
        r = json.load(open(p))
        c = r["config"]
        if c.get("tag"):
            continue
        rows.append({"model": c["model"], "features": c["features"], "seed": c["seed"],
                     **{f"test_{k}": v for k, v in r["test"].items() if k != "per_loop_mrel"},
                     "val_mae_log": r["val"]["mae_log"], "time": r["train_time_s"],
                     "params": r["n_params"], "path": p})
    return pd.DataFrame(rows)


def summarize(df):
    g = df.groupby(["model", "features"])
    s = g[["test_mae_log", "test_mrel", "test_r2_within", "time"]].agg(["mean", "std"])
    s["n_seeds"] = g.size()
    s.columns = ["_".join(c).rstrip("_") if isinstance(c, tuple) else c for c in s.columns]
    s.to_csv(os.path.join(ROOT, "results", "summary.csv"))
    return s.reset_index()


def latex_table(s, base):
    lines = []
    for f in FEATS:
        cells = []
        for m in ["gcn", "sage", "gin"]:
            r = s[(s.model == m) & (s.features == f)]
            if len(r) == 0:
                cells.append("--")
                continue
            r = r.iloc[0]
            cells.append(f"{100*r.test_mrel_mean:.2f} $\\pm$ {100*np.nan_to_num(r.test_mrel_std):.2f}"
                         f" & {r.test_r2_within_mean:.4f}")
        lines.append(f"{FEAT_NAMES[f]} & " + " & ".join(cells) + r" \\")
    with open(os.path.join(ROOT, "milestone", "table_main.tex"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    with open(os.path.join(ROOT, "milestone", "table_base.tex"), "w") as fh:
        names = {"linear_L": "Linear regression on $L$", "ridge_cycles":
                 "Ridge on $L$ + graph-level cycle counts",
                 "ridge_hepp": "Ridge on $L$ + $\\log$ Hepp bound$^\\dagger$",
                 "ridge_cycles_hepp": "Ridge on $L$ + cycles + Hepp$^\\dagger$"}
        for k, n in names.items():
            t = base[k]["test"]
            fh.write(f"{n} & {100*t['mrel']:.2f} & {t['r2_within']:.4f} \\\\\n")


def fig_data_overview():
    """Distribution of log P per loop order + Hepp-bound correlation."""
    data = load_dataset()
    L = np.array([int(d.loops) for d in data])
    y = np.array([float(d.y) for d in data])
    hepp = np.array([float(d.log_hepp) for d in data])
    fig, ax = plt.subplots(1, 2, figsize=(6.5, 2.4))
    Ls = np.unique(L)
    ax[0].violinplot([y[L == l] for l in Ls if (L == l).sum() > 1],
                     positions=[l for l in Ls if (L == l).sum() > 1], widths=0.8,
                     showextrema=False)
    for b in ax[0].collections:
        b.set_facecolor(COLORS["gcn"]); b.set_alpha(0.5)
    ax[0].scatter(Ls, [y[L == l].mean() for l in Ls], s=10, color=INK, zorder=3, label="mean")
    ax[0].set_xlabel("loop number $L$"); ax[0].set_ylabel(r"$\log\,\mathcal{P}(G)$")
    ax[0].set_xticks(Ls); ax[0].grid(axis="y", color=GRID, lw=0.6)
    ax[0].set_title("(a) periods grow exponentially in $L$", fontsize=9)
    m = L == 12
    ax[1].scatter(hepp[m], y[m], s=1, alpha=0.15, color=COLORS["gcn"], rasterized=True)
    ax[1].set_xlabel(r"$\log$ Hepp bound"); ax[1].set_ylabel(r"$\log\,\mathcal{P}(G)$")
    ax[1].set_title(f"(b) $L=12$ ({m.sum():,} graphs)", fontsize=9)
    ax[1].grid(color=GRID, lw=0.6)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "data_overview.pdf"))
    plt.close(fig)


def fig_ablation(s, base):
    """Grouped bars: test mean relative error per feature set and architecture."""
    fig, ax = plt.subplots(figsize=(6.5, 2.6))
    feats = [f for f in FEATS if (s.features == f).any()]
    x = np.arange(len(feats)); w = 0.26
    for i, m in enumerate(["gcn", "sage", "gin"]):
        vals, errs = [], []
        for f in feats:
            r = s[(s.model == m) & (s.features == f)]
            vals.append(100 * r.test_mrel_mean.iloc[0] if len(r) else np.nan)
            errs.append(100 * np.nan_to_num(r.test_mrel_std.iloc[0]) if len(r) else 0)
        ax.bar(x + (i - 1) * w, vals, w - 0.03, yerr=errs, color=COLORS[m], label=NAMES[m],
               error_kw={"lw": 0.8, "ecolor": MUTED})
    for k, ls, lab in [("linear_L", "--", "linear on $L$"),
                       ("ridge_cycles", ":", "ridge on graph cycle counts")]:
        v = 100 * base[k]["test"]["mrel"]
        ax.axhline(v, color=MUTED, ls=ls, lw=1)
        ax.text(len(feats) - 0.45, v, lab, ha="right", va="bottom", fontsize=7.5, color=MUTED)
    ax.set_yscale("log"); ax.set_axisbelow(True)
    ax.set_xticks(x, [FEAT_NAMES[f] for f in feats], rotation=20, ha="right")
    ax.set_ylabel("test mean rel. error (%)")
    ax.grid(axis="y", color=GRID, lw=0.6, which="both")
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.18), fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "ablation.pdf"))
    plt.close(fig)


def fig_scatter(df, base):
    """Predicted vs. true log P at L=12 (per-loop mean removed) for the best GNN."""
    best = df[df.seed == 0].sort_values("val_mae_log").iloc[0]
    r = json.load(open(best.path))["test_pred"]
    fig, ax = plt.subplots(1, 2, figsize=(6.5, 2.6), sharex=True, sharey=True)
    for a, (title, tp) in zip(ax, [("ridge on graph cycle counts", base["ridge_cycles"]["test_pred"]),
                                   (f"{NAMES[best.model]} + {FEAT_NAMES[best.features]}", r)]):
        y, p, L = map(np.array, (tp["y"], tp["pred"], tp["loops"]))
        m = L == 12
        mu = y[m].mean()
        a.scatter(y[m] - mu, p[m] - mu, s=1, alpha=0.2, color=COLORS["gcn"], rasterized=True)
        lim = [(y[m] - mu).min(), (y[m] - mu).max()]
        a.plot(lim, lim, color=INK, lw=0.8)
        mrel = np.abs(np.expm1(p[m] - y[m])).mean()
        a.set_title(f"{title}\nL=12 test, mean rel. err. {100*mrel:.1f}%", fontsize=8.5)
        a.set_xlabel(r"true $\log\mathcal{P} - \overline{\log\mathcal{P}}_{L=12}$")
        a.grid(color=GRID, lw=0.6)
    ax[0].set_ylabel("predicted (same offset)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "scatter_best.pdf"))
    plt.close(fig)
    return best


def fig_curves(df):
    """Validation curves for GIN with each feature set (seed 0)."""
    fig, ax = plt.subplots(figsize=(6.0, 2.2))
    sub = df[(df.model == "gin") & (df.seed == 0)]
    styles = {"const": (MUTED, "--"), "perm": (MUTED, ":"), "label": (COLORS["sage"], "--"), "cycle": (COLORS["gin"], "-"),
              "cycle+lap": (COLORS["gcn"], "-")}
    for f, (c, ls) in styles.items():
        r = sub[sub.features == f]
        if len(r) == 0:
            continue
        h = json.load(open(r.iloc[0].path))["history"]
        ax.plot([e["val_mae_log"] for e in h], color=c, ls=ls, lw=1.5, label=FEAT_NAMES[f])
    ax.set_yscale("log"); ax.set_xlabel("epoch"); ax.set_ylabel(r"val MAE of $\log\mathcal{P}$")
    ax.grid(color=GRID, lw=0.6, which="major"); ax.set_axisbelow(True)
    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.legend(frameon=False, fontsize=7.5, loc="center left", bbox_to_anchor=(1.01, 0.5))
    ax.set_title("GIN validation curves", fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "curves_gin.pdf")); plt.close(fig)


def extrap_table():
    """LaTeX rows for the extrapolation experiment (train L<=11, test L=12)."""
    base = json.load(open(os.path.join(ROOT, "results", "baselines_extrap.json")))
    lines = [r"Linear regression on $L$ & " + f"{100*base['linear_L']['test']['mrel']:.1f} & "
             f"{base['linear_L']['test']['r2_within']:.4f} \\\\",
             r"Ridge on $L$ + graph-level cycle counts & " +
             f"{100*base['ridge_cycles']['test']['mrel']:.1f} & "
             f"{base['ridge_cycles']['test']['r2_within']:.4f} \\\\",
             r"Ridge on $L$ + $\log$ Hepp bound$^\dagger$ & " +
             f"{100*base['ridge_hepp']['test']['mrel']:.2f} & "
             f"{base['ridge_hepp']['test']['r2_within']:.4f} \\\\", r"\midrule"]
    rows = []
    for p in sorted(glob.glob(os.path.join(ROOT, "results", "runs", "*_extrap*.json"))):
        r = json.load(open(p)); c = r["config"]
        rows.append((c["model"], c["features"], bool(c.get("resid_L")),
                     r["test"]["mrel"], r["test"]["r2_within"]))
    order = {m: i for i, m in enumerate(["gcn", "sage", "gin"])}
    for resid in (False, True):
        if resid:
            lines.append(r"\midrule\multicolumn{3}{l}{\emph{GNN predicts residual of linear fit on $L$}} \\")
        for m, f, _, e, r2 in sorted([t for t in rows if t[2] == resid],
                                     key=lambda t: (FEATS.index(t[1]), order[t[0]])):
            lines.append(f"{NAMES[m]} + {FEAT_NAMES[f]} & {100*e:.1f} & {r2:.3f} \\\\")
    with open(os.path.join(ROOT, "milestone", "table_extrap.tex"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    base = json.load(open(os.path.join(ROOT, "results", "baselines.json")))
    df = load_runs()
    s = summarize(df)
    pd.set_option("display.width", 200)
    print(s[["model", "features", "n_seeds", "test_mrel_mean", "test_mrel_std",
             "test_r2_within_mean", "time_mean"]].sort_values("test_mrel_mean").to_string())
    latex_table(s, base)
    fig_data_overview()
    fig_ablation(s, base)
    best = fig_scatter(df, base)
    print("best by val:", best.model, best.features)
    fig_curves(df)
    if glob.glob(os.path.join(ROOT, "results", "runs", "*_extrap.json")):
        extrap_table()
