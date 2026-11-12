# Predicting Feynman Periods with GNNs (CS 224W project)

Ethan Katz, Evelyn Song, Colin McKhann.

## Layout
- `code/data.py` – parse the Balduf & Shaban dataset (doi:10.5683/SP3/NLEDGH) into PyG `Data`
  objects with structural node features; grouped (leakage-free) train/val/test splits.
- `code/models.py` – shared GCN / GraphSAGE / GIN graph-regression model.
- `code/train.py` – train + evaluate one (model, node-feature) configuration.
- `code/baselines.py` – non-GNN baselines (linear on L, ridge on cycle counts, Hepp-bound reference).
- `code/run_grid.sh` – main grid (3 models x 8 feature sets x 2 seeds).
- `code/run_extra.sh` – random-permutation labels + no-BatchNorm constant controls (overwrites the diverged const runs).
- `code/run_extrap.sh`, `code/run_extrap_resid.sh` – extrapolation (train L<=11, test L=12), plain and residual-on-L.
- `code/make_figures.py` – summary table, LaTeX tables, figures.
- `milestone/` – milestone report (LaTeX + PDF). `figures/`, `results/` – outputs.

## Reproduce
```bash
uv venv --python 3.12 .venv && source .venv/bin/activate
uv pip install -r requirements.txt
mkdir -p data/raw   # then download periods_03..12 + README from Borealis (see download_data.sh)
bash code/download_data.sh
python code/data.py          # builds data/processed cache (~45 s)
python code/baselines.py
bash code/run_grid.sh        # ~50 min on a 10-core laptop CPU
bash code/run_extra.sh && bash code/run_extrap.sh && bash code/run_extrap_resid.sh
python code/make_figures.py
cd milestone && tectonic milestone.tex
```
