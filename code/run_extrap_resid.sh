#!/bin/bash
# Extrapolation runs where the GNN predicts the residual of a linear-in-L fit.
cd "$(dirname "$0")/.."
source .venv/bin/activate
for m in gcn sage gin; do for f in rwse cycle+lap+rwse; do echo "$m $f"; done; done |
  xargs -P ${JOBS:-6} -L 1 bash -c 'python -W ignore code/train.py --model $0 --features $1 --split extrap --epochs 80 --resid_L --tag _extrap_resid 2>&1 | grep -v -i warn'
