#!/bin/bash
# Extrapolation in loop order: train/val on L<=11 (~11k graphs), test on all 81,305 L=12 graphs.
cd "$(dirname "$0")/.."
source .venv/bin/activate
for m in gcn sage gin; do for f in label rwse cycle cycle+lap+rwse; do
  echo "$m $f"
done; done | xargs -P ${JOBS:-9} -L 1 bash -c 'python -W ignore code/train.py --model $0 --features $1 --split extrap --epochs 80 --tag _extrap 2>&1 | grep -v -i warn'
