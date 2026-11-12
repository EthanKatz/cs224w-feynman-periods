#!/bin/bash
# Runs the milestone experiment grid: 3 architectures x 8 node-feature sets x 2 seeds.
# Each job is single-threaded; 9 jobs run concurrently (M1 Max, 10 cores).
cd "$(dirname "$0")/.."
source .venv/bin/activate
for seed in 0 1; do for f in const label rni rwse cycle lap cycle+lap cycle+lap+rwse; do for m in gcn sage gin; do
  echo "$m $f $seed"
done; done; done | xargs -P 9 -L 1 bash -c 'python -W ignore code/train.py --model $0 --features $1 --seed $2 2>&1 | grep -v -i warn' 
