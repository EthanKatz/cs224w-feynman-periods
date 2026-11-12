#!/bin/bash
# Extra runs added after the first grid results:
#  * 'perm' = truly arbitrary (random-permutation) vertex labels
#  * constant-feature controls without BatchNorm (BatchNorm degenerates on constant inputs)
cd "$(dirname "$0")/.."
source .venv/bin/activate
for seed in 0 1; do for m in gcn sage gin; do
  echo "$m perm $seed batch"; echo "$m const $seed none"
done; done | xargs -P ${JOBS:-3} -L 1 bash -c 'python -W ignore code/train.py --model $0 --features $1 --seed $2 --norm $3 2>&1 | grep -v -i warn'
