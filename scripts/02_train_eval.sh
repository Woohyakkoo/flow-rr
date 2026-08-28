#!/usr/bin/env bash
# Train one model and score it on both labelled test splits.
#   TRAIN=cohface,pure SEEDS="42 1 2 3" bash scripts/02_train_eval.sh
set -euo pipefail
cd "$(dirname "$0")/.."
TRAIN="${TRAIN:-cohface,pure}"
EVAL="${EVAL:-cohface,mahnob}"
TAG="${TAG:-run}"
for s in ${SEEDS:-42}; do
  echo "===== $TRAIN  seed $s ====="
  python -m flowrr.train --train "$TRAIN" --eval "$EVAL" --seed "$s" --tag "$TAG"
done
