#!/usr/bin/env bash
# Video -> ST-maps + manifest. CPU-bound and the long pole: hours for the full corpus set.
# Safe to re-run: sessions whose .npz already exists are skipped.
set -euo pipefail
cd "$(dirname "$0")/.."
: "${FLOWRR_DATASETS:?set FLOWRR_DATASETS to the directory holding the corpora}"
python -m flowrr.preprocess.build_stmaps --jobs "${JOBS:-24}" "$@"
