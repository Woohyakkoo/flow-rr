#!/usr/bin/env bash
# Subject-level 7:1:2 splits, seeded. Seconds. Re-running gives the identical split.
set -euo pipefail
cd "$(dirname "$0")/.."
python -m flowrr.splits "$@"
