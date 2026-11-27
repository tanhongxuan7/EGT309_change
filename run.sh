#!/usr/bin/env bash
# Run the whole pipeline: prepare data -> train models -> evaluate models.
#
# Usage:
#   ./run.sh                 # leakage-free "sensor" feature set (default)
#   ./run.sh all             # "all" feature set (includes leaky derived columns, comparison only)
#   ./run.sh sensor --set tuning.enabled=false   # extra options are passed to train and evaluate
set -euo pipefail

cd "$(dirname "$0")"

FEATURE_SET="${1:-sensor}"
shift || true
PYTHON="${PYTHON:-python}"

echo "==> [1/3] Preparing data (feature set: ${FEATURE_SET})"
"$PYTHON" -m src.prepare_data --feature-set "$FEATURE_SET"

echo "==> [2/3] Training models"
"$PYTHON" -m src.train --feature-set "$FEATURE_SET" "$@"

echo "==> [3/3] Evaluating models"
"$PYTHON" -m src.evaluate --feature-set "$FEATURE_SET" "$@"

echo "Done. Models: saved_model/${FEATURE_SET}/  Reports: reports/${FEATURE_SET}/"
