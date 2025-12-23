#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if command -v conda >/dev/null 2>&1; then
  CONDA_BASE="$(conda info --base)"
  # shellcheck disable=SC1090
  source "$CONDA_BASE/etc/profile.d/conda.sh"
  conda activate torch-2.9.0
else
  echo "conda not found; please activate torch-2.9.0 manually." >&2
  exit 1
fi

export PYTHONPATH="$ROOT"
export MPLCONFIGDIR="$ROOT/.mplconfig"
export XDG_CACHE_HOME="$ROOT/.cache"

mkdir -p "$ROOT/.mplconfig" "$ROOT/.cache" "$ROOT/data/processed/minitest" "$ROOT/checkpoints" "$ROOT/predictions"

ATOM_COUNT=20
TRAIN_SAMPLES=100
TEST_SAMPLES=1000
NUM_CONFS=1
EPOCHS=10

prepare_split() {
  local split="$1"
  local prefix="$2"
  local max_samples="$3"
  if [[ -n "${DATA_DIR:-}" ]]; then
    python scripts/prepare_data.py \
      --data-dir "$DATA_DIR" \
      --split "$split" \
      --max-samples "$max_samples" \
      --atom-count "$ATOM_COUNT" \
      --allowed-elements C,H,O,N \
      --out-dir data/processed/minitest \
      --prefix "$prefix" \
      --shard-size "$max_samples"
  else
    python scripts/prepare_data.py \
      --split "$split" \
      --max-samples "$max_samples" \
      --atom-count "$ATOM_COUNT" \
      --allowed-elements C,H,O,N \
      --out-dir data/processed/minitest \
      --prefix "$prefix" \
      --shard-size "$max_samples"
  fi
}

echo "[1/6] prepare_data (train=${TRAIN_SAMPLES}, test=prebuilt)"
prepare_split train train_chon20 "$TRAIN_SAMPLES"
# prepare_split test test_chon12 "$TEST_SAMPLES"

TRAIN_MANIFEST="data/processed/minitest/train_chon20_manifest.json"
TEST_MANIFEST="data/processed/minitest/test_chon20_manifest.json"

echo "[2/6] train baselines and GNNs"
python scripts/train_knn.py \
  --manifest "$TRAIN_MANIFEST" \
  --output checkpoints/minitest_knn.pkl

python scripts/train_distance_regressor.py \
  --manifest "$TRAIN_MANIFEST" \
  --atom-count "$ATOM_COUNT" \
  --output checkpoints/minitest_distance_regressor.joblib

python scripts/train_mpnn.py \
  --manifest "$TRAIN_MANIFEST" \
  --atom-count "$ATOM_COUNT" \
  --epochs "$EPOCHS" \
  --output checkpoints/minitest_mpnn.pt

python scripts/train_egnn.py \
  --manifest "$TRAIN_MANIFEST" \
  --atom-count "$ATOM_COUNT" \
  --epochs "$EPOCHS" \
  --output checkpoints/minitest_egnn.pt

echo "[3/6] predict structures"
python scripts/predict_structures.py \
  --method etkdg \
  --manifest "$TEST_MANIFEST" \
  --num-confs "$NUM_CONFS" \
  --output predictions/minitest_etkdg.npz

python scripts/predict_structures.py \
  --method knn \
  --checkpoint checkpoints/minitest_knn.pkl \
  --manifest "$TEST_MANIFEST" \
  --output predictions/minitest_knn.npz

python scripts/predict_structures.py \
  --method distance_regressor \
  --checkpoint checkpoints/minitest_distance_regressor.joblib \
  --manifest "$TEST_MANIFEST" \
  --output predictions/minitest_distance_regressor.npz

python scripts/predict_structures.py \
  --method mpnn \
  --checkpoint checkpoints/minitest_mpnn.pt \
  --manifest "$TEST_MANIFEST" \
  --output predictions/minitest_mpnn.npz

python scripts/predict_structures.py \
  --method egnn \
  --checkpoint checkpoints/minitest_egnn.pt \
  --manifest "$TEST_MANIFEST" \
  --output predictions/minitest_egnn.npz

echo "[4/6] evaluate predictions (RDKit symmetry RMSD)"
python scripts/eval_predictions.py \
  --predictions predictions/minitest_etkdg.npz \
  --manifest "$TEST_MANIFEST"

python scripts/eval_predictions.py \
  --predictions predictions/minitest_knn.npz \
  --manifest "$TEST_MANIFEST"

python scripts/eval_predictions.py \
  --predictions predictions/minitest_distance_regressor.npz \
  --manifest "$TEST_MANIFEST"

python scripts/eval_predictions.py \
  --predictions predictions/minitest_mpnn.npz \
  --manifest "$TEST_MANIFEST"

python scripts/eval_predictions.py \
  --predictions predictions/minitest_egnn.npz \
  --manifest "$TEST_MANIFEST"

echo "[5/6] done"
echo "Manifests: $TRAIN_MANIFEST , $TEST_MANIFEST"
echo "Predictions: predictions/minitest_*.npz"
echo "Plots: predictions/minitest_*_rmsd.png"
