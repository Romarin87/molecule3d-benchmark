#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ -n "${CONDA_ENV:-}" ]] && command -v conda >/dev/null 2>&1; then
  CONDA_BASE="$(conda info --base)"
  # shellcheck disable=SC1090
  source "$CONDA_BASE/etc/profile.d/conda.sh"
  conda activate "$CONDA_ENV"
fi

export PYTHONPATH="$ROOT"
export MPLCONFIGDIR="$ROOT/.mplconfig"
export XDG_CACHE_HOME="$ROOT/.cache"

OUT_DIR="${OUT_DIR:-data/processed/work}"
ATOM_COUNT="${ATOM_COUNT:-20}"
ALLOWED_ELEMENTS="${ALLOWED_ELEMENTS:-C,H,O,N}"
TRAIN_SAMPLES="${TRAIN_SAMPLES:-100000}"
TRAIN_SPLIT="${TRAIN_SPLIT:-train}"
TRAIN_PREFIX="${TRAIN_PREFIX:-train_chon${ATOM_COUNT}}"
TRAIN_SHARD_SIZE="${TRAIN_SHARD_SIZE:-50000}"

mkdir -p "$ROOT/.mplconfig" "$ROOT/.cache" "$OUT_DIR"

prepare_split() {
  local split="$1"
  local prefix="$2"
  local max_samples="$3"
  local shard_size="$4"

  local args=(
    --split "$split"
    --max-samples "$max_samples"
    --out-dir "$OUT_DIR"
    --prefix "$prefix"
    --shard-size "$shard_size"
  )

  if [[ -n "${ATOM_COUNT:-}" ]]; then
    args+=(--atom-count "$ATOM_COUNT")
  fi
  if [[ -n "${ALLOWED_ELEMENTS:-}" ]]; then
    args+=(--allowed-elements "$ALLOWED_ELEMENTS")
  fi
  if [[ -n "${DATA_DIR:-}" ]]; then
    args=(--data-dir "$DATA_DIR" "${args[@]}")
  fi

  python scripts/prepare_data.py "${args[@]}"
}

echo "[1/2] prepare train split"
prepare_split "$TRAIN_SPLIT" "$TRAIN_PREFIX" "$TRAIN_SAMPLES" "$TRAIN_SHARD_SIZE"

TRAIN_MANIFEST="$OUT_DIR/${TRAIN_PREFIX}_manifest.json"

echo "Train manifest: $TRAIN_MANIFEST"
