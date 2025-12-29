#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
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

RUN_ROOT="${RUN_ROOT:-work/compare_egnn}"
TRAIN_MANIFEST="${TRAIN_MANIFEST:-data/processed/work/train_chon20_manifest.json}"
VAL_MANIFEST="${VAL_MANIFEST:-data/processed/work/test_chon20_manifest.json}"
ATOM_COUNT="${ATOM_COUNT:-20}"
ELEMENTS="${ELEMENTS:-C,H,O,N}"
MAX_DEGREE="${MAX_DEGREE:-6}"
DEVICE="${DEVICE:-cuda}"
SEED="${SEED:-0}"
MAX_EVAL="${MAX_EVAL:-0}"
PRED_SAMPLES="${PRED_SAMPLES:-$MAX_EVAL}"
EPOCHS="${EPOCHS:-100}"
LR="${LR:-1e-4}"
HIDDEN_DIM="${HIDDEN_DIM:-128}"
NUM_LAYERS_STR="${NUM_LAYERS_STR:-4 5}"
TRAIN_SIZES_STR="${TRAIN_SIZES_STR:-500 1000 2000 5000 6151}"
BATCH_SIZE="${BATCH_SIZE:-32}"
DO_EVAL="${DO_EVAL:-1}"
DO_PLOT="${DO_PLOT:-0}"
RESUME="${RESUME:-0}"
OVERWRITE="${OVERWRITE:-0}"

read -r -a NUM_LAYERS_LIST <<< "$NUM_LAYERS_STR"
read -r -a TRAIN_SIZES <<< "$TRAIN_SIZES_STR"

mkdir -p "$ROOT/.mplconfig" "$ROOT/.cache"

LOG_PATH="$RUN_ROOT/results.jsonl"
if [[ "$OVERWRITE" == "1" ]]; then
  : > "$LOG_PATH"
fi

if [[ ! -f "$TRAIN_MANIFEST" ]]; then
  echo "Train manifest not found: $TRAIN_MANIFEST" >&2
  exit 1
fi
if [[ "$DO_EVAL" == "1" && ! -f "$VAL_MANIFEST" ]]; then
  echo "Val manifest not found: $VAL_MANIFEST" >&2
  exit 1
fi

train_total="$(python - <<PY
import json
from pathlib import Path

data = json.load(Path("$TRAIN_MANIFEST").open())
total = data.get("total_kept")
if total is None:
    shards = data.get("shards", [])
    if isinstance(shards, list):
        total = sum(int(shard.get("count", 0)) for shard in shards)
try:
    total = int(total)
except (TypeError, ValueError):
    total = 0
print(total)
PY
)"
if [[ "$train_total" -gt 0 ]]; then
  filtered=()
  for size in "${TRAIN_SIZES[@]}"; do
    if (( size <= train_total )); then
      filtered+=("$size")
    else
      echo "[skip] train_size=$size exceeds total_kept=$train_total" >&2
    fi
  done
  if [[ "${#filtered[@]}" -eq 0 ]]; then
    echo "No train sizes <= total_kept ($train_total)." >&2
    exit 1
  fi
  TRAIN_SIZES=("${filtered[@]}")
fi

ATOM_ARGS=()
if [[ -n "${ATOM_COUNT:-}" ]]; then
  ATOM_ARGS+=(--atom-count "$ATOM_COUNT")
fi
ELEMENT_ARGS=()
if [[ -n "${ELEMENTS:-}" ]]; then
  ELEMENT_ARGS+=(--elements "$ELEMENTS")
fi

sanitize() {
  printf '%s' "$1" | sed 's/[.]/p/g; s/[^A-Za-z0-9]/_/g'
}

lr_tag="$(sanitize "$LR")"

for num_layers in "${NUM_LAYERS_LIST[@]}"; do
  for train_size in "${TRAIN_SIZES[@]}"; do
    run_tag="egnn_hd${HIDDEN_DIM}_l${num_layers}_lr${lr_tag}_ep${EPOCHS}_n${train_size}"
    run_dir="$RUN_ROOT/${run_tag}"
    ckpt_path="$run_dir/${run_tag}.pt"
    train_log="$run_dir/${run_tag}_train.json"
    pred_path="$run_dir/${run_tag}.npz"
    metrics_path="$run_dir/${run_tag}_metrics.json"
    per_sample_path="$run_dir/${run_tag}_per_sample.jsonl"

    if [[ "$RESUME" == "1" && -f "$metrics_path" ]]; then
      echo "[skip] $run_tag"
      continue
    fi

    mkdir -p "$run_dir"
    echo "[train] $run_tag"
    python scripts/train_egnn.py \
      --manifest "$TRAIN_MANIFEST" \
      "${ATOM_ARGS[@]}" \
      "${ELEMENT_ARGS[@]}" \
      --max-degree "$MAX_DEGREE" \
      --max-train "$train_size" \
      --hidden-dim "$HIDDEN_DIM" \
      --num-layers "$num_layers" \
      --epochs "$EPOCHS" \
      --lr "$LR" \
      --batch-size "$BATCH_SIZE" \
      --device "$DEVICE" \
      --seed "$SEED" \
      --output "$ckpt_path" \
      > "$train_log"

    if [[ "$DO_EVAL" == "1" ]]; then
      python scripts/predict_structures.py \
        --method egnn \
        --checkpoint "$ckpt_path" \
        --manifest "$VAL_MANIFEST" \
        --max-samples "$PRED_SAMPLES" \
        --output "$pred_path" \
        --device "$DEVICE"

      eval_args=(
        --predictions "$pred_path"
        --manifest "$VAL_MANIFEST"
        --max-eval "$MAX_EVAL"
        --output "$metrics_path"
        --per-sample-output "$per_sample_path"
      )
      if [[ "$DO_PLOT" != "1" ]]; then
        eval_args+=(--no-plot)
      fi
      python scripts/eval_predictions.py "${eval_args[@]}"

      python - <<PY >> "$LOG_PATH"
import json
train = json.load(open("$train_log"))
metrics = json.load(open("$metrics_path"))
out = {
    "run_tag": "$run_tag",
    "model": "egnn",
    "train_size_requested": int("$train_size"),
    "hidden_dim": int("$HIDDEN_DIM"),
    "num_layers": int("$num_layers"),
    "epochs": int("$EPOCHS"),
    "lr": float("$LR"),
    "max_degree": int("$MAX_DEGREE"),
    "metrics": metrics.get("metrics", {}),
    "train_log": train,
    "metrics_path": "$metrics_path",
    "checkpoint": "$ckpt_path",
}
print(json.dumps(out))
PY
    else
      python - <<PY >> "$LOG_PATH"
import json
train = json.load(open("$train_log"))
out = {
    "run_tag": "$run_tag",
    "model": "egnn",
    "train_size_requested": int("$train_size"),
    "hidden_dim": int("$HIDDEN_DIM"),
    "num_layers": int("$num_layers"),
    "epochs": int("$EPOCHS"),
    "lr": float("$LR"),
    "max_degree": int("$MAX_DEGREE"),
    "train_log": train,
    "checkpoint": "$ckpt_path",
}
print(json.dumps(out))
PY
    fi
  done
done
