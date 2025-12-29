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

RUN_ROOT="${RUN_ROOT:-work/compare_distance_regressor}"
TRAIN_MANIFEST="${TRAIN_MANIFEST:-data/processed/work/train_chon20_manifest.json}"
VAL_MANIFEST="${VAL_MANIFEST:-data/processed/work/test_chon20_manifest.json}"
ATOM_COUNT="${ATOM_COUNT:-20}"
ELEMENTS="${ELEMENTS:-C,H,O,N}"
MAX_DEGREE="${MAX_DEGREE:-6}"
MODEL_LIST_STR="${MODEL_LIST_STR:-rf}"
SEED="${SEED:-0}"
MAX_EVAL="${MAX_EVAL:-0}"
PRED_SAMPLES="${PRED_SAMPLES:-$MAX_EVAL}"
TRAIN_SIZES_STR="${TRAIN_SIZES_STR:-500 1000 2000 5000 6151}"
USE_MMFF="${USE_MMFF:-0}"
DO_EVAL="${DO_EVAL:-1}"
DO_PLOT="${DO_PLOT:-0}"
RESUME="${RESUME:-0}"
OVERWRITE="${OVERWRITE:-0}"

read -r -a MODEL_LIST <<< "$MODEL_LIST_STR"
read -r -a TRAIN_SIZES <<< "$TRAIN_SIZES_STR"

mkdir -p "$ROOT/.mplconfig" "$ROOT/.cache"
mkdir -p "$RUN_ROOT"

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
if [[ -z "${ATOM_COUNT:-}" ]]; then
  echo "ATOM_COUNT is required for distance regressor." >&2
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

ELEMENT_ARGS=()
if [[ -n "${ELEMENTS:-}" ]]; then
  ELEMENT_ARGS+=(--elements "$ELEMENTS")
fi

mmff_tag="mmff${USE_MMFF}"

for model_name in "${MODEL_LIST[@]}"; do
  for train_size in "${TRAIN_SIZES[@]}"; do
    run_tag="distreg_${model_name}_md${MAX_DEGREE}_${mmff_tag}_n${train_size}"
    run_dir="$RUN_ROOT/${run_tag}"
    ckpt_path="$run_dir/${run_tag}.joblib"
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
    python scripts/train_distance_regressor.py \
      --manifest "$TRAIN_MANIFEST" \
      --atom-count "$ATOM_COUNT" \
      "${ELEMENT_ARGS[@]}" \
      --max-degree "$MAX_DEGREE" \
      --max-train "$train_size" \
      --model "$model_name" \
      --seed "$SEED" \
      --output "$ckpt_path" \
      > "$train_log"

    if [[ "$DO_EVAL" == "1" ]]; then
      pred_args=(
        --method distance_regressor
        --checkpoint "$ckpt_path"
        --manifest "$VAL_MANIFEST"
        --max-samples "$PRED_SAMPLES"
        --output "$pred_path"
      )
      if [[ "$USE_MMFF" == "1" ]]; then
        pred_args+=(--use-mmff)
      fi
      python scripts/predict_structures.py "${pred_args[@]}"

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
    "model": "distance_regressor",
    "train_size_requested": int("$train_size"),
    "regressor": "$model_name",
    "max_degree": int("$MAX_DEGREE"),
    "use_mmff": bool(int("$USE_MMFF")),
    "metrics": metrics.get("metrics", {}),
    "train_log": train,
    "metrics_path": "$metrics_path",
    "checkpoint": "$ckpt_path",
}
print(json.dumps(out))
PY
    fi
  done
done
