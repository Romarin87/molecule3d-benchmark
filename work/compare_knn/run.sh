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

RUN_ROOT="${RUN_ROOT:-work/compare_knn}"
TRAIN_MANIFEST="${TRAIN_MANIFEST:-data/processed/work/train_chon20_manifest.json}"
VAL_MANIFEST="${VAL_MANIFEST:-data/processed/work/test_chon20_manifest.json}"
ATOM_COUNT="${ATOM_COUNT:-20}"
K_LIST_STR="${K_LIST_STR:-1}"
FP_RADIUS="${FP_RADIUS:-2}"
FP_BITS="${FP_BITS:-2048}"
USE_CHIRALITY="${USE_CHIRALITY:-0}"
SEED="${SEED:-0}"
MAX_EVAL="${MAX_EVAL:-0}"
PRED_SAMPLES="${PRED_SAMPLES:-$MAX_EVAL}"
TRAIN_SIZES_STR="${TRAIN_SIZES_STR:-1000 5000 10000 20000}"
DO_EVAL="${DO_EVAL:-1}"
DO_PLOT="${DO_PLOT:-0}"
RESUME="${RESUME:-0}"
OVERWRITE="${OVERWRITE:-0}"

read -r -a K_LIST <<< "$K_LIST_STR"
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

ATOM_ARGS=()
if [[ -n "${ATOM_COUNT:-}" ]]; then
  ATOM_ARGS+=(--atom-count "$ATOM_COUNT")
fi

chirality_tag="chir${USE_CHIRALITY}"

for k in "${K_LIST[@]}"; do
  for train_size in "${TRAIN_SIZES[@]}"; do
    run_tag="knn_k${k}_r${FP_RADIUS}_b${FP_BITS}_${chirality_tag}_n${train_size}"
    run_dir="$RUN_ROOT/${run_tag}"
    ckpt_path="$run_dir/${run_tag}.pkl"
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
    train_args=(
      --manifest "$TRAIN_MANIFEST"
      --max-train "$train_size"
      --k "$k"
      --fp-radius "$FP_RADIUS"
      --fp-bits "$FP_BITS"
      --output "$ckpt_path"
    )
    if [[ "$USE_CHIRALITY" == "1" ]]; then
      train_args+=(--use-chirality)
    fi
    python scripts/train_knn.py "${train_args[@]}" "${ATOM_ARGS[@]}" > "$train_log"

    if [[ "$DO_EVAL" == "1" ]]; then
      python scripts/predict_structures.py \
        --method knn \
        --checkpoint "$ckpt_path" \
        --manifest "$VAL_MANIFEST" \
        --max-samples "$PRED_SAMPLES" \
        --output "$pred_path"

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
    "model": "knn",
    "train_size_requested": int("$train_size"),
    "k": int("$k"),
    "fp_radius": int("$FP_RADIUS"),
    "fp_bits": int("$FP_BITS"),
    "use_chirality": bool(int("$USE_CHIRALITY")),
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
