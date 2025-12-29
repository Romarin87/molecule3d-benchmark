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

RUN_ROOT="${RUN_ROOT:-work/compare_etkdg}"
VAL_MANIFEST="${VAL_MANIFEST:-data/processed/work/test_chon20_manifest.json}"
ATOM_COUNT="${ATOM_COUNT:-20}"
DEVICE="${DEVICE:-cuda}"
SEED="${SEED:-0}"
MAX_EVAL="${MAX_EVAL:-0}"
PRED_SAMPLES="${PRED_SAMPLES:-$MAX_EVAL}"
NUM_CONFS_STR="${NUM_CONFS_STR:-10}"
USE_MMFF="${USE_MMFF:-0}"
MAX_ITERS="${MAX_ITERS:-200}"
DO_EVAL="${DO_EVAL:-1}"
DO_PLOT="${DO_PLOT:-0}"
RESUME="${RESUME:-0}"
OVERWRITE="${OVERWRITE:-0}"

read -r -a NUM_CONFS_LIST <<< "$NUM_CONFS_STR"

mkdir -p "$ROOT/.mplconfig" "$ROOT/.cache"
mkdir -p "$RUN_ROOT"

LOG_PATH="$RUN_ROOT/results.jsonl"
if [[ "$OVERWRITE" == "1" ]]; then
  : > "$LOG_PATH"
fi

if [[ "$DO_EVAL" == "1" && ! -f "$VAL_MANIFEST" ]]; then
  echo "Val manifest not found: $VAL_MANIFEST" >&2
  exit 1
fi

ATOM_ARGS=()
if [[ -n "${ATOM_COUNT:-}" ]]; then
  ATOM_ARGS+=(--atom-count "$ATOM_COUNT")
fi

mmff_tag="mmff${USE_MMFF}"

for num_confs in "${NUM_CONFS_LIST[@]}"; do
  run_tag="etkdg_c${num_confs}_${mmff_tag}_seed${SEED}_it${MAX_ITERS}"
  run_dir="$RUN_ROOT/${run_tag}"
  pred_path="$run_dir/${run_tag}.npz"
  pred_log="$run_dir/${run_tag}_predict.json"
  metrics_path="$run_dir/${run_tag}_metrics.json"
  per_sample_path="$run_dir/${run_tag}_per_sample.jsonl"

  if [[ "$RESUME" == "1" && -f "$metrics_path" ]]; then
    echo "[skip] $run_tag"
    continue
  fi

  mkdir -p "$run_dir"
  echo "[predict] $run_tag"
  pred_args=(
    --method etkdg
    --manifest "$VAL_MANIFEST"
    --num-confs "$num_confs"
    --seed "$SEED"
    --max-iters "$MAX_ITERS"
    --output "$pred_path"
  )
  if [[ "$USE_MMFF" == "1" ]]; then
    pred_args+=(--use-mmff)
  fi
  python scripts/predict_structures.py "${pred_args[@]}" "${ATOM_ARGS[@]}" > "$pred_log"

  if [[ "$DO_EVAL" == "1" ]]; then
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
metrics = json.load(open("$metrics_path"))
out = {
    "run_tag": "$run_tag",
    "model": "etkdg",
    "num_confs": int("$num_confs"),
    "use_mmff": bool(int("$USE_MMFF")),
    "seed": int("$SEED"),
    "max_iters": int("$MAX_ITERS"),
    "metrics": metrics.get("metrics", {}),
    "predict_log": "$pred_log",
    "metrics_path": "$metrics_path",
}
print(json.dumps(out))
PY
  fi
done
