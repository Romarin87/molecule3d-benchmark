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

RUN_ROOT="${RUN_ROOT:-work/compare_egnn_transformer}"
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
NUM_HEADS="${NUM_HEADS:-4}"
DROPOUT="${DROPOUT:-0.0}"
RBF_BINS="${RBF_BINS:-16}"
RBF_CUTOFF="${RBF_CUTOFF:-5.0}"
TRAIN_SIZES_STR="${TRAIN_SIZES_STR:-1000 5000 10000 20000}"
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
    run_tag="egnntr_hd${HIDDEN_DIM}_l${num_layers}_h${NUM_HEADS}_lr${lr_tag}_ep${EPOCHS}_n${train_size}"
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
    python scripts/train_egnn_transformer.py \
      --manifest "$TRAIN_MANIFEST" \
      "${ATOM_ARGS[@]}" \
      "${ELEMENT_ARGS[@]}" \
      --max-degree "$MAX_DEGREE" \
      --max-train "$train_size" \
      --hidden-dim "$HIDDEN_DIM" \
      --num-layers "$num_layers" \
      --num-heads "$NUM_HEADS" \
      --dropout "$DROPOUT" \
      --rbf-bins "$RBF_BINS" \
      --rbf-cutoff "$RBF_CUTOFF" \
      --epochs "$EPOCHS" \
      --lr "$LR" \
      --device "$DEVICE" \
      --seed "$SEED" \
      --output "$ckpt_path" \
      > "$train_log"

    if [[ "$DO_EVAL" == "1" ]]; then
      python scripts/predict_structures.py \
        --method egnn_transformer \
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
    "model": "egnn_transformer",
    "train_size_requested": int("$train_size"),
    "hidden_dim": int("$HIDDEN_DIM"),
    "num_layers": int("$num_layers"),
    "num_heads": int("$NUM_HEADS"),
    "epochs": int("$EPOCHS"),
    "lr": float("$LR"),
    "max_degree": int("$MAX_DEGREE"),
    "dropout": float("$DROPOUT"),
    "rbf_bins": int("$RBF_BINS"),
    "rbf_cutoff": float("$RBF_CUTOFF"),
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
    "model": "egnn_transformer",
    "train_size_requested": int("$train_size"),
    "hidden_dim": int("$HIDDEN_DIM"),
    "num_layers": int("$num_layers"),
    "num_heads": int("$NUM_HEADS"),
    "epochs": int("$EPOCHS"),
    "lr": float("$LR"),
    "max_degree": int("$MAX_DEGREE"),
    "dropout": float("$DROPOUT"),
    "rbf_bins": int("$RBF_BINS"),
    "rbf_cutoff": float("$RBF_CUTOFF"),
    "train_log": train,
    "checkpoint": "$ckpt_path",
}
print(json.dumps(out))
PY
    fi
  done
done
