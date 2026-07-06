#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

LOG_DIR="${LOG_DIR:-logs}"
PYTHON_BIN="${PYTHON_BIN:-python}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"

CONFIGS=(
  "configs/direct_prompting/qwen_2_5_0_5b_instruct.yaml"
  "configs/direct_prompting/qwen_2_5_3b_instruct.yaml"
  "configs/direct_prompting/qwen_2_5_7b_instruct.yaml"
  "configs/direct_prompting/llama_3_1_8b_instruct.yaml"
  "configs/direct_prompting/mistral_7b_instruct_v0_3.yaml"
  "configs/direct_prompting/gemma_2_9b_it.yaml"
  "configs/direct_prompting/gemma_3_4b_it.yaml"
  "configs/direct_prompting/seallm_7b_chat.yaml"
  "configs/direct_prompting/vistral_7b_chat.yaml"
)

mkdir -p "$LOG_DIR"

echo "Repo: $ROOT_DIR"
echo "Log dir: $LOG_DIR"

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "Warning: HF_TOKEN is not set. Gated/private Hugging Face models may fail."
fi

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi | tee "$LOG_DIR/hf_probe_gpu_$TIMESTAMP.log"
else
  echo "nvidia-smi not found; continuing without GPU status."
fi

for config in "${CONFIGS[@]}"; do
  name="$(basename "$config" .yaml)"
  log_file="$LOG_DIR/${name}_$TIMESTAMP.log"

  echo
  echo "============================================================"
  echo "Running $config"
  echo "Log: $log_file"
  echo "============================================================"

  "$PYTHON_BIN" -u -m src.probe_models \
    --config "$config" \
    "$@" 2>&1 | tee "$log_file"
done

echo
echo "All configured Hugging Face probes finished."
