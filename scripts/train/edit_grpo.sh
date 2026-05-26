#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_DIR"

: "${MODEL_NAME_OR_PATH:?Set MODEL_NAME_OR_PATH to a pretrained Qwen-Kontext checkpoint.}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2,3,4,5,6,7}"
NPROC_PER_NODE="${NPROC_PER_NODE:-6}"
MASTER_ADDR="${MASTER_ADDR:-localhost}"
MASTER_PORT="${MASTER_PORT:-25000}"
RUN_NAME="${RUN_NAME:-qwenkontext-edit-grpo}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/rl/kontext/$RUN_NAME}"
DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-scripts/train/deepspeed/zero3.json}"
EDITREWARD_URL="${EDITREWARD_URL:-http://127.0.0.1:18088/}"
PROMPTS_FILE="${PROMPTS_FILE:-https://huggingface.co/wangfuyun/PrompRL/resolve/main/data/omni_edit_train_50k.parquet}"
REPORT_TO="${REPORT_TO:-none}"
if [[ -n "${WANDB_PROJECT:-}" && "$REPORT_TO" == "none" ]]; then
  REPORT_TO="wandb"
fi

TORCHRUN_ARGS=(
  --nproc_per_node="$NPROC_PER_NODE"
  --nnodes="${NNODES:-1}"
  --node_rank="${NODE_RANK:-0}"
  --master_addr="$MASTER_ADDR"
  --master_port="$MASTER_PORT"
)

TRAIN_ARGS=(
  -m unirl.train_edit
  --reward_funcs editreward format
  --deepspeed "$DEEPSPEED_CONFIG"
  --output_dir "$OUTPUT_DIR"
  --model_name_or_path "$MODEL_NAME_OR_PATH"
  --prompts_file "$PROMPTS_FILE"
  --image_column "${IMAGE_COLUMN:-image}"
  --prompt_column "${PROMPT_COLUMN:-prompt}"
  --editreward_url "$EDITREWARD_URL"
  --max_prompt_length "${MAX_PROMPT_LENGTH:-8192}"
  --max_completion_length "${MAX_COMPLETION_LENGTH:-512}"
  --num_generations "${NUM_GENERATIONS:-8}"
  --num_skip_refinement "${NUM_SKIP_REFINEMENT:-2}"
  --num_sde "${NUM_SDE:-4}"
  --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE:-1}"
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS:-1}"
  --logging_steps "${LOGGING_STEPS:-1}"
  --learning_rate "${LEARNING_RATE:-3e-7}"
  --bf16 "${BF16:-true}"
  --report_to "$REPORT_TO"
  --gradient_checkpointing "${GRADIENT_CHECKPOINTING:-true}"
  --attn_implementation "${ATTN_IMPLEMENTATION:-flash_attention_2}"
  --max_pixels "${MAX_PIXELS:-200704}"
  --min_pixels "${MIN_PIXELS:-200704}"
  --image_size "${IMAGE_SIZE:-1024}"
  --save_total_limit "${SAVE_TOTAL_LIMIT:-4}"
  --save_strategy "${SAVE_STRATEGY:-steps}"
  --save_steps "${SAVE_STEPS:-100}"
  --beta "${BETA:-1e-2}"
  --num_train_epochs "${NUM_TRAIN_EPOCHS:-10}"
  --run_name "$RUN_NAME"
)

if [[ -n "${DATASET_CACHE_DIR:-}" ]]; then
  TRAIN_ARGS+=(--dataset_cache_dir "$DATASET_CACHE_DIR")
fi

export PROMPTRL_EDIT_GUIDANCE_SCALE="${PROMPTRL_EDIT_GUIDANCE_SCALE:-${EDIT_GUIDANCE_SCALE:-2.5}}"
export PROMPTRL_EDIT_NUM_INFERENCE_STEPS="${PROMPTRL_EDIT_NUM_INFERENCE_STEPS:-${NUM_INFERENCE_STEPS:-${EDIT_NUM_INFERENCE_STEPS:-8}}}"
export PROMPTRL_EDIT_HEIGHT="${PROMPTRL_EDIT_HEIGHT:-${EDIT_HEIGHT:-1024}}"
export PROMPTRL_EDIT_WIDTH="${PROMPTRL_EDIT_WIDTH:-${EDIT_WIDTH:-1024}}"
export PROMPTRL_EDIT_SDE_NOISE_SCALE="${PROMPTRL_EDIT_SDE_NOISE_SCALE:-${SDE_NOISE_SCALE:-0.8}}"
export DIT_LEARNING_RATE="${DIT_LEARNING_RATE:-2e-7}"
export LLM_LEARNING_RATE="${LLM_LEARNING_RATE:-3e-7}"
# Per-group AdamW betas. DiT sees a non-stationary prompt distribution under joint RL, so
# lowering DIT_BETA1 (e.g. to 0 or 0.5) is worth ablating. Defaults match HF Trainer.
export DIT_BETA1="${DIT_BETA1:-0.9}"
export DIT_BETA2="${DIT_BETA2:-0.999}"
export LLM_BETA1="${LLM_BETA1:-0.9}"
export LLM_BETA2="${LLM_BETA2:-0.999}"

torchrun "${TORCHRUN_ARGS[@]}" "${TRAIN_ARGS[@]}"
