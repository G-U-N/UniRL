#!/usr/bin/env bash
set -euo pipefail

# Launch GenEval prompt-refinement GRPO training with the FLUX.1-dev T2I
# diffusion expert. The trainer auto-detects the backbone from MODEL_NAME_OR_PATH
# (must contain the substring "qwenflux").
#
# Defaults: prompt-only (LLM updates, DiT frozen). Set JOINT=true to update the
# FLUX transformer jointly with SDE log-prob tracking.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_DIR"

: "${MODEL_NAME_OR_PATH:?Set MODEL_NAME_OR_PATH to a pretrained Qwen-Flux checkpoint (the path must contain 'qwenflux').}"

JOINT="${JOINT:-false}"
TRAINER_TAG="prompt"
if [[ "$JOINT" == "true" ]]; then
  TRAINER_TAG="joint"
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2,3,4,5,6,7}"
NPROC_PER_NODE="${NPROC_PER_NODE:-6}"
MASTER_ADDR="${MASTER_ADDR:-localhost}"
MASTER_PORT="${MASTER_PORT:-25002}"
RUN_NAME="${RUN_NAME:-qwenflux-geneval-${TRAINER_TAG}-grpo}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/rl/flux/$RUN_NAME}"
DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-scripts/train/deepspeed/zero3.json}"
GENEVAL_URL="${GENEVAL_URL:-http://127.0.0.1:18085/}"
PROMPTS_FILE="${PROMPTS_FILE:-assets/rl_datasets/train_metadata.jsonl}"
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
  -m unirl.train_geneval
  --reward_funcs gen_eval format
  --deepspeed "$DEEPSPEED_CONFIG"
  --output_dir "$OUTPUT_DIR"
  --model_name_or_path "$MODEL_NAME_OR_PATH"
  --prompts_file "$PROMPTS_FILE"
  --prompt_column "${PROMPT_COLUMN:-prompt}"
  --geneval_url "$GENEVAL_URL"
  --max_prompt_length "${MAX_PROMPT_LENGTH:-8192}"
  --max_completion_length "${MAX_COMPLETION_LENGTH:-512}"
  --num_generations "${NUM_GENERATIONS:-8}"
  --num_sde "${NUM_SDE:-6}"
  --num_inference_steps "${NUM_INFERENCE_STEPS:-12}"
  --guidance_scale "${GUIDANCE_SCALE:-3.5}"
  --image_height "${IMAGE_HEIGHT:-1024}"
  --image_width "${IMAGE_WIDTH:-1024}"
  --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE:-1}"
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS:-1}"
  --logging_steps "${LOGGING_STEPS:-1}"
  --learning_rate "${LEARNING_RATE:-1e-6}"
  --bf16 "${BF16:-true}"
  --report_to "$REPORT_TO"
  --gradient_checkpointing "${GRADIENT_CHECKPOINTING:-true}"
  --attn_implementation "${ATTN_IMPLEMENTATION:-flash_attention_2}"
  --save_total_limit "${SAVE_TOTAL_LIMIT:-4}"
  --save_strategy "${SAVE_STRATEGY:-steps}"
  --save_steps "${SAVE_STEPS:-100}"
  --beta "${BETA:-1e-2}"
  --num_train_epochs "${NUM_TRAIN_EPOCHS:-5}"
  --run_name "$RUN_NAME"
)

if [[ "$JOINT" == "true" ]]; then
  TRAIN_ARGS+=(--joint True)
fi

if [[ -n "${GENEVAL_ONLY_STRICT:-}" ]]; then
  TRAIN_ARGS+=(--geneval_only_strict "$GENEVAL_ONLY_STRICT")
fi

# FLUX.1-dev T2I defaults: 20 inference steps, first 10 with SDE log-prob tracking.
export PROMPTRL_GENEVAL_GUIDANCE_SCALE="${PROMPTRL_GENEVAL_GUIDANCE_SCALE:-${GUIDANCE_SCALE:-3.5}}"
export PROMPTRL_GENEVAL_NUM_INFERENCE_STEPS="${PROMPTRL_GENEVAL_NUM_INFERENCE_STEPS:-${NUM_INFERENCE_STEPS:-20}}"
export PROMPTRL_GENEVAL_HEIGHT="${PROMPTRL_GENEVAL_HEIGHT:-${IMAGE_HEIGHT:-1024}}"
export PROMPTRL_GENEVAL_WIDTH="${PROMPTRL_GENEVAL_WIDTH:-${IMAGE_WIDTH:-1024}}"
export PROMPTRL_GENEVAL_SDE_NOISE_SCALE="${PROMPTRL_GENEVAL_SDE_NOISE_SCALE:-${SDE_NOISE_SCALE:-0.8}}"
export DIT_LEARNING_RATE="${DIT_LEARNING_RATE:-2e-7}"
export LLM_LEARNING_RATE="${LLM_LEARNING_RATE:-1e-6}"
export DIT_BETA1="${DIT_BETA1:-0.9}"
export DIT_BETA2="${DIT_BETA2:-0.999}"
export LLM_BETA1="${LLM_BETA1:-0.9}"
export LLM_BETA2="${LLM_BETA2:-0.999}"

torchrun "${TORCHRUN_ARGS[@]}" "${TRAIN_ARGS[@]}"
