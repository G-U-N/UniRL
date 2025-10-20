export WANDB_BASE_URL=https://api.wandb.ai
export WANDB_PROJECT=UniRL-Zero
export WANDB_API_KEY="[YOUR WANDB_API_KEY]"
export WANDB_RUN_NAME=Qwen-VL-3B-GRPO-$(date +%Y-%m-%d-%H-%M-%S)
export RUN_NAME=T2ICOT_GENEVAL
wandb login $WANDB_API_KEY

torchrun --nproc_per_node=8 \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=localhost \
    --master_port=25000 \
    -m unirl.grpo_blip3o \
    --reward_funcs gen_eval format \
    --deepspeed scripts/train/rl/deepspeed_scripts/zero3.json \
    --output_dir outputs/rl/blip3o/$RUN_NAME \
    --model_name_or_path outputs/pretrain/blip3o/checkpoint-cot-cold-start \
    --prompts_file assets/rl_datasets/train_metadata.jsonl \
    --max_prompt_length 8192 \
    --num_generations 8 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --logging_steps 1 \
    --learning_rate 1e-5 \
    --bf16 true \
    --report_to wandb \
    --gradient_checkpointing true \
    --attn_implementation flash_attention_2 \
    --max_pixels 802816 \
    --save_total_limit 5 \
    --save_strategy steps \
    --task t2icot \
    --beta 4e-3 \
    --save_steps 50 \
    --num_train_epochs 10 \
    --run_name $RUN_NAME 


