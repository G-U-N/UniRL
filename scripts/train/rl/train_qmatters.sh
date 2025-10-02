export WANDB_BASE_URL=https://api.wandb.ai
export WANDB_PROJECT=UniRL-Zero
export WANDB_API_KEY="5fad14fb83d2bb193c07ab026d2d714a4cdffafd"
export WANDB_RUN_NAME=Qwen-VL-3B-GRPO-$(date +%Y-%m-%d-%H-%M-%S)
export RUN_NAME=Kontext_I2I
wandb login $WANDB_API_KEY

torchrun --nproc_per_node=8 \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=localhost \
    --master_port=25000 \
    -m unirl.grpo_pmatters \
    --reward_funcs sim_direction clip_sim \
    --deepspeed scripts/train/rl/deepspeed_scripts/zero3.json \
    --output_dir outputs/rl/kontext/$RUN_NAME \
    --model_name_or_path /home/ubuntu/open-r1-multimodal/qwenkontext/qwenkontext-test \
    --prompts_file assets/rl_datasets/i2i_dataset.parquet \
    --max_prompt_length 8192 \
    --num_generations 8 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --logging_steps 1 \
    --learning_rate 1e-5 \
    --bf16 true \
    --task i2i-text \
    --report_to wandb \
    --gradient_checkpointing true \
    --attn_implementation flash_attention_2 \
    --max_pixels 802816 \
    --save_total_limit 4 \
    --save_strategy steps \
    --beta 3e-4 \
    --save_steps 100 \
    --num_train_epochs 200 \
    --run_name $RUN_NAME


export WANDB_BASE_URL=https://api.wandb.ai
export WANDB_PROJECT=UniRL-Zero
export WANDB_API_KEY="5fad14fb83d2bb193c07ab026d2d714a4cdffafd"
export WANDB_RUN_NAME=Qwen-VL-3B-GRPO-$(date +%Y-%m-%d-%H-%M-%S)
export RUN_NAME=FLUX
wandb login $WANDB_API_KEY

torchrun --nproc_per_node=8 \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=localhost \
    --master_port=25000 \
    -m unirl.grpo_pmatters \
    --reward_funcs gen_eval format \
    --deepspeed scripts/train/rl/deepspeed_scripts/zero3.json \
    --output_dir outputs/rl/flux/$RUN_NAME \
    --model_name_or_path /home/ubuntu/open-r1-multimodal/qwenflux/qwenflux-test \
    --prompts_file assets/rl_datasets/train_metadata.jsonl \
    --max_prompt_length 8192 \
    --num_generations 8 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --logging_steps 1 \
    --learning_rate 1e-5 \
    --bf16 true \
    --task t2i-text \
    --report_to wandb \
    --gradient_checkpointing true \
    --attn_implementation flash_attention_2 \
    --max_pixels 802816 \
    --save_total_limit 16 \
    --save_strategy steps \
    --beta 3e-4 \
    --save_steps 500 \
    --num_train_epochs 200 \
    --run_name $RUN_NAME


export WANDB_BASE_URL=https://api.wandb.ai
export WANDB_PROJECT=UniRL-Zero
export WANDB_API_KEY="5fad14fb83d2bb193c07ab026d2d714a4cdffafd"
export WANDB_RUN_NAME=Qwen-VL-3B-GRPO-$(date +%Y-%m-%d-%H-%M-%S)
export RUN_NAME=FLUX
wandb login $WANDB_API_KEY

torchrun --nproc_per_node=8 \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=localhost \
    --master_port=25000 \
    -m unirl.grpo_pmatters \
    --reward_funcs gen_eval format \
    --deepspeed scripts/train/rl/deepspeed_scripts/zero3.json \
    --output_dir outputs/rl/flux/$RUN_NAME \
    --model_name_or_path /home/ubuntu/open-r1-multimodal/qwenflux/qwenflux-test \
    --prompts_file assets/rl_datasets/train_metadata.jsonl \
    --max_prompt_length 8192 \
    --num_generations 8 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --logging_steps 1 \
    --learning_rate 1e-5 \
    --bf16 true \
    --task t2i-diff \
    --report_to wandb \
    --gradient_checkpointing true \
    --attn_implementation flash_attention_2 \
    --max_pixels 802816 \
    --save_total_limit 16 \
    --save_strategy steps \
    --beta 3e-4 \
    --save_steps 500 \
    --num_train_epochs 10 \
    --run_name $RUN_NAME