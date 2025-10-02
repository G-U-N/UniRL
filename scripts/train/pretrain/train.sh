export OUTPUT_FOLDER=./outputs/pretrain/joint
export IMG_FOLDER="."

export NCCL_DEBUG=INFO

torchrun --nproc_per_node=8 --master_port=29501 -m \
    unimodel.blip3o.train.train_mem \
    --deepspeed scripts/train/pretrain/deepspeed_scripts/zero1.json \
    --model_name_or_path  Qwen/Qwen2.5-VL-3B-Instruct \
    --version qwen \
    --data_type "mix" \
    --gen_vision_tower Efficient-Large-Model/Sana_1600M_1024px_BF16_diffusers \
    --down_projector_type transformer \
    --bf16 True \
    --output_dir ${OUTPUT_FOLDER} \
    --per_device_train_batch_size 16 \
    --gradient_accumulation_steps 1 \
    --eval_strategy "no" \
    --save_strategy "steps" \
    --save_steps 10000 \
    --save_total_limit 20 \
    --learning_rate 1e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.003 \
    --lr_scheduler_type "cosine_with_min_lr" \
    --lr_scheduler_kwargs '{"min_lr":1e-5}' \
    --model_max_length 1024 \
    --logging_steps 1 \
    --tf32 True \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --n_query 256 \
    --n_und_query 576 \
    --report_to tensorboard \
    --num_train_epochs=10 \
    --run_name blip3o_qwen_vl_3b > output.log 2>&1
