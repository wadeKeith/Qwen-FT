#!/bin/bash
if [ -d "wandb" ]; then
    rm -rf wandb
    echo "Wandb dir clean"
else
    echo "Wandb dir doesn't exists"
fi

# You can use 2B instead of 7B
# MODEL_NAME="Qwen/Qwen2-VL-7B-Instruct"
# MODEL_NAME="Qwen/Qwen2-VL-2B-Instruct"
# MODEL_NAME="Qwen/Qwen2.5-VL-3B-Instruct"
MODEL_NAME="Qwen/Qwen2.5-VL-7B-Instruct"

GLOBAL_BATCH_SIZE=64
BATCH_PER_DEVICE=4
NUM_DEVICES=4
GRAD_ACCUM_STEPS=$((GLOBAL_BATCH_SIZE / (BATCH_PER_DEVICE * NUM_DEVICES)))

export OUTPUT_DIR="./checkpoints/3b/libero"
export CFLAGS="-I/usr/include"
export LDFLAGS="-L/usr/lib/x86_64-linux-gnu"
export CUTLASS_PATH="/home/yin/cutlass/"
export PYTHONPATH=src:$PYTHONPATH


if [ ! -d "$OUTPUT_DIR" ]; then
    mkdir "$OUTPUT_DIR"
    echo "Folder '$OUTPUT_DIR' created"
else
    echo "Folder '$OUTPUT_DIR' already exists"
fi


export WANDB_PROJECT="qwen-2.5-vl-libero"
export WANDB_MODE="online"
export WANDB_NAME="3b-fft-250323"
export CUDA_VISIBLE_DEVICES=0,1,2,3

deepspeed src/training/train.py \
    --use_liger True \
    --deepspeed scripts/zero2_offload.json \
    --model_id $MODEL_NAME \
    --data_path data/qwen_data.json \
    --image_folder ./ \
    --remove_unused_columns False \
    --freeze_vision_tower False \
    --freeze_llm False \
    --tune_merger True \
    --lora_enable False \
    --bf16 True \
    --fp16 False \
    --disable_flash_attn2 False \
    --output_dir $OUTPUT_DIR \
    --num_train_epochs 10 \
    --per_device_train_batch_size $BATCH_PER_DEVICE \
    --gradient_accumulation_steps $GRAD_ACCUM_STEPS \
    --image_min_pixels $((512 * 28 * 28)) \
    --image_max_pixels $((1280 * 28 * 28)) \
    --learning_rate 1e-5 \
    --merger_lr 1e-5 \
    --vision_lr 2e-6 \
    --weight_decay 0.1 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --gradient_checkpointing True \
    --report_to wandb \
    --run_name $WANDB_NAME \
    --lazy_preprocess True \
    --save_strategy "epoch" \
    --save_total_limit 10 \
    --dataloader_num_workers 32


#    --save_strategy "steps"
#    --save_steps 1 \