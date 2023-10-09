#!/bin/bash

# change1: use peft-0.3.0.dev0 (previous: peft 0.6.0.dev0)
# change2: use new config

CUDA_VISIBLE_DEVICES=0,1 python finetune_v100.py \
    --base_model '/media/volume1/aicasr/llama-7b-hf' \
    --data_path '/home/t36668/projects/public/alpaca-cleaned/tw_v2/alpaca_data_cleaned_tw2.json' \
    --output_dir './models/lora-alpaca-tw-v2' \
    --num_epochs 2 \
    --lora_target_modules '[q_proj,k_proj,v_proj,o_proj]' \
    --lora_r 16 \
    --cutoff_len 512 \
    ----batch_size 32

