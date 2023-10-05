#!/bin/bash

# change1: use peft-0.3.0.dev0 (previous: peft 0.6.0.dev0)
# change2: use new config

CUDA_VISIBLE_DEVICES=0,1 python finetune_v100.py \
    --base_model '/media/volume1/aicasr/llama-7b-hf' \
    --data_path '/home/t36668/projects/icd-transformers/dataset/alpaca_cmuh/alpaca_cmuh_v1.json' \
    --output_dir './models/lora-alpaca-cmuh-v1c' \
    --num_epochs 1 \
    --lora_target_modules '[q_proj,k_proj,v_proj,o_proj]' \
    --lora_r 16 \
    --cutoff_len 512 \
    --group_by_length \
    ----batch_size 32

# CUDA_VISIBLE_DEVICES=3 python finetune.py \ --base_model='decapoda-research/llama-7b-hf' \ --data_path 'alpaca_data_gpt4.json' \ --num_epochs=5 \ --cutoff_len=512 \ --group_by_length \ --output_dir='./lora-alpaca' \ --lora_target_modules='[q_proj,k_proj,v_proj,o_proj]' \ --lora_r=16 \ --micro_batch_size=8
