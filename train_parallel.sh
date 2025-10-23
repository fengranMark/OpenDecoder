#!/bin/bash

train_mode=open
top_k=10
experiment_name=Qwen2.5-7B-Instruct-OpenDecoder_nq_hotpotqa_${train_mode}_top${top_k}
WANDB_PROJECT=$experiment_name  # project name in wandb
MODEL_PATH=./checkpoints/Qwen2.5-7B-Instruct-OpenDecoder
NQ_DATA_PATH=./datasets/nq/train.jsonl
NQ_RAG_DATA_PATH=./datasets/nq/RAG_train_input_robust.jsonl
NQ_LLM_SCORE_PATH=./datasets/nq/train_LLM_ranking_score.jsonl
NQ_QPP_SCORE_PATH=./datasets/nq/train_QPP_score.jsonl
HOTPOTQA_DATA_PATH=./datasets/hotpotqa/train.jsonl
HOTPOTQA_RAG_DATA_PATH=./datasets/hotpotqa/RAG_train_input_robust.jsonl
HOTPOTQA_LLM_SCORE_PATH=./datasets/hotpotqa/train_LLM_ranking_score.jsonl
HOTPOTQA_QPP_SCORE_PATH=./datasets/hotpotqa/train_QPP_score.jsonl
RAG_TEXT_PATH=./datasets/wikipedia/pid2psg.pkl
MODEL_PATTERN=qwen_decoder # the path to the model in src/model
src_path=./src
log_folder="./logs/${experiment_name}"
mkdir -p $log_folder
log_name=$(date +"%m-%d_%H-%M").log

CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 --master_port=29501 ./src/train.py \
    --model_name_or_path $MODEL_PATH \
    --NQ_data_path $NQ_DATA_PATH \
    --NQ_RAG_data_path $NQ_RAG_DATA_PATH \
    --NQ_LLM_score_path $NQ_LLM_SCORE_PATH \
    --NQ_QPP_score_path $NQ_QPP_SCORE_PATH \
    --hotpotqa_data_path $HOTPOTQA_DATA_PATH \
    --hotpotqa_RAG_data_path $HOTPOTQA_RAG_DATA_PATH \
    --hotpotqa_LLM_score_path $HOTPOTQA_LLM_SCORE_PATH \
    --hotpotqa_QPP_score_path $HOTPOTQA_QPP_SCORE_PATH \
    --RAG_text_path $RAG_TEXT_PATH \
    --model_pattern $MODEL_PATTERN \
    --train_mode $train_mode \
    --add_irrelevant_psg True \
    --add_LLM_scores False \
    --add_QPP_scores False \
    --top_k $top_k \
    --normalization_type normal \
    --shuffle_RAG False \
    --src_path $src_path \
    --bf16 True \
    --output_dir ./outputs/${experiment_name} \
    --run_name ${experiment_name} \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_safetensors False \
    --gradient_checkpointing True \
    --save_total_limit 1 \
    --learning_rate 1e-5 \
    --weight_decay 0.1 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 2 \
    --model_max_length 4096 \
    --lazy_loading True \
    
