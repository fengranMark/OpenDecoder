#!/bin/bash

MODEL_PATH=ckpts/Qwen2.5-3B-Instruct_nq_hotpotqa_open_top10_irrel/checkpoint-20000
NQ_DATA_PATH=./datasets/nq/test.jsonl
NQ_RAG_DATA_PATH=./datasets/nq/RAG_test_input.jsonl
HOTPOTQA_DATA_PATH=./datasets/hotpotqa/hotpot_dev.jsonl
HOTPOTQA_RAG_DATA_PATH=./datasets/hotpotqa/RAG_dev_input.jsonl
POPQA_DATA_PATH=./datasets/popqa/popqa_longtail.jsonl
POPQA_RAG_DATA_PATH=./datasets/popqa/RAG_dev_input.jsonl
TRIQA_DATA_PATH=./datasets/trivialqa/trivialqa_test.jsonl
TRIQA_RAG_DATA_PATH=./datasets/trivialqa/RAG_test_input.jsonl
TWIKI_DATA_PATH=./datasets/2wiki/2wiki_dev.jsonl
TWIKI_RAG_DATA_PATH=./datasets/2wiki/RAG_dev_input.jsonl
RAG_TEXT_PATH=./datasets/wikipedia/pid2psg.pkl
OUTPUT_DATA_PATH=./Qwen2.5-3B-Instruct_nq_hotpotqa_open_top10_irrel/all
RESULT_PATH=./Qwen2.5-3B-Instruct_nq_hotpotqa_open_top10_irrel/all/result.txt
MODEL_PATTERN=qwen_decoder # the path to the model in src/model
src_path=./src
log_name=$(date +"%m-%d_%H-%M").log

python ./inference.py \
    --model_name_or_path $MODEL_PATH \
    --NQ_data_path $NQ_DATA_PATH \
    --NQ_RAG_data_path $NQ_RAG_DATA_PATH \
    --hotpotqa_data_path $HOTPOTQA_DATA_PATH \
    --hotpotqa_RAG_data_path $HOTPOTQA_RAG_DATA_PATH \
    --popqa_data_path $POPQA_DATA_PATH \
    --popqa_RAG_data_path $POPQA_RAG_DATA_PATH \
    --trivialqa_data_path $TRIQA_DATA_PATH \
    --trivialqa_RAG_data_path $TRIQA_RAG_DATA_PATH \
    --twiki_data_path $TWIKI_DATA_PATH \
    --twiki_RAG_data_path $TWIKI_RAG_DATA_PATH \
    --RAG_text_path $RAG_TEXT_PATH \
    --output_data_path $OUTPUT_DATA_PATH \
    --result_path $RESULT_PATH \
    --model_pattern $MODEL_PATTERN \
    --train_mode "open" \ # with standard sft of opendecoder
    --add_irrelevant_psg False \
    --full_irrelevant_psg True \
    --top_k 10 \
    --normalization_type normal \
    --shuffle_RAG False \
    --src_path $src_path \
    --bf16 True \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --model_max_length 4096 \
    --lazy_loading True \
