#!/bin/bash

experiment_name=Qwen2.5-7B-Instruct-OpenDecoder
model_path=./checkpoints/Qwen2.5-7B-Instruct
config_path=./src/model/qwen_decoder/final_config.json
model_pattern=qwen_decoder
savedir_imodel=./checkpoints/${experiment_name}
src_path=./src
savepath_basemodel_namemodules=./checkpoints/${experiment_name}/basemodel.txt
savepath_imodel_namemodules=./checkpoints/${experiment_name}/imodel.txt

log_folder="../logs/${experiment_name}"
mkdir -p $log_folder
log_name=$(date +"%m-%d_%H-%M").log

python utils/iniModel.py \
    --savepath_basemodel_namemodules ${savepath_basemodel_namemodules} \
    --savepath_imodel_namemodules ${savepath_imodel_namemodules} \
    --model_name_or_path ${model_path} \
    --config_path ${config_path} \
    --model_pattern ${model_pattern} \
    --src_path ${src_path} \
    --savedir_imodel ${savedir_imodel} 
