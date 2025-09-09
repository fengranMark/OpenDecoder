
from dataclasses import dataclass, field
#from torch.utils.data import Dataset, DataLoader
import pathlib
from typing import Optional
import torch
import transformers
import os
import sys
import copy
#import wandb
#wandb.init(mode="disabled")

from transformers import Trainer
from dataset.fs_dataset import (
    make_supervised_data_module,
    make_supervised_data_module_wiki,
)
from eval.eval_metric import compute_metrics
from dataset.RAG_dataset import Train_SFT_Dataset, Train_Open_Dataset, Train_RelDPO_Dataset, Train_MultiOpen_Dataset
from dataset.convert_dataset_rl import convert_torch_dataset_to_hf
import importlib

def load_imodel_and_iconfig_package(model_pattern, src_path):
    model_path = os.path.join(src_path, "model")

    if not os.path.exists(model_path):
        print(f"path not exist: {model_path}")
        return None, None

    if model_path not in sys.path:
        sys.path.append(model_path)

    try:
        IModelForCausalLM = importlib.import_module(
            f"{model_pattern}.modeling"
        ).IModelForCausalLM
        IConfig = importlib.import_module(f"{model_pattern}.configuration").IConfig
        return IModelForCausalLM, IConfig
    except ModuleNotFoundError as e:
        print(f"Module load fail: {e}")
        return None, None


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="Qwen/Qwen2.5-7B")
    enable_flash_attn: bool = field(default=False)
    is_base: bool = field(default=False)
    model_pattern: Optional[str] = field(default="phoenix")
    src_path: Optional[str] = field(default="phoenix")
    num_equal_loop_layers: Optional[int] = field(default=None)
    # loop_pattern: Optional[list] = field(default=None)


@dataclass
class DataArguments:
    NQ_data_path: str = field(
        default=None, metadata={"help": "Path to the training data."}
    )
    NQ_RAG_data_path: str = field(
        default=None, metadata={"help": "Path to the training data."}
    )
    NQ_LLM_score_path: str = field(
        default=None, metadata={"help": "Path to the training data."}
    )
    NQ_QPP_score_path: str = field(
        default=None, metadata={"help": "Path to the training data."}
    )
    hotpotqa_data_path: str = field(
        default=None, metadata={"help": "Path to the training data."}
    )
    hotpotqa_RAG_data_path: str = field(
        default=None, metadata={"help": "Path to the training data."}
    )
    hotpotqa_LLM_score_path: str = field(
        default=None, metadata={"help": "Path to the training data."}
    )
    hotpotqa_QPP_score_path: str = field(
        default=None, metadata={"help": "Path to the training data."}
    )
    RAG_text_path: str = field(
        default=None, metadata={"help": "Path to the training data."}
    )
    val_data_path: str = field(
        default=None, metadata={"help": "Path to the validation data."}
    )
    top_k: int = field(default=5)
    add_irrelevant_psg: bool = field(default=False)
    add_LLM_scores: bool = field(default=False)
    add_QPP_scores: bool = field(default=False)
    normalization_type: str = field(default="normal")
    shuffle_RAG: bool = field(default=False)
    lazy_loading: bool = False
    system_prompt: str = field(default=None)


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    predict_with_generate: bool = True
    train_mode: str = field(default="sft")
    model_max_length: int = field(
        default=4096,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    checkpoint = None


def train():
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if model_args.is_base:
        config = transformers.AutoConfig.from_pretrained(model_args.model_name_or_path)
    else:
        IModelForCausalLM, IConfig = load_imodel_and_iconfig_package(
            model_args.model_pattern, model_args.src_path
        )
        config = IConfig.from_pretrained(model_args.model_name_or_path)
    enable_flash_attn = False
    if (
        model_args.enable_flash_attn
        and getattr(config, "_attn_implementation", None) is not None
    ):
        config._attn_implementation = "flash_attention_2"
        enable_flash_attn = True

    if model_args.is_base:
        model = transformers.AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            config=config,
            torch_dtype=torch.bfloat16 if enable_flash_attn else "auto",
            cache_dir=training_args.cache_dir,
            trust_remote_code=True,
        )
    else:
        model = IModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            config=config,
            torch_dtype=torch.bfloat16 if enable_flash_attn else "auto",
            cache_dir=training_args.cache_dir,
        )

    

    # check the trainable parameters
    print("\nTrainable parameters:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name)
    #breakpoint()

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="left",
        use_fast=False,
        trust_remote_code=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.unk_token
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    special_passage_tokens = [f"Passage_{i+1}:" for i in range(20)]
    #special_passage_tokens.append("<|answer>")  # assume max 20 retrieved passages
    tokenizer.add_special_tokens({'additional_special_tokens': special_passage_tokens})
    model.resize_token_embeddings(len(tokenizer))  # Resize embeddings to accommodate new tokens
    if training_args.train_mode == "sft":
        train_dataset = Train_SFT_Dataset(tokenizer, data_args)
    elif training_args.train_mode == "open":
        if data_args.add_LLM_scores or data_args.add_QPP_scores:
            train_dataset = Train_MultiOpen_Dataset(tokenizer, data_args)
        else:
            train_dataset = Train_Open_Dataset(tokenizer, data_args)
    else:
        raise ValueError
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        #eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        #compute_metrics=compute_metrics,
    )
   
    

    trainer.train()

    trainer.save_state()
    torch.cuda.synchronize()
    trainer.save_model(output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()
