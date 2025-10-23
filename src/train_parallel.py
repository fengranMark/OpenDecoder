import torch
import transformers
import os
import sys
import importlib
from dataclasses import dataclass, field
from typing import Optional
from transformers import Trainer
from RAG_dataset import Train_SFT_Dataset, Train_Open_Dataset, Train_MultiOpen_Dataset


# -----------------------------
# 分布式训练初始化函数
# -----------------------------
def setup_distributed():
    if torch.distributed.is_available() and torch.distributed.is_initialized() is False:
        torch.distributed.init_process_group(backend="nccl", timeout=torch.distributed.timedelta(seconds=600))
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
        print(f"[Rank {torch.distributed.get_rank()}] Initialized distributed training with {torch.cuda.device_count()} GPUs")


# -----------------------------
# 模型和配置加载函数
# -----------------------------
def load_imodel_and_iconfig_package(model_pattern, src_path):
    model_path = os.path.join(src_path, "model")
    if not os.path.exists(model_path):
        print(f"path not exist: {model_path}")
        return None, None
    if model_path not in sys.path:
        sys.path.append(model_path)

    try:
        IModelForCausalLM = importlib.import_module(f"{model_pattern}.modeling").IModelForCausalLM
        IConfig = importlib.import_module(f"{model_pattern}.configuration").IConfig
        return IModelForCausalLM, IConfig
    except ModuleNotFoundError as e:
        print(f"Module load fail: {e}")
        return None, None


# -----------------------------
# 参数定义
# -----------------------------
@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="Qwen/Qwen2.5-7B")
    enable_flash_attn: bool = field(default=False)
    is_base: bool = field(default=False)
    model_pattern: Optional[str] = field(default="phoenix")
    src_path: Optional[str] = field(default="phoenix")


@dataclass
class DataArguments:
    NQ_data_path: str = field(default=None)
    NQ_RAG_data_path: str = field(default=None)
    NQ_LLM_score_path: str = field(default=None)
    NQ_QPP_score_path: str = field(default=None)
    hotpotqa_data_path: str = field(default=None)
    hotpotqa_RAG_data_path: str = field(default=None)
    hotpotqa_LLM_score_path: str = field(default=None)
    hotpotqa_QPP_score_path: str = field(default=None)
    RAG_text_path: str = field(default=None)
    val_data_path: str = field(default=None)
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
    model_max_length: int = field(default=4096)
    ddp_find_unused_parameters: bool = field(default=False)  # ✅ 多卡稳定性
    gradient_checkpointing: bool = field(default=True)        # ✅ 减少显存占用


# -----------------------------
# 主训练流程
# -----------------------------
def train():
    # 初始化分布式环境（单卡不会影响）
    setup_distributed()

    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # 模型加载
    if model_args.is_base:
        config = transformers.AutoConfig.from_pretrained(model_args.model_name_or_path)
        model = transformers.AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            config=config,
            torch_dtype=torch.bfloat16,
            cache_dir=training_args.cache_dir,
            trust_remote_code=True,
        )
    else:
        IModelForCausalLM, IConfig = load_imodel_and_iconfig_package(model_args.model_pattern, model_args.src_path)
        config = IConfig.from_pretrained(model_args.model_name_or_path)
        model = IModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            config=config,
            torch_dtype=torch.bfloat16,
            cache_dir=training_args.cache_dir,
        )

    # tokenizer
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
    tokenizer.add_special_tokens({'additional_special_tokens': special_passage_tokens})
    model.resize_token_embeddings(len(tokenizer))

    # dataset
    if training_args.train_mode == "sft":
        train_dataset = Train_SFT_Dataset(tokenizer, data_args)
    elif training_args.train_mode == "open":
        if data_args.add_LLM_scores or data_args.add_QPP_scores:
            train_dataset = Train_MultiOpen_Dataset(tokenizer, data_args)
        else:
            train_dataset = Train_Open_Dataset(tokenizer, data_args)
    else:
        raise ValueError

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
    )

    trainer.train()
    trainer.save_state()
    torch.cuda.synchronize()
    trainer.save_model(output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()
