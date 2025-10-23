from dataclasses import dataclass, field
from torch.utils.data import Dataset, DataLoader
import pathlib
from typing import Optional
import torch
import transformers
import os
import sys
import tqdm
import argparse
import json
from RAG_dataset import Eval_SFT_Dataset, Eval_Open_Dataset, Eval_MultiOpen_Dataset
import importlib
os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6"


def load_imodel_and_iconfig_package(model_pattern, src_path):
    model_path = os.path.join(src_path, "model")

    if not os.path.exists(model_path):
        print(f"Incorrect Path: {model_path}")
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
        print(f"Module Error: {e}")
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
    popqa_data_path: str = field(
        default=None, metadata={"help": "Path to the training data."}
    )
    popqa_RAG_data_path: str = field(
        default=None, metadata={"help": "Path to the training data."}
    )
    popqa_LLM_score_path: str = field(
        default=None, metadata={"help": "Path to the training data."}
    )
    popqa_QPP_score_path: str = field(
        default=None, metadata={"help": "Path to the training data."}
    )
    trivialqa_data_path: str = field(
        default=None, metadata={"help": "Path to the training data."}
    )
    trivialqa_RAG_data_path: str = field(
        default=None, metadata={"help": "Path to the training data."}
    )
    trivialqa_LLM_score_path: str = field(
        default=None, metadata={"help": "Path to the training data."}
    )
    trivialqa_QPP_score_path: str = field(
        default=None, metadata={"help": "Path to the training data."}
    )
    twiki_data_path: str = field(
        default=None, metadata={"help": "Path to the training data."}
    )
    twiki_RAG_data_path: str = field(
        default=None, metadata={"help": "Path to the training data."}
    )
    twiki_LLM_score_path: str = field(
        default=None, metadata={"help": "Path to the training data."}
    )
    twiki_QPP_score_path: str = field(
        default=None, metadata={"help": "Path to the training data."}
    )
    RAG_text_path: str = field(
        default=None, metadata={"help": "Path to the training data."}
    )
    val_data_path: str = field(
        default=None, metadata={"help": "Path to the validation data."}
    )
    output_data_path: str = field(
        default=None, metadata={"help": "Path to the validation data."}
    )
    result_path: str = field(
        default=None, metadata={"help": "Path to the validation data."}
    )
    top_k: int = field(default=5)
    add_irrelevant_psg: bool = field(default=False)
    add_LLM_scores: bool = field(default=False)
    add_QPP_scores: bool = field(default=False)
    full_irrelevant_psg: bool = field(default=False)
    normalization_type: str = field(default="normal")
    shuffle_RAG: bool = field(default=False)
    lazy_loading: bool = False
    system_prompt: str = field(default=None)


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    predict_with_generate: bool = True
    mode: str = field(default="open")
    model_max_length: int = field(
        default=4096,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    checkpoint = None


def inference():
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
    device = torch.device("cuda:6" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
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
    model.resize_token_embeddings(len(tokenizer))  # Resize embeddings to accommodate new tokens
    if training_args.mode == "sft":
        eval_dataset = Eval_SFT_Dataset(tokenizer, data_args)
    elif training_args.mode == "open":
        if data_args.add_LLM_scores or data_args.add_QPP_scores:
            eval_dataset = Eval_MultiOpen_Dataset(tokenizer, data_args)
        else:
            eval_dataset = Eval_Open_Dataset(tokenizer, data_args)
    else:
        raise ValueError
    eval_loader = DataLoader(eval_dataset, batch_size=1, shuffle=False)

    predictions, golds =  [], []
    f1, acc, em = [], [], []
    cur = 0
    os.makedirs(data_args.output_data_path, exist_ok=True)
    output_file = os.path.join(data_args.output_data_path, "pred.json")
    print("output_file_path", output_file)
    with open(output_file, 'w') as fout:
        record = {}
        relevant_scores = None
        for batch in eval_loader:
            cur += 1
            qid = batch["qid"]
            question = batch["question"]
            input_ids = batch["input_ids"].to(device)             # [seq_len]
            attention_mask = batch["attention_mask"].to(device)    # [seq_len]
            if training_args.mode == "open":
                relevant_scores = batch["relevant_scores"].to(device)  # [seq_len]
            gold_answers = batch["gold_answers"]
            with torch.no_grad():
                output_ids = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    relevant_scores=relevant_scores,
                    max_new_tokens=100,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    )
            generated_text = tokenizer.decode(output_ids[0][input_ids.shape[-1]:], skip_special_tokens=True).strip().replace("assistant", "").replace("<|im_start|>\n", "").replace("system\n", "")
            
            record["id"] = qid[0]
            record["question"] = question[0]
            record["pred"] = generated_text
            record["gold_answers"] = gold_answers
            fout.write(json.dumps(record)+ "\n")
            fout.flush()
            # predictions.append(generated_text)
            # golds.append(gold_answers) # gold_answers is a list of answers
            # if cur % 100 == 0:
            #     print(f"Processed {cur}")
            #breakpoint()



if __name__ == "__main__":
    inference()
