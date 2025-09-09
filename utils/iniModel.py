import transformers
from dataclasses import dataclass, field
from typing import Optional
import os
import sys
import shutil

import os
import sys
import importlib


def load_imodel_and_iconfig_package(model_pattern, src_path):
    # create loading path
    model_path = os.path.join(src_path, "model")

    if not os.path.exists(model_path):
        print(f"path not exist: {model_path}")
        return None, None

    if model_path not in sys.path:
        sys.path.append(model_path)

    # load model config
    try:
        IModelForCausalLM = importlib.import_module(
            f"{model_pattern}.modeling"
        ).IModelForCausalLM
        IConfig = importlib.import_module(f"{model_pattern}.configuration").IConfig
        return IModelForCausalLM, IConfig
    except ModuleNotFoundError as e:
        print(f"model load fail: {e}")
        return None, None


@dataclass
class InitModelArguments:
    model_name_or_path: Optional[str] = field(default="Qwen/Qwen2.5-7B")
    savepath_basemodel_namemodules: Optional[str] = field(default="Qwen/Qwen2.5-7B")
    savepath_imodel_namemodules: Optional[str] = field(default="Qwen/Qwen2.5-7B")
    src_path: Optional[str] = field(default="Qwen/Qwen2.5-7B")
    savedir_imodel: Optional[str] = field(default="Qwen/Qwen2.5-7B")
    model_pattern: Optional[str] = field(default="Qwen/Qwen2.5-7B")
    config_path: Optional[str] = field(default="Qwen/Qwen2.5-7B")


def initModel():
    parser = transformers.HfArgumentParser((InitModelArguments))
    (args,) = parser.parse_args_into_dataclasses()
    sys.path.append(args.src_path)

    IModelForCausalLM, IConfig = load_imodel_and_iconfig_package(
        args.model_pattern, args.src_path
    )
    iconfig = IConfig.from_pretrained(args.config_path)
    imodel = IModelForCausalLM.from_pretrained(args.model_name_or_path, config=iconfig)
    baseconfig = transformers.AutoConfig.from_pretrained(args.model_name_or_path)
    basemodel = transformers.AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path, config=baseconfig, trust_remote_code=True
    )

    # Save model name and module shape for debug
    # Open the file in write mode
    if not os.path.exists(args.savepath_basemodel_namemodules):
        os.makedirs(os.path.dirname(args.savepath_basemodel_namemodules), exist_ok=True)
    with open(args.savepath_basemodel_namemodules, "w") as f:
        # Print model named_modules() and write to file
        f.write("Base Model Submodules:\n")
        for name, module in basemodel.named_modules():
            f.write(f"{name}: {module}\n")

        # Print config and write to file
        f.write("\nModel Config:\n")
        f.write(str(baseconfig))

    if not os.path.exists(args.savepath_imodel_namemodules):
        os.makedirs(os.path.dirname(args.savepath_imodel_namemodules), exist_ok=True)
    with open(args.savepath_imodel_namemodules, "w") as f:
        # Print model named_modules() and write to file
        f.write("I Model Submodules:\n")
        for name, module in imodel.named_modules():
            f.write(f"{name}: {module}\n")

        # Print config and write to file
        f.write("\nIModel Config:\n")
        f.write(str(baseconfig))

    print("Start Model Reintialization")
    # Reintialize base model
    for layer_index in range(iconfig.num_hidden_layers):
        imodel.model.layers[layer_index].self_attn = (
                basemodel.model.layers[layer_index].self_attn
            )
        imodel.model.layers[layer_index].input_layernorm = (
            basemodel.model.layers[layer_index].input_layernorm
        )
        imodel.model.layers[layer_index].post_attention_layernorm = (
            basemodel.model.layers[layer_index].post_attention_layernorm
        )
        imodel.model.layers[layer_index].mlp = (
            basemodel.model.layers[layer_index].mlp
        )

    imodel.save_pretrained(
        save_directory=args.savedir_imodel, config=iconfig, safe_serialization=False
    )
    copy_files = []
    for item in os.listdir(args.model_name_or_path):
        if os.path.exists(os.path.join(args.savedir_imodel, item)):
            continue
        if item.startswith("pytorch_model") and item.endswith(".bin"):
            continue
        if item.endswith(".index.json") or item.endswith(".safetensors"):
            continue
        if item.endswith(".config.json") or item.endswith(".safetensors"):
            continue
        s = os.path.join(args.model_name_or_path, item)
        if os.path.isfile(s):
            shutil.copy(s, os.path.join(args.savedir_imodel, item))
        copy_files.append(os.path.join(args.savedir_imodel, item))

    shutil.copy(args.config_path, os.path.join(args.savedir_imodel, "config.json"))
    copy_files.append(os.path.join(args.savedir_imodel, "config.json"))
    print(f"iModel is saved in:\n{args.savedir_imodel} \n copy file:\n{copy_files}")


if __name__ == "__main__":
    initModel()
