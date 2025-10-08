
import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model,PeftModel, get_peft_model_state_dict
from peft.tuners.lora import LoraLayer
from datasets import load_dataset,Dataset
import argparse
from utils import replicability
import bitsandbytes as bnb
import json
import os
import tqdm
import random
import copy
# import pytrec_eval
# from pyserini.search.lucene import LuceneSearcher
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR

torch.backends.cuda.matmul.allow_tf32 = True # A bool that controls whether TensorFloat-32 tensor cores may be used in matrix multiplications on Ampere or newer GPUs.
IGNORE_INDEX = -100


def parser_binary(text):

    if text in ["Relevant", "Irrelevant"]:
        return "1" if text =="Relevant" else "0"

    print(f"Parsing:***********\nOriginal text:\n{text}\n")
    if "Relevant" in text:
        text_ = "1"
    elif "Irrelevant" in text or "Ir" in text:
        text_ = "0"
    else:
        text_ = "0"
    print(f"Parsed text:\n{text_}\n***********\n")
    return text_

def extract_first_digit(text):
    for char in text:
        if char.isdigit():
            return char
    return "0"

def parser_digit(text):
    if text in ["0", "1", "2", "3", "4"]:
        return text
    else:
        print(f"Parsing:***********\nOriginal text:\n{text}\n")
        text_ = extract_first_digit(text)
        print(f"Parsed text:\n{text_}\n***********\n")
    return text_

class Prompter:
    def __init__(self, args):
        self.args=args
        if self.args.prompt == "binary":
            self.template ="Instruction: Please assess the relevance of the provided passage to the following question. Please output \"Relevant\" or \"Irrelevant\".\n{demonstrations}Question: {question}\nPassage: {passage}\nOutput:"
            self.spliter="Output:"
            self.pos_label ="Relevant"
            self.neg_label="Irrelevant"
            self.demonstration="Question: {question}\nPassage: {passage}\nOutput: {output}\n"
            self.parser = parser_binary

        elif self.args.prompt == "ikat":
            self.template = "You are a search quality rater evaluating the relevance of web pages.\nGiven the persona of the user, user query, and a web page, you must provide a score on an integer scale of 0 to 4 to indicate to what extent the given document meets the information needs of the user.\nThe scores have the following meanings:\n\n0: fails to meet\n1: slightly meets\n2: moderately meets\n3: highly meets\n4: fully meets\n\nUser persona: {ptkb}\nQuery: {query}\nPassage: {passage}\nScore:"
            self.spliter = "Score:"
            self.labels = ["0", "1", "2", "3", "4"]
            self.parser = parser_digit
        else:
            raise Exception


class SavePeftModelCallback(transformers.TrainerCallback):
    def save_model(self, args, state, kwargs):
        print('Saving PEFT checkpoint...')
        if state.best_model_checkpoint is not None:
            checkpoint_folder = os.path.join(state.best_model_checkpoint, "adapter_model")
        else:
            checkpoint_folder = os.path.join(args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}")

        peft_model_path = os.path.join(checkpoint_folder, "adapter_model")
        kwargs["model"].save_pretrained(peft_model_path)

        pytorch_model_path = os.path.join(checkpoint_folder, "pytorch_model.bin")
        if os.path.exists(pytorch_model_path):
            os.remove(pytorch_model_path)

    def on_save(self, args, state, control, **kwargs):
        # Event called after a checkpoint save.
        self.save_model(args, state, kwargs)
        return control

    def on_train_end(self, args, state, control, **kwargs):
        # Event called at the end of training.
        def touch(fname, times=None):
            with open(fname, 'a'):
                os.utime(fname, times)

        touch(os.path.join(args.output_dir, 'completed'))
        self.save_model(args, state, kwargs)

def load_qpp_data(args):

    if args.query_demon_path is not None:
        query_demon = {}
        query_reader = open(args.query_demon_path, 'r').readlines()
        for line in query_reader:
            qid, qtext = line.split('\t')
            query_demon[qid] = qtext.replace("\t", "").replace("\n", "").replace("\r", "")

        searcher_demon = LuceneSearcher(args.index_demon_path)
        
        with open(args.run_demon_path, 'r') as f_run:
            run_demon = pytrec_eval.parse_run(f_run)

        with open(args.qrels_demon_path, 'r') as f_qrels:
            qrels_demon  = pytrec_eval.parse_qrel(f_qrels)

        # postive examples
        demonstration_list = []

        for qid, qtext in query_demon.items():
            if qid not in qrels_demon:
                continue

            demonstration=""

            pid_list = [pid for (pid, score) in sorted(run_demon[qid].items(), key=lambda x: x[1], reverse=True)]

            # sample a positive
            for pid in qrels_demon[qid]:
                passage_dict = json.loads(searcher_demon.doc(pid).raw())
                passage_text = passage_dict['contents'] if 'contents' in passage_dict else passage_dict['passage']
                passage_text = passage_text.replace("\t", " ").replace("\n", " ").replace("\r", " ")


                demonstration+= args.prompter.demonstration.format(question=qtext, passage=passage_text, output=args.prompter.pos_label)
                # one query only has one positive passage
                break

            # sample a negative
            for pid in qrels_demon[qid]:
                if pid in pid_list:
                    pid_list.remove(pid)

            if len(pid_list) < args.num_demon_per_class:
                print(qid, qrels[qid], pid_list)
                continue

            # one query only has a negative passage
            neg_pids = random.sample(pid_list, 1)
            for pid in neg_pids:
                passage_dict = json.loads(searcher_demon.doc(pid).raw())
                passage_text = passage_dict['contents'] if 'contents' in passage_dict else passage_dict['passage']
                passage_text = passage_text.replace("\t", " ").replace("\n", " ").replace("\r", " ")

            demonstration+= args.prompter.demonstration.format(question=qtext, passage=passage_text, output=args.prompter.neg_label)

            demonstration_list.append(demonstration)

        demonstration_list_sampled = random.sample(demonstration_list, args.num_demon_per_class)
        demonstrations = "".join(demonstration_list_sampled)

        print(f"demonstrations:\n{demonstrations}")


    query = {}
    doc_ids={}
    doc_texts={}
    query_reader = open(args.query_path, 'r').readlines()
    # count_query=0
    for line in query_reader:
        data = json.loads(line.strip())
        sample_id = data["qid"]
        rewrites = data["question"]
        query[sample_id] = rewrites
        doc_ids[sample_id] = data["doc_ids"]
        doc_texts[sample_id] = data["doc_text"]

    examples = []
    pos_num=0
    neg_num=0

    print("run load_qpp_data")
    for qid, qtext in query.items():

        pid_list = doc_ids[qid]

        if args.infer:
            # print("load_qpp_data args.infer")
            for idx, pid in enumerate(pid_list):
                example = {}
                example["example_id"] = f"{qid}#{idx + 1}#{pid}"

                # print(qid, pid,example["example_id"])
                passage_text = doc_texts[qid][idx]
                passage_text = passage_text.replace("\t", " ").replace("\n", " ").replace("\r", " ")

                # print("passage_text:",passage_text)

                if args.query_demon_path is not None:
                    example["input"] = args.prompter.template.format(demonstrations=demonstrations, question=qtext,passage=passage_text[:args.max_char_len])
                else:
                    example["input"] = args.prompter.template.format(demonstrations="", question=qtext,passage=passage_text[:args.max_char_len])
                example["output"] = args.prompter.neg_label
                # example["output"] = args.prompter.pos_label
                examples.append(example)
        else:
            for pid in qrels[qid]:
                example = {}
                example["example_id"] = f"{qid}#pos#{pid}"

                passage_dict = json.loads(searcher.doc(pid).raw())
                passage_text = passage_dict['contents'] if 'contents' in passage_dict else passage_dict['passage']
                
                passage_text = passage_text.replace("\t", " ").replace("\n", " ").replace("\r", " ")
                

                example["input"] = args.prompter.template.format(demonstrations="", question=qtext,passage=passage_text)
                example["output"] = args.prompter.pos_label
                examples.append(example)
                pos_num += 1

            # negative sampling
            pid_list_ = copy.deepcopy(pid_list)

            for pid in qrels[qid]:
                if pid in pid_list_:
                    pid_list_.remove(pid)

            if len(pid_list)<args.num_negs:
                print(f"Skip sampling negatives for {qid} because it has insufficient negatives:\n{qrels[qid]}, {run[qid]}, {pid_list_}")
                continue


            pid_list__ = pid_list_[:args.neg_top]

            neg_pid_list = random.sample(pid_list__, args.num_negs)

            for pid in neg_pid_list:
                #negative examples
                example = {}
                example["example_id"] = f"{qid}#neg#{pid}"
                str_pid=str(pid)
                passage_dict = json.loads(searcher.doc(str_pid).raw())
                passage_text = passage_dict['contents'] if 'contents' in passage_dict else passage_dict['passage']
                passage_text = passage_text.replace("\t", " ").replace("\n", " ").replace("\r", " ")

                example["input"] = args.prompter.template.format(demonstrations="", question=qtext, passage=passage_text)
                example["output"] = args.prompter.neg_label
                examples.append(example)
                neg_num+=1

    # assert len(examples)==(pos_num+neg_num), print("len(examples): ", len(examples))

    # print(f"pos_num: {pos_num}, neg_num: {neg_num}")
    print("sanity check:\n{} {}\n\n{} {}\n".format(examples[0]["input"],examples[0]["output"], examples[-1]["input"], examples[-1]["output"]))
    return examples

def load_rj_data(args):

    if args.query_demon_path is not None:
        query_demon = {}
        query_reader = open(args.query_demon_path, 'r').readlines()
        for line in query_reader:
            qid, qtext = line.split('\t')
            query_demon[qid] = qtext.replace("\t", "").replace("\n", "").replace("\r", "")

        searcher_demon = LuceneSearcher(args.index_demon_path)

        with open(args.run_demon_path, 'r') as f_run:
            run_demon = pytrec_eval.parse_run(f_run)

        with open(args.qrels_demon_path, 'r') as f_qrels:
            qrels_demon  = pytrec_eval.parse_qrel(f_qrels)

        # postive examples
        demonstration_list = []

        for qid, qtext in query_demon.items():
            if qid not in qrels_demon:
                continue

            demonstration=""

            pid_list = [pid for (pid, score) in sorted(run_demon[qid].items(), key=lambda x: x[1], reverse=True)]

            # sample one positive example
            for pid in qrels_demon[qid]:
                passage_dict = json.loads(searcher_demon.doc(pid).raw())
                passage_text = passage_dict['contents'] if 'contents' in passage_dict else passage_dict['passage']
                passage_text = passage_text.replace("\t", " ").replace("\n", " ").replace("\r", " ")


                demonstration+= args.prompter.demonstration.format(question=qtext, passage=passage_text, output=args.prompter.pos_label)
                # one query only has one positive passage
                break

            # sample a negative
            for pid in qrels_demon[qid]:
                if pid in pid_list:
                    pid_list.remove(pid)

            if len(pid_list) < args.num_demon_per_class:
                print(qid, qrels[qid], pid_list)
                continue

            # one query only has one negative passage
            neg_pids = random.sample(pid_list, 1)

            # 1
            for pid in neg_pids:
                passage_dict = json.loads(searcher_demon.doc(pid).raw())
                passage_text = passage_dict['contents'] if 'contents' in passage_dict else passage_dict['passage']
                passage_text = passage_text.replace("\t", " ").replace("\n", " ").replace("\r", " ")

            demonstration+= args.prompter.demonstration.format(question=qtext, passage=passage_text, output=args.prompter.neg_label)

            demonstration_list.append(demonstration)

        demonstration_list_sampled = random.sample(demonstration_list, args.num_demon_per_class)
        demonstrations = "".join(demonstration_list_sampled)

        print(f"demonstrations:\n{demonstrations}")

    query = {}
    query_reader = open(args.query_path, 'r').readlines()
    for line in query_reader:
        qid, qtext = line.split('\t')
        query[qid] = qtext.replace("\t", "").replace("\n", "").replace("\r", "")

    if "msmarco" in args.dataset_class:
        searcher = LuceneSearcher(args.index_path)

    elif "ikat" == args.dataset_class:
        ptkb = {}
        ptkb_reader = open(args.ptkb_path, 'r').readlines()
        for line in ptkb_reader:
            qid, ptkb_text = line.split('\t')
            ptkb[qid] = ptkb_text.replace("\t", "").replace("\n", "").replace("\r", "")

        corpus = {}
        corpus_reader = open(args.index_path, 'r').readlines()
        for line in corpus_reader:
            pid, passage_text = line.split('\t')
            corpus[pid] = passage_text.replace("\t", "").replace("\n", "").replace("\r", "")
    else:
        raise Exception

    with open(args.qrels_path, 'r') as f_qrels:
        qrels = pytrec_eval.parse_qrel(f_qrels)

    examples = []
    count={}

    for qid, pid2rel in qrels.items():
        for pid, rel in pid2rel.items():
            example = {}

            if "msmarco" in args.dataset_class:
                passage_dict = json.loads(searcher.doc(pid).raw())
                passage_text = passage_dict['contents'] if 'contents' in passage_dict else passage_dict['passage']
                passage_text = passage_text.replace("\t", " ").replace("\n", " ").replace("\r", " ")

                if args.query_demon_path is not None:
                    example["input"] = args.prompter.template.format(demonstrations=demonstrations, question=query[qid],passage=passage_text[:args.max_char_len])
                else:
                    example["input"] = args.prompter.template.format(demonstrations="", question=query[qid],passage=passage_text[:args.max_char_len])


                if rel >= args.rel_threshold:
                    example["output"] = args.prompter.pos_label
                    example["example_id"] = f"{qid}#pos#{pid}"
                else:
                    example["output"] = args.prompter.neg_label
                    example["example_id"] = f"{qid}#neg#{pid}"

            elif "ikat" == args.dataset_class:
                if args.prompt == "ikat":
                    example["input"] = args.prompter.template.format(query=query[qid], ptkb=ptkb[qid],passage=corpus[pid][:args.max_char_len])
                    example["example_id"] = f"{qid}#{rel}#{pid}"
                    example["output"] = str(rel)

                elif args.prompt == "binary":

                    example["input"] = args.prompter.template.format(demonstrations="", question=query[qid], passage=corpus[pid][:args.max_char_len])

                    if rel >= args.rel_threshold:
                        example["output"] = args.prompter.pos_label
                        example["example_id"] = f"{qid}#pos#{pid}"
                    else:
                        example["output"] = args.prompter.neg_label
                        example["example_id"] = f"{qid}#neg#{pid}"


            if example["output"] not in count:
                count[example["output"]]=1
            else:
                count[example["output"]]+=1

            #print(example["input"],"\n")

            examples.append(example)

    return examples


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )

def find_all_linear_names(model):
    cls = bnb.nn.Linear4bit
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names: # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)


def train(args):

    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1

    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        gradient_accumulation_steps = gradient_accumulation_steps // world_size

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        device_map=device_map,
        torch_dtype=torch.bfloat16,
        #cache_dir=args.cache_dir,
        #token=args.token,
        quantization_config=BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        llm_int8_has_fp16_weight=False,
    ))

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, padding_side=args.padding_side)#, cache_dir=args.cache_dir, token=args.token)

    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = args.padding_side

    model.config.torch_dtype =torch.bfloat16
    model.config.pad_token_id = model.config.eos_token_id
    model.generation_config.pad_token_id = model.generation_config.eos_token_id
    #model.generation_config.eos_token_id = model.generation_config.eos_token_id

    print(f"model.config:\n{model.config}")
    print(f"model.generation_config:\n{model.generation_config}")

    if not ddp and torch.cuda.device_count() > 1:
        # keeps Trainer from trying its own DataParallelism when more than 1 gpu is available
        setattr(model, 'model_parallel', True)
        setattr(model, 'is_parallelizable', True)

    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model.gradient_checkpointing_enable()  # reduce the memeory, but increase the training time

    model = get_peft_model(model, LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules = find_all_linear_names(model)
    ))

    print_trainable_parameters(model)

    def tokenize(prompt, add_eos_token=True):
        result = tokenizer(
            prompt,
            truncation=True, # Truncate to a maximum length specified with the argument max_length or to the maximum acceptable input length for the model if that argument is not provided.
            max_length=args.max_input_length,
            padding=False,
            return_tensors=None,
        )

        if (
                result["input_ids"][-1] != tokenizer.eos_token_id
                and len(result["input_ids"]) < args.max_input_length
                and add_eos_token
        ):
            result["input_ids"].append(tokenizer.eos_token_id)
            result["attention_mask"].append(1)
        
        result["labels"] = result["input_ids"].copy()

        return result

    def generate_and_tokenize_prompt(data_point):
        prompt_label = " ".join([data_point["input"], data_point["output"]])
        tokenized_prompt_label = tokenize(prompt_label)

        if not args.train_on_inputs:
            prompt = data_point["input"]
            tokenized_prompt = tokenize(prompt, add_eos_token=False)
            prompt_len = len(tokenized_prompt["input_ids"])
            tokenized_prompt_label["labels"] = [IGNORE_INDEX] * prompt_len + tokenized_prompt_label["labels"][prompt_len:]  # could be sped up, probably

        return tokenized_prompt_label

    if args.rj:
        examples = load_rj_data(args)
    else:
        examples = load_qpp_data(args)

    dataset = Dataset.from_list(examples)
    dataset = dataset.shuffle().map(generate_and_tokenize_prompt)
    print(f"dataset.column_names:\n{dataset.column_names}")

    training_args = transformers.TrainingArguments(
            #remove_unused_columns=False, #  Whether or not to automatically remove the columns unused by the model forward method
            report_to='none', # default to ['tensorboard', 'wandb']
            num_train_epochs=args.num_epochs,
            per_device_train_batch_size=args.batch_size, # 8 for 65B
            gradient_accumulation_steps=1,
            #warmup_ratio=0.05,
            #max_steps=100,
            #save_steps = 10,
            save_strategy="epoch",
            save_total_limit=None, # If a value is passed, will limit the total amount of checkpoints. Deletes the older checkpoints in output_dir.
            max_grad_norm=args.max_grad_norm,
            learning_rate=args.lr,
            fp16=True,
            logging_steps=args.logging_steps,
            output_dir=args.checkpoint_path_,
            optim=args.optim,
            lr_scheduler_type="constant",
            group_by_length=args.group_by_length, #  Whether or not to group together samples of roughly the same length in the training dataset (to minimize padding applied and be more efficient). Only useful if applying dynamic padding.
        )

    print(f"training_args:\n{training_args}")

    data_collator = transformers.DataCollatorForSeq2Seq(
        tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
    )

    trainer = transformers.Trainer(
        model=model,
        train_dataset=dataset,
        args=training_args,
        data_collator=data_collator,
        callbacks=[SavePeftModelCallback]
    )
    model.config.use_cache = False  # silence the warnings. Please re-enable for inference!

    for name, module in model.named_modules():
        #if isinstance(module, LoraLayer):
            #module = module.to(torch.bfloat16)

        if 'norm' in name:
            module = module.to(torch.float32)

        if 'lm_head' in name or 'embed_tokens' in name:
            if hasattr(module, 'weight'):
                if module.weight.dtype == torch.float32:
                    module = module.to(torch.bfloat16)

    trainer.train()
    model.save_pretrained(args.checkpoint_path_)
    return None


def infer(args):


    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        #cache_dir=args.cache_dir,
        #token=args.token,
        quantization_config=BitsAndBytesConfig(
        load_in_4bit=True,
        load_in_8bit=False,
        #bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    )

    if args.checkpoint_name:
        model = PeftModel.from_pretrained(model, args.checkpoint_path_)
        #model = model.merge_and_unload() # not necessary

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path,padding_side=args.padding_side)#, cache_dir=args.cache_dir, token=args.token)

    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = args.padding_side

    model.config.torch_dtype =torch.bfloat16
    model.config.pad_token_id = model.config.eos_token_id
    model.generation_config.pad_token_id = model.generation_config.eos_token_id

    if isinstance(model.generation_config.eos_token_id, list):
        model.generation_config.pad_token_id = model.generation_config.eos_token_id[0] # llama 3 128001
        #model.generation_config.eos_token_id = model.generation_config.eos_token_id[0]
    else:
        model.generation_config.pad_token_id = model.generation_config.eos_token_id # llama 3 128001

    model.eval()


    if args.rj:
        examples = load_rj_data(args)
    else:
        examples = load_qpp_data(args)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("device:",device)

    it = range(0, len(examples), args.batch_size)
    print("len(examples)=",len(examples),"    batch_size=",args.batch_size)

    for start_idx in tqdm.tqdm(it,miniters=10, desc="Processing"):
        # one batch
        rng = slice(start_idx, start_idx + args.batch_size)

        # padding=True or 'longest': Pad to the longest sequence in the batch (or no padding if only a single sequence if provided).
        inputs = tokenizer([example['input'] for example in examples[rng]], padding=True, truncation=True, max_length=args.max_input_length, return_tensors='pt',return_attention_mask=True).to(device)
                
        with torch.no_grad():
            outputs = model(**inputs)
            last_token_logits = outputs.logits[:, -1, :]  # shape: [batch_size, vocab_size]

        relevant_tokens = tokenizer.encode("Relevant", add_special_tokens=False)

        relevant_id = relevant_tokens[0]
        for idx, example in enumerate(examples[rng]):
            example["score_relevant"] = last_token_logits[idx, relevant_id].item()

    with open(f"{args.output_dir_}", 'w') as rj_w:
        for idx, example in enumerate(examples):
            qid, rank, pid = example["example_id"].split("#")
            # rel = example["prediction"]
            # pre = example["prediction"]
            rel = example["score_relevant"]
            # rel = example["score_irrelevant"]
            rj_w.write(f"{qid} 0 {pid} {rel}\n")

    return None


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--infer", action='store_true')
    parser.add_argument("--rj", action='store_true')

    parser.add_argument("--prompt", type=str) # binary, ikat_digit

    parser.add_argument("--model_name_or_path", type=str)
    parser.add_argument("--checkpoint_path", type=str)
    parser.add_argument("--checkpoint_name", type=str, default=None)
    parser.add_argument("--demonstration_path", type=str, default=None)
    parser.add_argument("--train_on_inputs", action='store_true')
    parser.add_argument("--output_dir", type=str)

    parser.add_argument("--query_path", type=str, default=None)
    parser.add_argument("--ptkb_path", type=str, default=None)
    parser.add_argument("--index_path", type=str, default=None)
    parser.add_argument("--run_path", type=str, default=None)
    parser.add_argument("--qrels_path", type=str, default=None)

    parser.add_argument("--query_demon_path", type=str, default=None)
    parser.add_argument("--index_demon_path", type=str, default=None)
    parser.add_argument("--run_demon_path", type=str, default=None)
    parser.add_argument("--qrels_demon_path", type=str, default=None)
    

    parser.add_argument("--truncation_side", type=str, default='left')
    parser.add_argument("--padding_side", type=str, default='left')
    parser.add_argument("--max_char_len", type=int, default=1400)
    parser.add_argument("--max_input_length", type=int, default=2048)
    parser.add_argument("--max_new_tokens", type=int, default=4)

    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)  # 1e-4
    parser.add_argument("--optim", type=str, default="paged_adamw_32bit")
    parser.add_argument("--max_grad_norm", type=float, default=0.3)
    parser.add_argument("--group_by_length", action='store_true')
    parser.add_argument("--num_epochs", type=int, default=5)
    parser.add_argument("--per_device_train_batch_size", type=int, default=32)
    parser.add_argument("--logging_steps", type=int, default=10)

    parser.add_argument("--lora_r", type=int, default=64)  
    parser.add_argument("--lora_alpha", type=int, default=16) 
    parser.add_argument("--lora_dropout", type=float, default=0.1)

    parser.add_argument("--num_demon_per_class", type=int, default=1)
    parser.add_argument("--num_negs", type=int, default=None)
    parser.add_argument("--neg_top", type=int, default=1000)
    parser.add_argument("--k", type=int, default=None)

    args = parser.parse_args()

    print(args)

    args.dataset_class = args.query_path.split("/")[-1]
    # args.dataset_class = args.query_path.split("/")[-3]
    args.dataset_name = args.query_path.split("/")[-1].split(".")[0]
    args.query_type = "-".join(args.query_path.split("/")[-1].split(".")[1].split("-")[1:])
    # args.qrels_name = ".".join(args.qrels_path.split("/")[-1].split(".")[0:-1])

    args.prompter = Prompter(args)

    if not args.rj:
        # args.retriever = "-".join(args.run_path.split("/")[-1].split(".")[1].split("-")[1:])
        args.retriever = "test_NQ_QPP"

    args.base_model = args.model_name_or_path.split("/")[-1]

    if args.infer is True:
        # inference mode with a fine-tuned checkpoint
        if args.checkpoint_name:
            args.checkpoint_path_ = f"{args.checkpoint_path}/{args.checkpoint_name}/"
            if "/" in args.checkpoint_name:
                args.checkpoint_name=args.checkpoint_name.replace("/","-")
            if args.rj:
                args.setup = f"{args.qrels_name}.{args.query_type}-{args.base_model}-ckpt-{args.checkpoint_name}"
            else:
                args.setup = f"{args.dataset_name}.{args.retriever}.{args.query_type}-{args.base_model}-ckpt-{args.checkpoint_name}.k{args.k}"
        else:
            # in-context learning (few-shot) or zero-shot
            if args.rj:
                if args.query_demon_path is not None:
                    dataset_name_demon = args.query_demon_path.split("/")[-1].split(".")[0]
                    retriever_demon = "-".join(args.run_demon_path.split("/")[-1].split(".")[1].split("-")[1:])
                    setup_demon=f"{dataset_name_demon}.{retriever_demon}-demon{args.num_demon_per_class}"

                    args.setup = f"{args.qrels_name}.{args.query_type}-{args.base_model}-icl-{setup_demon}"
                else:
                    args.setup = f"{args.qrels_name}.{args.query_type}-{args.base_model}"
            else:
                if args.query_demon_path is not None:
                    dataset_name_demon = args.query_demon_path.split("/")[-1].split(".")[0]
                    retriever_demon = "-".join(args.run_demon_path.split("/")[-1].split(".")[1].split("-")[1:])
                    setup_demon=f"{dataset_name_demon}.{retriever_demon}-demon{args.num_demon_per_class}"

                    args.setup = f"{args.dataset_name}.{args.retriever}.{args.query_type}-{args.base_model}-icl-{setup_demon}"
                else:
                    args.setup = f"{args.dataset_name}.{args.retriever}.{args.query_type}-{args.base_model}"


        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)

        args.output_dir_ = f"{args.output_dir}/{args.setup}"


    else:
        # training mode
        if args.rj:
            args.setup = f"{args.qrels_name}.{args.base_model}"
        else:
            args.setup = f"{args.dataset_name}.{args.retriever}.{args.query_type}-{args.base_model}-neg{args.num_negs}-top{args.neg_top}"

        args.checkpoint_path_ = f"{args.checkpoint_path}/{args.setup}/"

    replicability(seed=args.random_seed)

    if args.infer is True:
        infer(args)
    else:
        train(args)


