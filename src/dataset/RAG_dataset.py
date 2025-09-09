import sys
sys.path.append('..')
sys.path.append('.')
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
import json, pickle
import numpy as np
import random
from tqdm import trange, tqdm
from transformers import Trainer
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from dataclasses import dataclass
from typing import List, Dict, Any

def pload(path):
	with open(path, 'rb') as f:
		res = pickle.load(f)
	print('load path = {} object'.format(path))
	return res

class Train_Open_Dataset(Dataset):
    #def __init__(self, tokenizer, NQ_data_path, NQ_RAG_data_path, RAG_text_path, max_length=1024, top_k=5):
    def __init__(self, tokenizer, data_args, max_length=1300):
        self.tokenizer = tokenizer
        self.max_length = data_args.top_k * 150 + 200
        self.data = []
        self.RAG_data = []
        self.pid2passage = {}
        self.top_k = data_args.top_k
        self.data_args = data_args
        self.NQ_data_path = data_args.NQ_data_path
        self.NQ_RAG_data_path = data_args.NQ_RAG_data_path
        self.hotpotqa_data_path = data_args.hotpotqa_data_path
        self.hotpotqa_RAG_data_path = data_args.hotpotqa_RAG_data_path
        self.RAG_text_path = data_args.RAG_text_path

        if len(self.NQ_data_path) > 0:
            with open(self.NQ_data_path, encoding="utf-8") as f:
                NQ_data = f.readlines()
                self.data.extend(NQ_data)

            with open(self.NQ_RAG_data_path, encoding="utf-8") as f:
                NQ_RAG_data = f.readlines()
                self.RAG_data.extend(NQ_RAG_data)
            print("loaded NQ data!!!!!")
            assert len(NQ_data) == len(NQ_RAG_data)

        if len(self.hotpotqa_data_path) > 0:
            with open(self.hotpotqa_data_path, encoding="utf-8") as f:
                hotpotqa_data = f.readlines()
                self.data.extend(hotpotqa_data)

            with open(self.hotpotqa_RAG_data_path, encoding="utf-8") as f:
                hotpotqa_RAG_data = f.readlines()
                self.RAG_data.extend(hotpotqa_RAG_data)
            print("loaded hotpotqa data!!!")
            assert len(hotpotqa_data) == len(hotpotqa_RAG_data)

        self.pid2passage = pload(self.RAG_text_path)

    def filtering_invalid_string(self, psg_string):
        #invalid_tokens = []
        for i in range(100):
            invalid_token = f"!{str(i)} "
            #invalid_tokens.append(invalid_token)
            #psg_string = psg_string.replace(invalid_token, "").strip()
            #psg_string = psg_string.replace("FDFDFDFDFDFDFDFDF", "").strip()
        return psg_string
    
    def inject_less_relevant_psg(self, context_example, position_list, irrel_nums):
        '''
        position_list: indicate the position in top_k, used for RAG training
        irrel_nums: the number of totally irrlevant passages for RAG training
        '''
        # extract the other/indicated top-k
        full_RAG_pid = context_example["top_pid"]
        full_RAG_scores = context_example["top_pid_score"]
        irrel_RAG_pid = context_example["irrel_pid"]
        irrel_RAG_scores = context_example["irrel_pid_score"]
        add_RAG_pid, add_RAG_scores = [], []
        for pos in position_list:
            add_RAG_pid.append(full_RAG_pid[pos - 1])
            add_RAG_scores.append(full_RAG_scores[pos - 1])
        #len_collection = 21015324
        for i in range(irrel_nums):
            #irrel_int = random.randint(1, len_collection)
            add_RAG_pid.append(irrel_RAG_pid[i])
            add_RAG_scores.append(irrel_RAG_scores[i])
        return add_RAG_pid, add_RAG_scores

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data_example = json.loads(self.data[idx])
        context_example = json.loads(self.RAG_data[idx])
        RAG_pid, RAG_scores = [], []
        if self.data_args.add_irrelevant_psg:
            RAG_pid = context_example["top_pid"][:self.top_k - 5]
            RAG_scores = context_example["top_pid_score"][:self.top_k - 5]
        else:
            RAG_pid = context_example["top_pid"][:self.top_k]
            RAG_scores = context_example["top_pid_score"][:self.top_k]
        # inject irrelevant psgs for RAG training
        if self.data_args.add_irrelevant_psg:
            position_list = [30, 50, 100] # add three psg in top-k
            irrel_nums = 2
            add_RAG_pid, add_RAG_scores = self.inject_less_relevant_psg(context_example, position_list, irrel_nums)
            RAG_pid.extend(add_RAG_pid)
            RAG_scores.extend(add_RAG_scores)

        if self.data_args.shuffle_RAG:
            RAG_paired = list(zip(RAG_pid, RAG_scores))
            random.shuffle(RAG_paired)
            shuffled_RAG_pid, shuffled_RAG_scores = zip(*RAG_paired)
            RAG_pid, RAG_scores = list(shuffled_RAG_pid), list(shuffled_RAG_scores)
        
        #breakpoint()
        assert len(RAG_pid) == self.top_k
        
        # Combine question + retrieved passages into input prompt        
        context_parts = []
        for i, pid in enumerate(RAG_pid[:self.top_k]):
            passage_token = f"Passage_{i+1}:"
            context_parts.append(f"{passage_token} {self.pid2passage[pid]}")
        context = "\n".join(context_parts)
        question = data_example["question"]
        if "golden_answers" in data_example: # NQ
            answer = data_example["golden_answers"][0]
        elif "answer" in data_example: # hotpotqa
            answer = data_example["answer"]
        #context = "1 2 3 4 !5\n"
        # Prepare chat format input
        messages = [
            {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
            {"role": "user", "content": f"You should answer the question by referring to the knowledge provided below and integrating the usefulness of your own knowledge.\n{context}\n\nQuestion:{question}\n"},
            {"role": "assistant", "content": f"{answer}"}
        ]

        # Tokenize using Qwen tokenizer's chat template

        tokenized_text = self.tokenizer.apply_chat_template(
            messages, tokenize=False
        )
        
        tokenized = self.tokenizer(
            tokenized_text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_length
        )
        input_ids = tokenized["input_ids"].squeeze(0)
        attention_mask = tokenized["attention_mask"].squeeze(0)

        # normalization
        if self.data_args.normalization_type == "min-max":
            min_score, max_score = min(RAG_scores), max(RAG_scores)
            norm_scores = [(s - min_score + 0.1) / (max_score - min_score) for s in RAG_scores]
        elif self.data_args.normalization_type == "exp":
            decay_lambda = 0.5
            ranks = np.arange(len(RAG_scores))  # 0, 1, 2, ..., 9
            decay_weights = np.exp(-decay_lambda * ranks)
            norm_scores = decay_weights / decay_weights.sum()
        else:
            max_score = max(RAG_scores) if max(RAG_scores) > 0 else 1.0
            norm_scores = [score / max_score for score in RAG_scores]

        #breakpoint()
        # mask before ground_truth
        im_start_id = self.tokenizer.convert_tokens_to_ids("<|im_start|>")
        assistant_token_id = self.tokenizer.convert_tokens_to_ids("assistant")
        im_start_positions = (input_ids == im_start_id).nonzero(as_tuple=True)[0].tolist()
        label_start = 0
        for i in reversed(im_start_positions):
            try:
                if input_ids[i+1].item() == assistant_token_id:
                    label_start = i
                    break
            except:
                label_start = input_ids[i]

        labels = input_ids.clone()
        labels[:label_start] = -100
        labels[input_ids == self.tokenizer.pad_token_id] = -100
        # self.tokenizer.decode(input_ids)

        # Relevance score assignment
        relevance_scores = torch.ones(self.max_length, dtype=torch.float)
        passage_start_indices = []
        for i in range(self.top_k):
            tok = f"Passage_{i+1}:"
            tok_id = self.tokenizer.convert_tokens_to_ids(tok)
            if tok_id == self.tokenizer.unk_token_id:
                raise ValueError(f"Token '{tok}' is not defined in tokenizer vob, please add_special_tokens")
            # find token's position in input_ids, only use the first one
            matches = (input_ids == tok_id).nonzero(as_tuple=True)[0]
            if len(matches) == 0: #or data_example["id"] == "train_96":
                #bad = 1
                break
                #raise ValueError(f"Token '{tok}' is not in the list, please check top_k")
            passage_start_indices.append(matches[0].item())

        # extract the position
        passage_spans = []
        if label_start == 0:
            label_start = 1024
        #for i in range(self.top_k):
        for i in range(len(passage_start_indices)):
            start = passage_start_indices[i]
            end = passage_start_indices[i+1] if i < len(passage_start_indices) - 1 else label_start - 1
            #end = passage_start_indices[i+1] if i < self.top_k - 1 else label_start - 1
            passage_spans.append((start, end))

        # assign relevance score
        for i, (start, end) in enumerate(passage_spans):
            relevance_scores[start:end] = norm_scores[i]
        #breakpoint()
        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
            "passage_spans": passage_spans, # for pair-wise ranking loss
            "relevant_scores": relevance_scores, # for opendecoder
    }

class Train_MultiOpen_Dataset(Dataset):
    #def __init__(self, tokenizer, NQ_data_path, NQ_RAG_data_path, RAG_text_path, max_length=1024, top_k=5):
    def __init__(self, tokenizer, data_args, max_length=1300):
        self.tokenizer = tokenizer
        self.max_length = data_args.top_k * 150 + 200
        self.data, self.RAG_data = [], []
        self.LLM_score_data, self.QPP_score_data = [], []
        self.pid2passage = {}
        self.top_k = data_args.top_k
        self.data_args = data_args
        self.NQ_data_path = data_args.NQ_data_path
        self.NQ_RAG_data_path = data_args.NQ_RAG_data_path
        self.NQ_LLM_score_path = data_args.NQ_LLM_score_path
        self.NQ_QPP_score_path = data_args.NQ_QPP_score_path
        self.hotpotqa_data_path = data_args.hotpotqa_data_path
        self.hotpotqa_RAG_data_path = data_args.hotpotqa_RAG_data_path
        self.hotpotqa_LLM_score_path = data_args.hotpotqa_LLM_score_path
        self.hotpotqa_QPP_score_path = data_args.hotpotqa_QPP_score_path
        self.RAG_text_path = data_args.RAG_text_path

        if len(self.NQ_data_path) > 0:
            with open(self.NQ_data_path, encoding="utf-8") as f:
                NQ_data = f.readlines()
                self.data.extend(NQ_data)

            with open(self.NQ_RAG_data_path, encoding="utf-8") as f:
                NQ_RAG_data = f.readlines()
                self.RAG_data.extend(NQ_RAG_data)

            if self.data_args.add_LLM_scores:
                with open(self.NQ_LLM_score_path, encoding="utf-8") as f:
                    NQ_LLM_score_data = f.readlines()
                    self.LLM_score_data.extend(NQ_LLM_score_data)
            if self.data_args.add_QPP_scores:
                with open(self.NQ_QPP_score_path, encoding="utf-8") as f:
                    NQ_QPP_score_data = f.readlines()
                    self.QPP_score_data.extend(NQ_QPP_score_data)
            print("loaded NQ data!!!!!")
            assert len(NQ_data) == len(NQ_RAG_data)

        if len(self.hotpotqa_data_path) > 0:
            with open(self.hotpotqa_data_path, encoding="utf-8") as f:
                hotpotqa_data = f.readlines()
                self.data.extend(hotpotqa_data)

            with open(self.hotpotqa_RAG_data_path, encoding="utf-8") as f:
                hotpotqa_RAG_data = f.readlines()
                self.RAG_data.extend(hotpotqa_RAG_data)

            if self.data_args.add_LLM_scores:
                with open(self.hotpotqa_LLM_score_path, encoding="utf-8") as f:
                    hotpotqa_LLM_score_data = f.readlines()
                    self.LLM_score_data.extend(hotpotqa_LLM_score_data)
            if self.data_args.add_QPP_scores:
                with open(self.hotpotqa_QPP_score_path, encoding="utf-8") as f:
                    hotpotqa_QPP_score_data = f.readlines()
                    self.QPP_score_data.extend(hotpotqa_QPP_score_data)
            print("loaded hotpotqa data!!!")
            assert len(hotpotqa_data) == len(hotpotqa_RAG_data)

        self.pid2passage = pload(self.RAG_text_path)

    def filtering_invalid_string(self, psg_string):
        #invalid_tokens = []
        for i in range(100):
            invalid_token = f"!{str(i)} "
            #invalid_tokens.append(invalid_token)
            #psg_string = psg_string.replace(invalid_token, "").strip()
            #psg_string = psg_string.replace("FDFDFDFDFDFDFDFDF", "").strip()
        return psg_string
    
    def inject_less_relevant_psg(self, context_example, position_list, irrel_nums):
        '''
        position_list: indicate the position in top_k, used for RAG training
        irrel_nums: the number of totally irrlevant passages for RAG training
        '''
        # extract the other/indicated top-k
        full_RAG_pid = context_example["top_pid"]
        full_RAG_scores = context_example["top_pid_score"]
        irrel_RAG_pid = context_example["irrel_pid"]
        irrel_RAG_scores = context_example["irrel_pid_score"]
        add_RAG_pid, add_RAG_scores = [], []
        for pos in position_list:
            add_RAG_pid.append(full_RAG_pid[pos - 1])
            add_RAG_scores.append(full_RAG_scores[pos - 1])
        #len_collection = 21015324
        for i in range(irrel_nums):
            #irrel_int = random.randint(1, len_collection)
            add_RAG_pid.append(irrel_RAG_pid[i])
            add_RAG_scores.append(irrel_RAG_scores[i])
        return add_RAG_pid, add_RAG_scores

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data_example = json.loads(self.data[idx])
        context_example = json.loads(self.RAG_data[idx])
        if len(self.LLM_score_data) > 0 and self.data_args.add_LLM_scores:
            LLM_score_example = json.loads(self.LLM_score_data[idx])
        if len(self.QPP_score_data) > 0 and self.data_args.add_QPP_scores:
            QPP_score_example = json.loads(self.QPP_score_data[idx])
        RAG_pid, RAG_scores = [], []
        LLM_scores, QPP_scores = [], []
        if self.data_args.add_irrelevant_psg:
            RAG_pid = context_example["top_pid"][:self.top_k - 5]
            RAG_scores = context_example["top_pid_score"][:self.top_k - 5]
        else:
            RAG_pid = context_example["top_pid"][:self.top_k]
            RAG_scores = context_example["top_pid_score"][:self.top_k]
        # inject irrelevant psgs for RAG training
        if self.data_args.add_irrelevant_psg:
            position_list = [30, 50, 100] # add three psg in top-k
            irrel_nums = 2
            add_RAG_pid, add_RAG_scores = self.inject_less_relevant_psg(context_example, position_list, irrel_nums)
            RAG_pid.extend(add_RAG_pid)
            RAG_scores.extend(add_RAG_scores)
            if self.data_args.add_LLM_scores:
                assert LLM_score_example["doc_ids"] == RAG_pid
                LLM_scores.extend(LLM_score_example["LLM_scores"])
            if self.data_args.add_QPP_scores:
                #assert QPP_score_example["doc_ids"] == RAG_pid
                QPP_scores.extend(QPP_score_example["QPP_scores"])
        else:
            if self.data_args.add_LLM_scores:
                if len(LLM_score_example["doc_ids"]) == 10:
                    assert LLM_score_example["doc_ids"] == RAG_pid
                    LLM_scores.extend(LLM_score_example["LLM_scores"])
                elif len(LLM_score_example["doc_ids"]) > 10:
                    LLM_docids = LLM_score_example["doc_ids"][:10]
                    assert LLM_docids == RAG_pid
                    LLM_scores.extend(LLM_score_example["LLM_scores"][:10])
            if self.data_args.add_QPP_scores:
                if len(QPP_score_example["doc_ids"]) == 10:
                    assert QPP_score_example["doc_ids"] == RAG_pid
                    QPP_scores.extend(QPP_score_example["QPP_scores"])
                elif len(QPP_score_example["doc_ids"]) > 10:
                    QPP_docids = QPP_score_example["doc_ids"][:10]
                    assert QPP_docids == RAG_pid
                    QPP_scores.extend(QPP_score_example["QPP_scores"][:10])
        

        if self.data_args.shuffle_RAG:
            RAG_paired = list(zip(RAG_pid, LLM_scores))
            random.shuffle(RAG_paired)
            shuffled_RAG_pid, shuffled_RAG_scores = zip(*RAG_paired)
            RAG_pid, LLM_scores = list(shuffled_RAG_pid), list(shuffled_RAG_scores)
        
        #breakpoint()
        assert len(RAG_pid) == self.top_k
        
        # Combine question + retrieved passages into input prompt        
        context_parts = []
        for i, pid in enumerate(RAG_pid[:self.top_k]):
            passage_token = f"Passage_{i+1}:"
            context_parts.append(f"{passage_token} {self.pid2passage[pid]}")
        context = "\n".join(context_parts)
        question = data_example["question"]
        if "golden_answers" in data_example: # NQ
            answer = data_example["golden_answers"][0]
        elif "answer" in data_example: # hotpotqa
            answer = data_example["answer"]
        #context = "1 2 3 4 !5\n"
        # Prepare chat format input
        messages = [
            {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
            {"role": "user", "content": f"You should answer the question by referring to the knowledge provided below and integrating the usefulness of your own knowledge.\n{context}\n\nQuestion:{question}\n"},
            {"role": "assistant", "content": f"{answer}"}
        ]

        # Tokenize using Qwen tokenizer's chat template

        tokenized_text = self.tokenizer.apply_chat_template(
            messages, tokenize=False
        )
        
        tokenized = self.tokenizer(
            tokenized_text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_length
        )
        input_ids = tokenized["input_ids"].squeeze(0)
        attention_mask = tokenized["attention_mask"].squeeze(0)

        # normalization
        if self.data_args.normalization_type == "min-max":
            min_score, max_score = min(RAG_scores), max(RAG_scores)
            norm_scores = [(s - min_score + 0.1) / (max_score - min_score) for s in RAG_scores]
        elif self.data_args.normalization_type == "exp":
            decay_lambda = 0.5
            ranks = np.arange(len(RAG_scores))  # 0, 1, 2, ..., 9
            decay_weights = np.exp(-decay_lambda * ranks)
            norm_scores = decay_weights / decay_weights.sum()
        else:
            if self.data_args.add_LLM_scores and not self.data_args.add_QPP_scores:
                # old
                # agg_scores = [a + b / 2 for a, b in zip(RAG_scores, LLM_scores)]
                # max_score = max(agg_scores) if max(agg_scores) > 0 else 1.0
                # norm_scores = [score / max_score for score in agg_scores]
                # norm_scores = LLM_scores
                # new
                #max_LLM_score = max(LLM_scores) if max(LLM_scores) > 0 else 1.0
                #norm_LLM_scores = [score / max_LLM_score for score in LLM_scores]
                min_val, max_val = min(LLM_scores), max(LLM_scores)
                norm_LLM_scores = [(x - min_val) / (max_val - min_val) for x in LLM_scores]
                # norm_scores = norm_LLM_scores
                # max_RAG_score = max(LLM_scores) if max(RAG_scores) > 0 else 1.0
                # norm_RAG_scores = [score / max_RAG_score for score in RAG_scores]
                agg_scores = [a + b / 2 for a, b in zip(RAG_scores, norm_LLM_scores)]
                max_score = max(agg_scores) if max(agg_scores) > 0 else 1.0
                norm_scores = [score / max_score for score in agg_scores]
            elif self.data_args.add_QPP_scores and not self.data_args.add_LLM_scores:
                max_QPP_score = max(QPP_scores) if max(QPP_scores) > 0 else 1.0
                norm_QPP_scores = [score / max_QPP_score for score in QPP_scores]
                norm_scores = norm_QPP_scores
                # agg_scores = [a + b / 2 for a, b in zip(RAG_scores, norm_QPP_scores)]
                # max_score = max(agg_scores) if max(agg_scores) > 0 else 1.0
                # norm_scores = [score / max_score for score in agg_scores]
            elif self.data_args.add_LLM_scores and self.data_args.add_QPP_scores:
                max_QPP_score = max(QPP_scores) if max(QPP_scores) > 0 else 1.0
                norm_QPP_scores = [score / max_QPP_score for score in QPP_scores]  
                agg_scores = [a + b / 2 + c / 2 for a, b, c in zip(RAG_scores, norm_QPP_scores, LLM_scores)]
                max_score = max(agg_scores) if max(agg_scores) > 0 else 1.0
                norm_scores = [score / max_score for score in agg_scores]
                #norm_scores = [a + b / 2 + c / 2 for a, b, c in zip(RAG_scores, norm_QPP_scores, LLM_scores)]
            else:
                max_score = max(RAG_scores) if max(RAG_scores) > 0 else 1.0
                norm_scores = [score / max_score for score in RAG_scores]
  
        #breakpoint()
        # add LLM scores or QPP scores
        #if self.data_args.add_LLM_scores:
        #    norm_scores = [a + b for a, b in zip(norm_scores, LLM_scores)]
        #if self.data_args.add_QPP_scores:
        #    norm_scores = [a + b for a, b in zip(norm_scores, QPP_scores)]
            
        #breakpoint()
        # mask before ground_truth
        im_start_id = self.tokenizer.convert_tokens_to_ids("<|im_start|>")
        assistant_token_id = self.tokenizer.convert_tokens_to_ids("assistant")
        im_start_positions = (input_ids == im_start_id).nonzero(as_tuple=True)[0].tolist()
        label_start = 0
        for i in reversed(im_start_positions):
            try:
                if input_ids[i+1].item() == assistant_token_id:
                    label_start = i
                    break
            except:
                label_start = input_ids[i]

        labels = input_ids.clone()
        labels[:label_start] = -100
        labels[input_ids == self.tokenizer.pad_token_id] = -100
        # self.tokenizer.decode(input_ids)

        # Relevance score assignment
        relevance_scores = torch.ones(self.max_length, dtype=torch.float)
        passage_start_indices = []
        for i in range(self.top_k):
            tok = f"Passage_{i+1}:"
            tok_id = self.tokenizer.convert_tokens_to_ids(tok)
            if tok_id == self.tokenizer.unk_token_id:
                raise ValueError(f"Token '{tok}' is not defined in tokenizer vob, please add_special_tokens")
            # find token's position in input_ids, only use the first one
            matches = (input_ids == tok_id).nonzero(as_tuple=True)[0]
            if len(matches) == 0: #or data_example["id"] == "train_96":
                #bad = 1
                break
                #raise ValueError(f"Token '{tok}' is not in the list, please check top_k")
            passage_start_indices.append(matches[0].item())

        # extract the position
        passage_spans = []
        if label_start == 0:
            label_start = 1024
        #for i in range(self.top_k):
        for i in range(len(passage_start_indices)):
            start = passage_start_indices[i]
            end = passage_start_indices[i+1] if i < len(passage_start_indices) - 1 else label_start - 1
            #end = passage_start_indices[i+1] if i < self.top_k - 1 else label_start - 1
            passage_spans.append((start, end))

        # assign relevance score
        for i, (start, end) in enumerate(passage_spans):
            #try:
            relevance_scores[start:end] = norm_scores[i]
            #except:
            #    breakpoint()
        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
            "passage_spans": passage_spans, # for pair-wise ranking loss
            "relevant_scores": relevance_scores, # for opendecoder
    }

class Train_SFT_Dataset(Dataset):
    #def __init__(self, tokenizer, NQ_data_path, NQ_RAG_data_path, RAG_text_path, max_length=1024, top_k=5):
    def __init__(self, tokenizer, data_args, max_length=1300):
        self.tokenizer = tokenizer
        self.max_length = data_args.top_k * 150 + 200
        self.data = []
        self.RAG_data = []
        self.pid2passage = {}
        self.top_k = data_args.top_k
        self.data_args = data_args
        self.NQ_data_path = data_args.NQ_data_path
        self.NQ_RAG_data_path = data_args.NQ_RAG_data_path
        self.hotpotqa_data_path = data_args.hotpotqa_data_path
        self.hotpotqa_RAG_data_path = data_args.hotpotqa_RAG_data_path
        self.RAG_text_path = data_args.RAG_text_path

        if len(self.NQ_data_path) > 0:
            with open(self.NQ_data_path, encoding="utf-8") as f:
                NQ_data = f.readlines()
                self.data.extend(NQ_data)

            with open(self.NQ_RAG_data_path, encoding="utf-8") as f:
                NQ_RAG_data = f.readlines()
                self.RAG_data.extend(NQ_RAG_data)
            print("loaded NQ data!!!!!")

        if len(self.hotpotqa_data_path) > 0:
            with open(self.hotpotqa_data_path, encoding="utf-8") as f:
                hotpotqa_data = f.readlines()
                self.data.extend(hotpotqa_data)

            with open(self.hotpotqa_RAG_data_path, encoding="utf-8") as f:
                hotpotqa_RAG_data = f.readlines()
                self.RAG_data.extend(hotpotqa_RAG_data)
            print("loaded hotpotqa data!!!")

        self.pid2passage = pload(self.RAG_text_path)

    def inject_less_relevant_psg(self, context_example, position_list, irrel_nums):
        '''
        position_list: indicate the position in top_k, used for RAG training
        irrel_nums: the number of totally irrlevant passages for RAG training
        '''
        # extract the other/indicated top-k
        full_RAG_pid = context_example["top_pid"]
        full_RAG_scores = context_example["top_pid_score"]
        irrel_RAG_pid = context_example["irrel_pid"]
        irrel_RAG_scores = context_example["irrel_pid_score"]
        add_RAG_pid, add_RAG_scores = [], []
        for pos in position_list:
            add_RAG_pid.append(full_RAG_pid[pos - 1])
            add_RAG_scores.append(full_RAG_scores[pos - 1])
        #len_collection = 21015324
        for i in range(irrel_nums):
            #irrel_int = random.randint(1, len_collection)
            add_RAG_pid.append(irrel_RAG_pid[i])
            add_RAG_scores.append(irrel_RAG_scores[i])
        return add_RAG_pid, add_RAG_scores
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data_example = json.loads(self.data[idx])
        context_example = json.loads(self.RAG_data[idx])
        if self.data_args.add_irrelevant_psg:
            RAG_pid = context_example["top_pid"][:self.top_k - 5]
            RAG_scores = context_example["top_pid_score"][:self.top_k - 5]
        else:
            RAG_pid = context_example["top_pid"][:self.top_k]
            RAG_scores = context_example["top_pid_score"][:self.top_k]

        # inject irrelevant psgs for RAG training
        if self.data_args.add_irrelevant_psg:
            position_list = [30, 50, 100] # add three psg in top-k
            irrel_nums = 2
            add_RAG_pid, add_RAG_scores = self.inject_less_relevant_psg(context_example, position_list, irrel_nums)
            RAG_pid.extend(add_RAG_pid)
            RAG_scores.extend(add_RAG_scores)

        assert len(RAG_pid) == self.top_k

        # Combine question + retrieved passages into input prompt
        context_parts = []
        for i, pid in enumerate(RAG_pid[:self.top_k]):
            passage_token = f"Passage_{i+1}:"
            context_parts.append(f"{passage_token} {self.pid2passage[pid]}")
        context = "\n".join(context_parts)
        question = data_example["question"]
        if "golden_answers" in data_example: # NQ
            answer = data_example["golden_answers"][0]
        elif "answer" in data_example: # hotpotqa
            answer = data_example["answer"]
        messages = [
            {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
            {"role": "user", "content": f"You should answer the question by referring to the knowledge provided below and integrating the usefulness of your own knowledge.\n{context}\n\nQuestion:{question}\n"},
            {"role": "assistant", "content": f"{answer}"}
        ]

        # Tokenize using Qwen tokenizer's chat template
    
        tokenized_text = self.tokenizer.apply_chat_template(
            messages, tokenize=False
        )
        tokenized = self.tokenizer(
            tokenized_text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_length
        )
        input_ids = tokenized["input_ids"].squeeze(0)
        attention_mask = tokenized["attention_mask"].squeeze(0)


        # mask before ground_truth
        im_start_id = self.tokenizer.convert_tokens_to_ids("<|im_start|>")
        assistant_token_id = self.tokenizer.convert_tokens_to_ids("assistant")
        im_start_positions = (input_ids == im_start_id).nonzero(as_tuple=True)[0].tolist()
        label_start = 0
        for i in reversed(im_start_positions):
            try:
                if input_ids[i+1].item() == assistant_token_id:
                    label_start = i
                    break
            except:
                label_start = input_ids[i]

        labels = input_ids.clone()
        labels[:label_start] = -100
        labels[input_ids == self.tokenizer.pad_token_id] = -100
        # self.tokenizer.decode(input_ids)
        #breakpoint()
        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
    }

class Train_RelDPO_Dataset(Dataset):
    #def __init__(self, tokenizer, NQ_data_path, NQ_RAG_data_path, RAG_text_path, max_length=1024, top_k=5):
    def __init__(self, tokenizer, data_args, max_length=1300):
        self.tokenizer = tokenizer
        self.max_length = data_args.top_k * 150 + 200
        self.data = []
        self.RAG_data = []
        self.pid2passage = {}
        self.top_k = data_args.top_k
        self.data_args = data_args
        self.NQ_data_path = data_args.NQ_data_path
        self.NQ_RAG_data_path = data_args.NQ_RAG_data_path
        self.hotpotqa_data_path = data_args.hotpotqa_data_path
        self.hotpotqa_RAG_data_path = data_args.hotpotqa_RAG_data_path
        self.RAG_text_path = data_args.RAG_text_path

        if len(self.NQ_data_path) > 0:
            with open(self.NQ_data_path, encoding="utf-8") as f:
                NQ_data = f.readlines()[:100]
                self.data.extend(NQ_data)

            with open(self.NQ_RAG_data_path, encoding="utf-8") as f:
                NQ_RAG_data = f.readlines()[:100]
                self.RAG_data.extend(NQ_RAG_data)
            print("loaded NQ data!!!!!")

        self.pid2passage = pload(self.RAG_text_path)
    
    def inject_less_relevant_psg(self, context_example, position_list, irrel_nums):
        '''
        position_list: indicate the position in top_k, used for RAG training
        irrel_nums: the number of totally irrlevant passages for RAG training
        '''
        # extract the other/indicated top-k
        full_RAG_pid = context_example["top_pid"]
        full_RAG_scores = context_example["top_pid_score"]
        add_RAG_pid, add_RAG_scores = [], []
        for pos in position_list:
            add_RAG_pid.append(full_RAG_pid[pos - 1])
            add_RAG_scores.append(full_RAG_scores[pos - 1])
        len_collection = 21015324
        for i in range(irrel_nums):
            irrel_int = random.randint(1, len_collection)
            add_RAG_pid.append(irrel_int)
            add_RAG_scores.append(120 - i / 10)
        return add_RAG_pid, add_RAG_scores
    
    def inject_reject_answer(self):
        sample_idx = random.randint(1, len(self.data))
        sample_answer = json.loads(self.data[sample_idx])["golden_answers"][0]
        return sample_answer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data_example = json.loads(self.data[idx])
        context_example = json.loads(self.RAG_data[idx])
        RAG_pid, RAG_scores = [], []
        if self.data_args.add_irrelevant_psg:
            RAG_pid = context_example["top_pid"][:self.top_k - 5]
            RAG_scores = context_example["top_pid_score"][:self.top_k - 5]
        else:
            RAG_pid = context_example["top_pid"][:self.top_k]
            RAG_scores = context_example["top_pid_score"][:self.top_k]
        # inject irrelevant psgs for RAG training
        if self.data_args.add_irrelevant_psg:
            position_list = [30, 50, 100] # add three psg in top-k
            irrel_nums = 2
            add_RAG_pid, add_RAG_scores = self.inject_less_relevant_psg(context_example, position_list, irrel_nums)
            RAG_pid.extend(add_RAG_pid)
            RAG_scores.extend(add_RAG_scores)

        if self.data_args.shuffle_RAG:
            RAG_paired = list(zip(RAG_pid, RAG_scores))
            random.shuffle(RAG_paired)
            shuffled_RAG_pid, shuffled_RAG_scores = zip(*RAG_paired)
            RAG_pid, RAG_scores = list(shuffled_RAG_pid), list(shuffled_RAG_scores)
        
        assert len(RAG_pid) == self.top_k
        
        # Combine question + retrieved passages into input prompt        
        context_parts = []
        for i, pid in enumerate(RAG_pid[:self.top_k]):
            passage_token = f"Passage_{i+1}:"
            context_parts.append(f"{passage_token} {self.pid2passage[pid]}")
        context = "\n".join(context_parts)
        question = data_example["question"]
        if "golden_answers" in data_example: # NQ
            answer = data_example["golden_answers"][0]
        elif "answer" in data_example: # hotpotqa
            answer = data_example["answer"]

        # construct POSITIVE sample
        messages = [
            {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
            {"role": "user", "content": f"You should answer the question by referring to the knowledge provided below and integrating your own knowledge.\n{context}\n\nQuestion:{question}\n"},
            {"role": "assistant", "content": f"{answer}"}
        ]

        # Tokenize using Qwen tokenizer's chat template
        tokenized_text = self.tokenizer.apply_chat_template(
            messages, tokenize=False
        )
        tokenized = self.tokenizer(
            tokenized_text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_length
        )
        input_ids = tokenized["input_ids"].squeeze(0)
        attention_mask = tokenized["attention_mask"].squeeze(0)

        # construct NEGATIVE sample
        negative_answer = self.inject_reject_answer()
        neg_messages = [
            {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
            {"role": "user", "content": f"You should answer the question by referring to the knowledge provided below and integrating your own knowledge.\n{context}\n\nQuestion:{question}\n"},
            {"role": "assistant", "content": f"{negative_answer}"}
        ]
        neg_tokenized_text = self.tokenizer.apply_chat_template(neg_messages, tokenize=False)
        neg_tokenized = self.tokenizer(
                    neg_tokenized_text,
                    return_tensors="pt",
                    padding="max_length",
                    truncation=True,
                    max_length=self.max_length
                )
        neg_input_ids = neg_tokenized["input_ids"].squeeze(0)
        neg_attention_mask = neg_tokenized["attention_mask"].squeeze(0)

        # normalization
        if self.data_args.normalization_type == "min-max":
            min_score, max_score = min(RAG_scores), max(RAG_scores)
            norm_scores = [(s - min_score + 0.1) / (max_score - min_score) for s in RAG_scores]
        elif self.data_args.normalization_type == "exp":
            decay_lambda = 0.5
            ranks = np.arange(len(RAG_scores))  # 0, 1, 2, ..., 9
            decay_weights = np.exp(-decay_lambda * ranks)
            norm_scores = decay_weights / decay_weights.sum()
        else:
            max_score = max(RAG_scores) if max(RAG_scores) > 0 else 1.0
            norm_scores = [score / max_score for score in RAG_scores]

        # mask before ground_truth
        im_start_id = self.tokenizer.convert_tokens_to_ids("<|im_start|>")
        assistant_token_id = self.tokenizer.convert_tokens_to_ids("assistant")
        im_start_positions = (input_ids == im_start_id).nonzero(as_tuple=True)[0].tolist()
        label_start = 0
        for i in reversed(im_start_positions):
            try:
                if input_ids[i+1].item() == assistant_token_id:
                    label_start = i
                    break
            except:
                label_start = input_ids[i]

        labels = input_ids.clone()
        labels[:label_start] = -100
        labels[input_ids == self.tokenizer.pad_token_id] = -100
        neg_labels = neg_input_ids.clone()
        neg_labels[:label_start] = -100
        neg_labels[neg_input_ids == self.tokenizer.pad_token_id] = -100
        # self.tokenizer.decode(input_ids)

        # Relevance score assignment
        relevance_scores = torch.ones(self.max_length, dtype=torch.float)
        passage_start_indices = []
        for i in range(self.top_k):
            tok = f"Passage_{i+1}:"
            tok_id = self.tokenizer.convert_tokens_to_ids(tok)
            if tok_id == self.tokenizer.unk_token_id:
                raise ValueError(f"Token '{tok}' is not defined in tokenizer vob, please add_special_tokens")
            # find token's position in input_ids, only use the first one
            matches = (input_ids == tok_id).nonzero(as_tuple=True)[0]
            if len(matches) == 0: #or data_example["id"] == "train_96":
                #bad = 1
                break
                #raise ValueError(f"Token '{tok}' is not in the list, please check top_k")
            passage_start_indices.append(matches[0].item())

        # extract the position
        passage_spans = []
        if label_start == 0:
            label_start = 1024
        #for i in range(self.top_k):
        for i in range(len(passage_start_indices)):
            start = passage_start_indices[i]
            end = passage_start_indices[i+1] if i < len(passage_start_indices) - 1 else label_start - 1
            #end = passage_start_indices[i+1] if i < self.top_k - 1 else label_start - 1
            passage_spans.append((start, end))

        # assign relevance score
        for i, (start, end) in enumerate(passage_spans):
            relevance_scores[start:end] = norm_scores[i]
        #breakpoint()
        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
            "passage_spans": passage_spans, # for pair-wise ranking loss
            "relevant_scores": relevance_scores, # for opendecoder
            "neg_input_ids": neg_input_ids,
            "neg_labels": neg_labels,
            "neg_attention_mask": neg_attention_mask,
    }

class Eval_Open_Dataset(Dataset):
    def __init__(self, tokenizer, data_args):
        self.tokenizer = tokenizer
        self.max_length = data_args.top_k * 150 + 200
        self.data = []
        self.RAG_data = []
        self.pid2passage = {}
        self.top_k = data_args.top_k
        self.data_args = data_args
        self.NQ_data_path = data_args.NQ_data_path
        self.NQ_RAG_data_path = data_args.NQ_RAG_data_path
        self.hotpotqa_data_path = data_args.hotpotqa_data_path
        self.hotpotqa_RAG_data_path = data_args.hotpotqa_RAG_data_path
        self.popqa_data_path = data_args.popqa_data_path
        self.popqa_RAG_data_path = data_args.popqa_RAG_data_path
        self.trivialqa_data_path = data_args.trivialqa_data_path
        self.trivialqa_RAG_data_path = data_args.trivialqa_RAG_data_path
        self.twiki_data_path = data_args.twiki_data_path
        self.twiki_RAG_data_path = data_args.twiki_RAG_data_path
        self.RAG_text_path = data_args.RAG_text_path

        if len(self.NQ_data_path) > 0:
            with open(self.NQ_data_path, encoding="utf-8") as f:
                NQ_data = f.readlines()
                self.data.extend(NQ_data)

            with open(self.NQ_RAG_data_path, encoding="utf-8") as f:
                NQ_RAG_data = f.readlines()
                self.RAG_data.extend(NQ_RAG_data)

        if len(self.hotpotqa_data_path) > 0:
            with open(self.hotpotqa_data_path, encoding="utf-8") as f:
                hotpotqa_data = f.readlines()
                self.data.extend(hotpotqa_data)

            with open(self.hotpotqa_RAG_data_path, encoding="utf-8") as f:
                hotpotqa_RAG_data = f.readlines()
                self.RAG_data.extend(hotpotqa_RAG_data)

        if len(self.popqa_data_path):
            with open(self.popqa_data_path, encoding="utf-8") as f:
                popqa_data = f.readlines()
                self.data.extend(popqa_data)

            with open(self.popqa_RAG_data_path, encoding="utf-8") as f:
                popqa_RAG_data = f.readlines()
                self.RAG_data.extend(popqa_RAG_data)

        if len(self.trivialqa_data_path):
            with open(self.trivialqa_data_path, encoding="utf-8") as f:
                trivialqa_data = f.readlines()
                self.data.extend(trivialqa_data)

            with open(self.trivialqa_RAG_data_path, encoding="utf-8") as f:
                trivialqa_RAG_data = f.readlines()
                self.RAG_data.extend(trivialqa_RAG_data)

        if len(self.twiki_data_path):
            with open(self.twiki_data_path, encoding="utf-8") as f:
                twiki_data = f.readlines()
                self.data.extend(twiki_data)

            with open(self.twiki_RAG_data_path, encoding="utf-8") as f:
                twiki_RAG_data = f.readlines()
                self.RAG_data.extend(twiki_RAG_data)

        self.pid2passage = pload(self.RAG_text_path)

    def inject_less_relevant_psg(self, context_example, position_list, irrel_nums):
        '''
        position_list: indicate the position in top_k, used for RAG training
        irrel_nums: the number of totally irrlevant passages for RAG training
        '''
        # extract the other/indicated top-k
        full_RAG_pid = context_example["top_pid"]
        full_RAG_scores = context_example["top_pid_score"]
        irrel_RAG_pid = context_example["irrel_pid"]
        irrel_RAG_scores = context_example["irrel_pid_score"]
        add_RAG_pid, add_RAG_scores = [], []
        for pos in position_list:
            add_RAG_pid.append(full_RAG_pid[pos - 1])
            add_RAG_scores.append(full_RAG_scores[pos - 1])
        #len_collection = 21015324
        for i in range(irrel_nums):
            #irrel_int = random.randint(1, len_collection)
            add_RAG_pid.append(irrel_RAG_pid[i])
            add_RAG_scores.append(irrel_RAG_scores[i])
        return add_RAG_pid, add_RAG_scores
    
    def inject_full_relevant_psg(self, context_example, top_k):
        # extract the other/indicated top-k
        add_RAG_pid, add_RAG_scores = [], []
        irrel_RAG_pid = context_example["irrel_pid"]
        irrel_RAG_scores = context_example["irrel_pid_score"]
        #len_collection = 21015324
        for i in range(top_k):
            #irrel_int = random.randint(1, len_collection)
            add_RAG_pid.append(irrel_RAG_pid[i])
            add_RAG_scores.append(irrel_RAG_scores[i])
        return add_RAG_pid, add_RAG_scores

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data_example = json.loads(self.data[idx])
        context_example = json.loads(self.RAG_data[idx])
        if "id" in data_example: # NQ, popqa
            qid = str(data_example["id"])
        elif "_id" in data_example: # hotpotqa
            qid = str(data_example["_id"])
        
        if self.data_args.add_irrelevant_psg:
            RAG_pid = context_example["top_pid"][:self.top_k - 5]
            RAG_scores = context_example["top_pid_score"][:self.top_k - 5]
        elif self.data_args.full_irrelevant_psg:
            RAG_pid = []
            RAG_scores = []
        else:
            RAG_pid = context_example["top_pid"][:self.top_k]
            RAG_scores = context_example["top_pid_score"][:self.top_k]

        # inject irrelevant psgs for RAG training
        if self.data_args.add_irrelevant_psg:
            position_list = [30, 50, 100] # add three psg in top-k
            irrel_nums = 2
            add_RAG_pid, add_RAG_scores = self.inject_less_relevant_psg(context_example, position_list, irrel_nums)
            RAG_pid.extend(add_RAG_pid)
            RAG_scores.extend(add_RAG_scores)

        elif self.data_args.full_irrelevant_psg:
            add_RAG_pid, add_RAG_scores = self.inject_full_relevant_psg(context_example, self.top_k)
            RAG_pid.extend(add_RAG_pid)
            RAG_scores.extend(add_RAG_scores)

        if self.data_args.shuffle_RAG:
            RAG_paired = list(zip(RAG_pid, RAG_scores))
            random.shuffle(RAG_paired)
            shuffled_RAG_pid, shuffled_RAG_scores = zip(*RAG_paired)
            RAG_pid, RAG_scores = list(shuffled_RAG_pid), list(shuffled_RAG_scores)

        assert len(RAG_pid) == self.top_k

        #RAG_pid = context_example["top_pid"][:self.top_k]
        #RAG_scores = context_example["top_pid_score"][:self.top_k]
        if "golden_answers" in data_example: # NQ
            golden_answers = data_example["golden_answers"]
        elif "answer" in data_example: # hotpotqa
            golden_answers = data_example["answer"]
        elif "answers" in data_example: # popqa
            golden_answers = data_example["answers"]

        # Combine question + retrieved passages into input prompt
        context_parts = []
        for i, pid in enumerate(RAG_pid[:self.top_k]):
            special_token = f"Passage_{i+1}:"
            context_parts.append(f"{special_token} {self.pid2passage[pid]}")
        context = "\n".join(context_parts)
        #breakpoint()
        question = data_example["question"]

        # Prepare chat format input
        messages = [
            {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
            # RAG
            {"role": "user", "content": f"You should answer the question by referring to the knowledge provided below and integrating the usefulness of your own knowledge. Just directly answer it in serveral words as a short answer without any explanation.\n{context}\n\nQuestion:{question}\n"},
            # multi-hop
            #{"role": "user", "content": f"You should answer the question by referring to the knowledge provided below and integrating the usefulness of your own knowledge. Just directly answer it in serveral words as a short answer without any explanation.\n{context}\n\nQuestion:{question}\n"},
            # direct
            #{"role": "user", "content": f"You should answer the question based on your own knowledge. Just directly answer it in serveral words as a short answer without any explanation.\nQuestion:{question}\n"},
            #{"role": "assistant", "content": "The answer is "}
        ]

        # Tokenize using Qwen tokenizer's chat template      
        tokenized_text = self.tokenizer.apply_chat_template(
            messages, tokenize=False
        )
        tokenized = self.tokenizer(
            tokenized_text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_length
        )
        #breakpoint()
        #tokenized = self.tokenizer(tokenized_text, max_length=self.max_length, return_tensors="pt")
        #breakpoint()
        input_ids = tokenized["input_ids"].squeeze(0)
        attention_mask = tokenized["attention_mask"].squeeze(0)
        # Build relevance score tensor initialized to 0
        # normalize relevant scores
        norm_scores = []
        #breakpoint()
        if self.data_args.normalization_type == "min-max":
            min_score, max_score = min(RAG_scores), max(RAG_scores)
            norm_scores = [(s - min_score + 0.1) / (max_score - min_score) for s in RAG_scores]
        elif self.data_args.normalization_type == "exp":
            decay_lambda = 0.5
            ranks = np.arange(len(RAG_scores))  # 0, 1, 2, ..., 9
            decay_weights = np.exp(-decay_lambda * ranks)
            norm_scores = decay_weights / decay_weights.sum()
        else:
            max_score = max(RAG_scores) if max(RAG_scores) > 0 else 1.0
            norm_scores = [score / max_score for score in RAG_scores]


        # mask before ground_truth
        im_start_id = self.tokenizer.convert_tokens_to_ids("<|im_start|>")
        assistant_token_id = self.tokenizer.convert_tokens_to_ids("assistant")
        im_start_positions = (input_ids == im_start_id).nonzero(as_tuple=True)[0].tolist()
        label_start = 0
        for i in reversed(im_start_positions):
            try:
                if input_ids[i+1].item() == assistant_token_id:
                    label_start = i
                    break
            except:
                label_start = input_ids[i]

        labels = input_ids.clone()
        labels[:label_start] = -100
        labels[input_ids == self.tokenizer.pad_token_id] = -100
        # self.tokenizer.decode(input_ids)

        # Relevance score assignment
        relevance_scores = torch.ones(self.max_length, dtype=torch.float)
        passage_start_indices = []
        for i in range(self.top_k):
            tok = f"Passage_{i+1}:"
            tok_id = self.tokenizer.convert_tokens_to_ids(tok)
            if tok_id == self.tokenizer.unk_token_id:
                raise ValueError(f"Token '{tok}' is not defined in tokenizer vob, please add_special_tokens")
            # find token's position in input_ids, only use the first one
            matches = (input_ids == tok_id).nonzero(as_tuple=True)[0]
            if len(matches) == 0: #or data_example["id"] == "train_96":
                #bad = 1
                break
                #raise ValueError(f"Token '{tok}' is not in the list, please check top_k")
            passage_start_indices.append(matches[0].item())

        # extract the position
        passage_spans = []
        if label_start == 0:
            label_start = 1024
        #for i in range(self.top_k):
        for i in range(len(passage_start_indices)):
            start = passage_start_indices[i]
            end = passage_start_indices[i+1] if i < len(passage_start_indices) - 1 else label_start - 1
            #end = passage_start_indices[i+1] if i < self.top_k - 1 else label_start - 1
            passage_spans.append((start, end))

        # assign relevance score
        for i, (start, end) in enumerate(passage_spans):
            relevance_scores[start:end] = norm_scores[i]
        #breakpoint()
        return {
            "qid": qid,
            "question": question,
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
            "passage_spans": passage_spans, # for pair-wise ranking loss
            "relevant_scores": relevance_scores, # for opendecoder
            "gold_answers": golden_answers
    }

    
class Eval_SFT_Dataset(Dataset):
    def __init__(self, tokenizer, data_args, max_length=1024, top_k=5):
        self.tokenizer = tokenizer
        self.max_length = data_args.top_k * 150 + 200
        self.data = []
        self.RAG_data = []
        self.pid2passage = {}
        self.top_k = data_args.top_k
        self.data_args = data_args
        self.NQ_data_path = data_args.NQ_data_path
        self.NQ_RAG_data_path = data_args.NQ_RAG_data_path
        self.hotpotqa_data_path = data_args.hotpotqa_data_path
        self.hotpotqa_RAG_data_path = data_args.hotpotqa_RAG_data_path
        self.popqa_data_path = data_args.popqa_data_path
        self.popqa_RAG_data_path = data_args.popqa_RAG_data_path
        self.trivialqa_data_path = data_args.trivialqa_data_path
        self.trivialqa_RAG_data_path = data_args.trivialqa_RAG_data_path
        self.twiki_data_path = data_args.twiki_data_path
        self.twiki_RAG_data_path = data_args.twiki_RAG_data_path
        self.RAG_text_path = data_args.RAG_text_path

        if len(self.NQ_data_path) > 0:
            with open(self.NQ_data_path, encoding="utf-8") as f:
                NQ_data = f.readlines()
                self.data.extend(NQ_data)

            with open(self.NQ_RAG_data_path, encoding="utf-8") as f:
                NQ_RAG_data = f.readlines()
                self.RAG_data.extend(NQ_RAG_data)

        if len(self.hotpotqa_data_path) > 0:
            with open(self.hotpotqa_data_path, encoding="utf-8") as f:
                hotpotqa_data = f.readlines()
                self.data.extend(hotpotqa_data)

            with open(self.hotpotqa_RAG_data_path, encoding="utf-8") as f:
                hotpotqa_RAG_data = f.readlines()
                self.RAG_data.extend(hotpotqa_RAG_data)

        if len(self.popqa_data_path):
            with open(self.popqa_data_path, encoding="utf-8") as f:
                popqa_data = f.readlines()
                self.data.extend(popqa_data)

            with open(self.popqa_RAG_data_path, encoding="utf-8") as f:
                popqa_RAG_data = f.readlines()
                self.RAG_data.extend(popqa_RAG_data)

        if len(self.trivialqa_data_path):
            with open(self.trivialqa_data_path, encoding="utf-8") as f:
                trivialqa_data = f.readlines()
                self.data.extend(trivialqa_data)

            with open(self.trivialqa_RAG_data_path, encoding="utf-8") as f:
                trivialqa_RAG_data = f.readlines()
                self.RAG_data.extend(trivialqa_RAG_data)

        if len(self.twiki_data_path):
            with open(self.twiki_data_path, encoding="utf-8") as f:
                twiki_data = f.readlines()
                self.data.extend(twiki_data)

            with open(self.twiki_RAG_data_path, encoding="utf-8") as f:
                twiki_RAG_data = f.readlines()
                self.RAG_data.extend(twiki_RAG_data)

        self.pid2passage = pload(self.RAG_text_path)

    def inject_less_relevant_psg(self, context_example, position_list, irrel_nums):
        '''
        position_list: indicate the position in top_k, used for RAG training
        irrel_nums: the number of totally irrlevant passages for RAG training
        '''
        # extract the other/indicated top-k
        full_RAG_pid = context_example["top_pid"]
        full_RAG_scores = context_example["top_pid_score"]
        irrel_RAG_pid = context_example["irrel_pid"]
        irrel_RAG_scores = context_example["irrel_pid_score"]
        add_RAG_pid, add_RAG_scores = [], []
        #for pos in position_list:
        #    add_RAG_pid.append(full_RAG_pid[pos - 1])
        #    add_RAG_scores.append(full_RAG_scores[pos - 1])
        #len_collection = 21015324
        for i in range(irrel_nums):
            #irrel_int = random.randint(1, len_collection)
            add_RAG_pid.append(irrel_RAG_pid[i])
            add_RAG_scores.append(irrel_RAG_scores[i])
        return add_RAG_pid, add_RAG_scores
    
    def inject_full_relevant_psg(self, context_example, top_k):
        # extract the other/indicated top-k
        add_RAG_pid, add_RAG_scores = [], []
        irrel_RAG_pid = context_example["irrel_pid"]
        irrel_RAG_scores = context_example["irrel_pid_score"]
        #len_collection = 21015324
        for i in range(top_k):
            #irrel_int = random.randint(1, len_collection)
            add_RAG_pid.append(irrel_RAG_pid[i])
            add_RAG_scores.append(irrel_RAG_scores[i])
        return add_RAG_pid, add_RAG_scores

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data_example = json.loads(self.data[idx])
        context_example = json.loads(self.RAG_data[idx])
        if "id" in data_example: # NQ, popqa
            qid = str(data_example["id"])
        elif "_id" in data_example: # hotpotqa
            qid = str(data_example["_id"])

        if self.data_args.add_irrelevant_psg:
            RAG_pid = context_example["top_pid"][:self.top_k - 5]
            RAG_scores = context_example["top_pid_score"][:self.top_k - 5]
        elif self.data_args.full_irrelevant_psg:
            RAG_pid = []
            RAG_scores = []
        else:
            RAG_pid = context_example["top_pid"][:self.top_k]
            RAG_scores = context_example["top_pid_score"][:self.top_k]

        # inject irrelevant psgs for RAG training
        if self.data_args.add_irrelevant_psg:
            position_list = [30, 50, 100] # add three psg in top-k
            irrel_nums = 5
            add_RAG_pid, add_RAG_scores = self.inject_less_relevant_psg(context_example, position_list, irrel_nums)
            RAG_pid.extend(add_RAG_pid)
            RAG_scores.extend(add_RAG_scores)

        elif self.data_args.full_irrelevant_psg:
            add_RAG_pid, add_RAG_scores = self.inject_full_relevant_psg(context_example, self.top_k)
            RAG_pid.extend(add_RAG_pid)
            RAG_scores.extend(add_RAG_scores)

        assert len(RAG_pid) == self.top_k

        #RAG_pid = context_example["top_pid"][:self.top_k]
        #RAG_scores = context_example["top_pid_score"][:self.top_k]
        if "golden_answers" in data_example: # NQ
            golden_answers = data_example["golden_answers"]
        elif "answer" in data_example: # hotpotqa
            golden_answers = data_example["answer"]
        elif "answers" in data_example: # popqa
            golden_answers = data_example["answers"]

        # Combine question + retrieved passages into input prompt
        context_parts = []
        for i, pid in enumerate(RAG_pid[:self.top_k]):
            special_token = f"Passage_{i+1}:"
            context_parts.append(f"{special_token} {self.pid2passage[pid]}")
        context = "\n".join(context_parts)
        #breakpoint()
        question = data_example["question"]

        # Prepare chat format input
        messages = [
            {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
            # RAG
            {"role": "user", "content": f"You should answer the question by referring to the knowledge provided below and integrating the usefulness of your own knowledge. Just directly answer it in serveral words as a short answer without any explanation.\n{context}\n\nQuestion:{question}\n"},
            # multi-hop
            #{"role": "user", "content": f"You should answer the question by referring to the knowledge provided below and integrating the usefulness of your own knowledge. Answer it in less than ten words as a short answer without any explanation.\n{context}\n\nQuestion:{question}\n"},
            # direct
            #{"role": "user", "content": f"You should answer the question based on your own knowledge. Just directly answer it in serveral words as a short answer without any explanation.\nQuestion:{question}\n"},
        ]
  
        tokenized_text = self.tokenizer.apply_chat_template(
            messages, tokenize=False
        )
        tokenized = self.tokenizer(
            tokenized_text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_length
        )
        input_ids = tokenized["input_ids"].squeeze(0)
        attention_mask = tokenized["attention_mask"].squeeze(0)
        # Build relevance score tensor initialized to 0
 
        #breakpoint()
        labels = input_ids.clone()
        labels[labels == self.tokenizer.pad_token_id] = -100
        return {
            "qid": qid,
            "question": question,
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
            "gold_answers": golden_answers
        }

class Eval_MultiOpen_Dataset(Dataset):
    #def __init__(self, tokenizer, NQ_data_path, NQ_RAG_data_path, RAG_text_path, max_length=1024, top_k=5):
    def __init__(self, tokenizer, data_args):
        self.tokenizer = tokenizer
        self.max_length = data_args.top_k * 150 + 200
        self.data, self.RAG_data = [], []
        self.LLM_score_data, self.QPP_score_data = [], []
        self.pid2passage = {}
        self.top_k = data_args.top_k
        self.data_args = data_args
        self.NQ_data_path = data_args.NQ_data_path
        self.NQ_RAG_data_path = data_args.NQ_RAG_data_path
        self.NQ_LLM_score_path = data_args.NQ_LLM_score_path
        self.NQ_QPP_score_path = data_args.NQ_QPP_score_path
        self.hotpotqa_data_path = data_args.hotpotqa_data_path
        self.hotpotqa_RAG_data_path = data_args.hotpotqa_RAG_data_path
        self.hotpotqa_LLM_score_path = data_args.hotpotqa_LLM_score_path
        self.hotpotqa_QPP_score_path = data_args.hotpotqa_QPP_score_path
        self.popqa_data_path = data_args.popqa_data_path
        self.popqa_RAG_data_path = data_args.popqa_RAG_data_path
        self.popqa_LLM_score_path = data_args.popqa_LLM_score_path
        self.popqa_QPP_score_path = data_args.popqa_QPP_score_path
        self.trivialqa_data_path = data_args.trivialqa_data_path
        self.trivialqa_RAG_data_path = data_args.trivialqa_RAG_data_path
        self.trivialqa_LLM_score_path = data_args.trivialqa_LLM_score_path
        self.trivialqa_QPP_score_path = data_args.trivialqa_QPP_score_path
        self.twiki_data_path = data_args.twiki_data_path
        self.twiki_RAG_data_path = data_args.twiki_RAG_data_path
        self.twiki_LLM_score_path = data_args.twiki_LLM_score_path
        self.twiki_QPP_score_path = data_args.twiki_QPP_score_path
        self.RAG_text_path = data_args.RAG_text_path

        with open(self.NQ_data_path, encoding="utf-8") as f:
            NQ_data = f.readlines()
            self.data.extend(NQ_data)
        with open(self.NQ_RAG_data_path, encoding="utf-8") as f:
            NQ_RAG_data = f.readlines()
            self.RAG_data.extend(NQ_RAG_data)
        if self.data_args.add_LLM_scores:
            with open(self.NQ_LLM_score_path, encoding="utf-8") as f:
                NQ_LLM_score_data = f.readlines()
                self.LLM_score_data.extend(NQ_LLM_score_data)
        if self.data_args.add_QPP_scores:
            with open(self.NQ_QPP_score_path, encoding="utf-8") as f:
                NQ_QPP_score_data = f.readlines()
                self.QPP_score_data.extend(NQ_QPP_score_data)

        with open(self.hotpotqa_data_path, encoding="utf-8") as f:
            hotpotqa_data = f.readlines()
            self.data.extend(hotpotqa_data)
        with open(self.hotpotqa_RAG_data_path, encoding="utf-8") as f:
            hotpotqa_RAG_data = f.readlines()
            self.RAG_data.extend(hotpotqa_RAG_data)
        if self.data_args.add_LLM_scores:
            with open(self.hotpotqa_LLM_score_path, encoding="utf-8") as f:
                hotpotqa_LLM_score_data = f.readlines()
                self.LLM_score_data.extend(hotpotqa_LLM_score_data)
        if self.data_args.add_QPP_scores:
            with open(self.hotpotqa_QPP_score_path, encoding="utf-8") as f:
                hotpotqa_QPP_score_data = f.readlines()
                self.QPP_score_data.extend(hotpotqa_QPP_score_data)

        with open(self.popqa_data_path, encoding="utf-8") as f:
            popqa_data = f.readlines()
            self.data.extend(popqa_data)
        with open(self.popqa_RAG_data_path, encoding="utf-8") as f:
            popqa_RAG_data = f.readlines()
            self.RAG_data.extend(popqa_RAG_data)
        if self.data_args.add_LLM_scores:
            with open(self.popqa_LLM_score_path, encoding="utf-8") as f:
                popqa_LLM_score_data = f.readlines()
                self.LLM_score_data.extend(popqa_LLM_score_data)
        if self.data_args.add_QPP_scores:
            with open(self.popqa_QPP_score_path, encoding="utf-8") as f:
                popqa_QPP_score_data = f.readlines()
                self.QPP_score_data.extend(popqa_QPP_score_data)

        with open(self.trivialqa_data_path, encoding="utf-8") as f:
            trivialqa_data = f.readlines()
            self.data.extend(trivialqa_data)
        with open(self.trivialqa_RAG_data_path, encoding="utf-8") as f:
            trivialqa_RAG_data = f.readlines()
            self.RAG_data.extend(trivialqa_RAG_data)
        if self.data_args.add_LLM_scores:
            with open(self.trivialqa_LLM_score_path, encoding="utf-8") as f:
                trivialqa_LLM_score_data = f.readlines()
                self.LLM_score_data.extend(trivialqa_LLM_score_data)
        if self.data_args.add_QPP_scores:
            with open(self.trivialqa_QPP_score_path, encoding="utf-8") as f:
                trivialqa_QPP_score_data = f.readlines()
                self.QPP_score_data.extend(trivialqa_QPP_score_data)

        with open(self.twiki_data_path, encoding="utf-8") as f:
            twiki_data = f.readlines()
            self.data.extend(twiki_data)
        with open(self.twiki_RAG_data_path, encoding="utf-8") as f:
            twiki_RAG_data = f.readlines()
            self.RAG_data.extend(twiki_RAG_data)
        if self.data_args.add_LLM_scores:
            with open(self.twiki_LLM_score_path, encoding="utf-8") as f:
                twiki_LLM_score_data = f.readlines()
                self.LLM_score_data.extend(twiki_LLM_score_data)
        if self.data_args.add_QPP_scores:
            with open(self.twiki_QPP_score_path, encoding="utf-8") as f:
                twiki_QPP_score_data = f.readlines()
                self.QPP_score_data.extend(twiki_QPP_score_data)

        self.pid2passage = pload(self.RAG_text_path)

    def inject_less_relevant_psg(self, context_example, position_list, irrel_nums):
        '''
        position_list: indicate the position in top_k, used for RAG training
        irrel_nums: the number of totally irrlevant passages for RAG training
        '''
        # extract the other/indicated top-k
        full_RAG_pid = context_example["top_pid"]
        full_RAG_scores = context_example["top_pid_score"]
        irrel_RAG_pid = context_example["irrel_pid"]
        irrel_RAG_scores = context_example["irrel_pid_score"]
        add_RAG_pid, add_RAG_scores = [], []
        for pos in position_list:
            add_RAG_pid.append(full_RAG_pid[pos - 1])
            add_RAG_scores.append(full_RAG_scores[pos - 1])
        #len_collection = 21015324
        for i in range(irrel_nums):
            #irrel_int = random.randint(1, len_collection)
            add_RAG_pid.append(irrel_RAG_pid[i])
            add_RAG_scores.append(irrel_RAG_scores[i])
        return add_RAG_pid, add_RAG_scores
    
    def inject_full_relevant_psg(self, context_example, top_k):
        # extract the other/indicated top-k
        add_RAG_pid, add_RAG_scores = [], []
        irrel_RAG_pid = context_example["irrel_pid"]
        irrel_RAG_scores = context_example["irrel_pid_score"]
        #len_collection = 21015324
        for i in range(top_k):
            #irrel_int = random.randint(1, len_collection)
            add_RAG_pid.append(irrel_RAG_pid[i])
            add_RAG_scores.append(irrel_RAG_scores[i])
        return add_RAG_pid, add_RAG_scores

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data_example = json.loads(self.data[idx])
        context_example = json.loads(self.RAG_data[idx])
        if len(self.LLM_score_data) > 0 and self.data_args.add_LLM_scores:
            LLM_score_example = json.loads(self.LLM_score_data[idx])
        if len(self.QPP_score_data) > 0 and self.data_args.add_QPP_scores:
            QPP_score_example = json.loads(self.QPP_score_data[idx])
        if "id" in data_example: # NQ, popqa
            qid = str(data_example["id"])
        elif "_id" in data_example: # hotpotqa
            qid = str(data_example["_id"])
        
        if self.data_args.add_irrelevant_psg:
            RAG_pid = context_example["top_pid"][:self.top_k - 5]
            RAG_scores = context_example["top_pid_score"][:self.top_k - 5]
        elif self.data_args.full_irrelevant_psg:
            RAG_pid = []
            RAG_scores = []
        else:
            RAG_pid = context_example["top_pid"][:self.top_k]
            RAG_scores = context_example["top_pid_score"][:self.top_k]

        LLM_scores, QPP_scores = [], []


        # inject irrelevant psgs for RAG training
        if self.data_args.add_irrelevant_psg:
            position_list = [30, 50, 100] # add three psg in top-k
            irrel_nums = 2
            add_RAG_pid, add_RAG_scores = self.inject_less_relevant_psg(context_example, position_list, irrel_nums)
            RAG_pid.extend(add_RAG_pid)
            RAG_scores.extend(add_RAG_scores)
            if self.data_args.add_LLM_scores:
                if len(LLM_score_example["doc_ids"]) == 10:
                    assert LLM_score_example["doc_ids"] == RAG_pid
                    LLM_scores.extend(LLM_score_example["LLM_scores"])
                elif len(LLM_score_example["doc_ids"]) > 10:
                    LLM_docids = LLM_score_example["doc_ids"][:5] + LLM_score_example["doc_ids"][10:15]
                    assert LLM_docids == RAG_pid
                    LLM_scores.extend(LLM_score_example["LLM_scores"][:5]+ LLM_score_example["LLM_scores"][10:15])
            if self.data_args.add_QPP_scores:
                if len(QPP_score_example["doc_ids"]) == 10:
                    assert QPP_score_example["doc_ids"] == RAG_pid
                    QPP_scores.extend(QPP_score_example["QPP_scores"])
                elif len(QPP_score_example["doc_ids"]) > 10:
                    QPP_docids = QPP_score_example["doc_ids"][:5] + QPP_score_example["doc_ids"][10:15]
                    assert QPP_docids == RAG_pid
                    QPP_scores.extend(QPP_score_example["QPP_scores"][:5]+ QPP_score_example["QPP_scores"][10:15])
        elif self.data_args.full_irrelevant_psg:
            add_RAG_pid, add_RAG_scores = self.inject_full_relevant_psg(context_example, self.top_k)
            RAG_pid.extend(add_RAG_pid)
            RAG_scores.extend(add_RAG_scores)
            if self.data_args.add_LLM_scores:
                if len(LLM_score_example["doc_ids"]) == 10:
                    assert LLM_score_example["doc_ids"] == RAG_pid
                    LLM_scores.extend(LLM_score_example["LLM_scores"])
                elif len(LLM_score_example["doc_ids"]) > 10:
                    LLM_docids = LLM_score_example["doc_ids"][13:]
                    assert LLM_docids == RAG_pid
                    LLM_scores.extend(LLM_score_example["LLM_scores"][13:])
            if self.data_args.add_QPP_scores:
                if len(QPP_score_example["doc_ids"]) == 10:
                    assert QPP_score_example["doc_ids"] == RAG_pid
                    QPP_scores.extend(QPP_score_example["QPP_scores"])
                elif len(QPP_score_example["doc_ids"]) > 10:
                    QPP_docids = QPP_score_example["doc_ids"][13:]
                    assert QPP_docids == RAG_pid
                    QPP_scores.extend(QPP_score_example["QPP_scores"][13:])
        else:
            if self.data_args.add_LLM_scores:
                if len(LLM_score_example["doc_ids"]) == 10:
                    assert LLM_score_example["doc_ids"] == RAG_pid
                    LLM_scores.extend(LLM_score_example["LLM_scores"])
                elif len(LLM_score_example["doc_ids"]) > 10:
                    LLM_docids = LLM_score_example["doc_ids"][:10]
                    assert LLM_docids == RAG_pid
                    LLM_scores.extend(LLM_score_example["LLM_scores"][:10])
            if self.data_args.add_QPP_scores:
                if len(QPP_score_example["doc_ids"]) == 10:
                    assert QPP_score_example["doc_ids"] == RAG_pid
                    QPP_scores.extend(QPP_score_example["QPP_scores"])
                elif len(QPP_score_example["doc_ids"]) > 10:
                    QPP_docids = QPP_score_example["doc_ids"][:10]
                    assert QPP_docids == RAG_pid
                    QPP_scores.extend(QPP_score_example["QPP_scores"][:10])
            

        if self.data_args.shuffle_RAG:
            RAG_paired = list(zip(RAG_pid, RAG_scores))
            random.shuffle(RAG_paired)
            shuffled_RAG_pid, shuffled_RAG_scores = zip(*RAG_paired)
            RAG_pid, RAG_scores = list(shuffled_RAG_pid), list(shuffled_RAG_scores)

        assert len(RAG_pid) == self.top_k

        #RAG_pid = context_example["top_pid"][:self.top_k]
        #RAG_scores = context_example["top_pid_score"][:self.top_k]
        if "golden_answers" in data_example: # NQ
            golden_answers = data_example["golden_answers"]
        elif "answer" in data_example: # hotpotqa
            golden_answers = data_example["answer"]
        elif "answers" in data_example: # popqa
            golden_answers = data_example["answers"]

        # Combine question + retrieved passages into input prompt
        context_parts = []
        for i, pid in enumerate(RAG_pid[:self.top_k]):
            special_token = f"Passage_{i+1}:"
            context_parts.append(f"{special_token} {self.pid2passage[pid]}")
        context = "\n".join(context_parts)
        #breakpoint()
        question = data_example["question"]

        # Prepare chat format input
        messages = [
            {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
            # RAG
            {"role": "user", "content": f"You should answer the question by referring to the knowledge provided below and integrating the usefulness of your own knowledge. Just directly answer it in serveral words as a short answer without any explanation.\n{context}\n\nQuestion:{question}\n"},
            # multi-hop
            #{"role": "user", "content": f"You should answer the question by referring to the knowledge provided below and integrating the usefulness of your own knowledge. Just directly answer it in serveral words as a short answer without any explanation.\n{context}\n\nQuestion:{question}\n"},
            # direct
            #{"role": "user", "content": f"You should answer the question based on your own knowledge. Just directly answer it in serveral words as a short answer without any explanation.\nQuestion:{question}\n"},
            #{"role": "assistant", "content": "The answer is "}
        ]

        # Tokenize using Qwen tokenizer's chat template      
        tokenized_text = self.tokenizer.apply_chat_template(
            messages, tokenize=False
        )
        tokenized = self.tokenizer(
            tokenized_text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_length
        )

        input_ids = tokenized["input_ids"].squeeze(0)
        attention_mask = tokenized["attention_mask"].squeeze(0)
        # Build relevance score tensor initialized to 0
        # normalize relevant scores
        norm_scores = []
        #breakpoint()
        if self.data_args.normalization_type == "min-max":
            min_score, max_score = min(RAG_scores), max(RAG_scores)
            norm_scores = [(s - min_score + 0.1) / (max_score - min_score) for s in RAG_scores]
        elif self.data_args.normalization_type == "exp":
            decay_lambda = 0.5
            ranks = np.arange(len(RAG_scores))  # 0, 1, 2, ..., 9
            decay_weights = np.exp(-decay_lambda * ranks)
            norm_scores = decay_weights / decay_weights.sum()
        else:
            if self.data_args.add_LLM_scores and not self.data_args.add_QPP_scores:
                # old
                agg_scores = [a + b / 2 for a, b in zip(RAG_scores, LLM_scores)]
                max_score = max(agg_scores) if max(agg_scores) > 0 else 1.0
                norm_scores = [score / max_score for score in agg_scores]
                # norm_scores = LLM_scores
                # new
                #max_LLM_score = max(LLM_scores) if max(LLM_scores) > 0 else 1.0
                #norm_LLM_scores = [score / max_LLM_score for score in LLM_scores]
                # min_val, max_val = min(LLM_scores), max(LLM_scores)
                # norm_LLM_scores = [(x - min_val) / (max_val - min_val) for x in LLM_scores]
                # norm_scores = norm_LLM_scores
                # agg_scores = [a + b / 2 for a, b in zip(RAG_scores, norm_LLM_scores)]
                # max_score = max(agg_scores) if max(agg_scores) > 0 else 1.0
                # norm_scores = [score / max_score for score in agg_scores]
                #breakpoint()
            elif self.data_args.add_QPP_scores and not self.data_args.add_LLM_scores:
                max_QPP_score = max(QPP_scores) if max(QPP_scores) > 0 else 1.0
                norm_QPP_scores = [score / max_QPP_score for score in QPP_scores]
                norm_scores = norm_QPP_scores
                # agg_scores = [a + b / 2 for a, b in zip(RAG_scores, norm_QPP_scores)]
                # max_score = max(agg_scores) if max(agg_scores) > 0 else 1.0
                # norm_scores = [score / max_score for score in agg_scores]
            elif self.data_args.add_LLM_scores and self.data_args.add_QPP_scores:
                max_QPP_score = max(QPP_scores) if max(QPP_scores) > 0 else 1.0
                norm_QPP_scores = [score / max_QPP_score for score in QPP_scores]  
                max_LLM_score = max(LLM_scores) if max(LLM_scores) > 0 else 1.0
                norm_LLM_scores = [score / max_LLM_score for score in LLM_scores]
                agg_scores = [a + b / 2 + c / 2 for a, b, c in zip(RAG_scores, norm_QPP_scores, norm_LLM_scores)]
                max_score = max(agg_scores) if max(agg_scores) > 0 else 1.0
                norm_scores = [score / max_score for score in agg_scores]
                #norm_scores = [a + b / 2 + c / 2 for a, b, c in zip(RAG_scores, norm_QPP_scores, LLM_scores)]
            else:
                max_score = max(RAG_scores) if max(RAG_scores) > 0 else 1.0
                norm_scores = [score / max_score for score in RAG_scores]


        # mask before ground_truth
        im_start_id = self.tokenizer.convert_tokens_to_ids("<|im_start|>")
        assistant_token_id = self.tokenizer.convert_tokens_to_ids("assistant")
        im_start_positions = (input_ids == im_start_id).nonzero(as_tuple=True)[0].tolist()
        label_start = 0
        for i in reversed(im_start_positions):
            try:
                if input_ids[i+1].item() == assistant_token_id:
                    label_start = i
                    break
            except:
                label_start = input_ids[i]

        labels = input_ids.clone()
        labels[:label_start] = -100
        labels[input_ids == self.tokenizer.pad_token_id] = -100
        # self.tokenizer.decode(input_ids)

        # Relevance score assignment
        relevance_scores = torch.ones(self.max_length, dtype=torch.float)
        passage_start_indices = []
        for i in range(self.top_k):
            tok = f"Passage_{i+1}:"
            tok_id = self.tokenizer.convert_tokens_to_ids(tok)
            if tok_id == self.tokenizer.unk_token_id:
                raise ValueError(f"Token '{tok}' is not defined in tokenizer vob, please add_special_tokens")
            # find token's position in input_ids, only use the first one
            matches = (input_ids == tok_id).nonzero(as_tuple=True)[0]
            if len(matches) == 0: #or data_example["id"] == "train_96":
                #bad = 1
                break
                #raise ValueError(f"Token '{tok}' is not in the list, please check top_k")
            passage_start_indices.append(matches[0].item())

        # extract the position
        passage_spans = []
        if label_start == 0:
            label_start = 1024
        #for i in range(self.top_k):
        for i in range(len(passage_start_indices)):
            start = passage_start_indices[i]
            end = passage_start_indices[i+1] if i < len(passage_start_indices) - 1 else label_start - 1
            #end = passage_start_indices[i+1] if i < self.top_k - 1 else label_start - 1
            passage_spans.append((start, end))

        # assign relevance score
        for i, (start, end) in enumerate(passage_spans):
            relevance_scores[start:end] = norm_scores[i]
        #breakpoint()
        return {
            "qid": qid,
            "question": question,
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
            "passage_spans": passage_spans, # for pair-wise ranking loss
            "relevant_scores": relevance_scores, # for opendecoder
            "gold_answers": golden_answers
    }
