import sys
sys.path.append('..')
sys.path.append('.')
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


import json
import random
from tqdm import trange, tqdm
#import pandas as pd

import torch
from torch.utils.data import Dataset

class NQ_retrieval(Dataset):
    def __init__(self, args, tokenizer, filename):
        self.examples = []
        
        with open(filename, encoding="utf-8") as f:
            data = f.readlines()

        n = len(data)

        for i in tqdm(trange(n)):
            record = json.loads(data[i])
            qid = record['id']
            question = record["question"]
            answer = record["golden_answers"][0]
            question = tokenizer.encode(question, add_special_tokens = True, max_length = args.max_query_length)
            answer = tokenizer.encode(answer, add_special_tokens = True, max_length = args.max_answer_length)
            question, question_mask = padding_seq_to_same_length(question, max_pad_length = args.max_query_length)
            answer, answer_mask = padding_seq_to_same_length(answer, max_pad_length = args.max_answer_length)
            self.examples.append([qid, question, question_mask, answer, answer_mask])
    
    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        return self.examples[item]

    @staticmethod
    def get_collate_fn(args):
        
        def collate_fn(batch: list):
            collated_dict = {"bt_qid": [],
                             "bt_question": [],
                             "bt_question_mask": [],
                             "bt_answer":[],
                             "bt_answer_mask":[],
                            }
            for example in batch:
                collated_dict["bt_qid"].append(example[0])
                collated_dict["bt_question"].append(example[1])
                collated_dict["bt_question_mask"].append(example[2])
                collated_dict["bt_answer"].append(example[3])
                collated_dict["bt_answer_mask"].append(example[4])
                

            not_need_to_tensor_keys = set(["bt_qid"])

            for key in collated_dict:
                if key not in not_need_to_tensor_keys:
                    collated_dict[key] = torch.tensor(collated_dict[key], dtype=torch.long)
            return collated_dict

        return collate_fn

class hotpotqa_retrieval(Dataset):
    def __init__(self, args, tokenizer, filename):
        self.examples = []

        with open(filename, encoding="utf-8") as f:
            data = json.load(f)

        for did, data in enumerate(data):
            qid = data['_id']
            question = data["question"]
            breakpoint()
            answer = data["answer"]
            question = tokenizer.encode(question, add_special_tokens = True, max_length = args.max_query_length)
            answer = tokenizer.encode(answer, add_special_tokens = True, max_length = args.max_answer_length)
            question, question_mask = padding_seq_to_same_length(question, max_pad_length = args.max_query_length)
            answer, answer_mask = padding_seq_to_same_length(answer, max_pad_length = args.max_answer_length)
            self.examples.append([qid, question, question_mask, answer, answer_mask])
    
    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        return self.examples[item]

    @staticmethod
    def get_collate_fn(args):
        
        def collate_fn(batch: list):
            collated_dict = {"bt_qid": [],
                             "bt_question": [],
                             "bt_question_mask": [],
                             "bt_answer":[],
                             "bt_answer_mask":[],
                            }
            for example in batch:
                collated_dict["bt_qid"].append(example[0])
                collated_dict["bt_question"].append(example[1])
                collated_dict["bt_question_mask"].append(example[2])
                collated_dict["bt_answer"].append(example[3])
                collated_dict["bt_answer_mask"].append(example[4])
                

            not_need_to_tensor_keys = set(["bt_qid"])

            for key in collated_dict:
                if key not in not_need_to_tensor_keys:
                    collated_dict[key] = torch.tensor(collated_dict[key], dtype=torch.long)
            return collated_dict

        return collate_fn

class popqa_retrieval(Dataset):
    def __init__(self, args, tokenizer, filename):
        self.examples = []
        with open(filename, encoding="utf-8") as f:
            data = f.readlines()

        n = len(data)

        for i in tqdm(trange(n)):
            record = json.loads(data[i])
            qid = record['id']
            question = record["question"]
            answer = record["answers"][0]
            question = tokenizer.encode(question, add_special_tokens = True, max_length = args.max_query_length)
            answer = tokenizer.encode(answer, add_special_tokens = True, max_length = args.max_answer_length)
            question, question_mask = padding_seq_to_same_length(question, max_pad_length = args.max_query_length)
            answer, answer_mask = padding_seq_to_same_length(answer, max_pad_length = args.max_answer_length)
            self.examples.append([qid, question, question_mask, answer, answer_mask])
    
    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        return self.examples[item]

    @staticmethod
    def get_collate_fn(args):
        
        def collate_fn(batch: list):
            collated_dict = {"bt_qid": [],
                             "bt_question": [],
                             "bt_question_mask": [],
                             "bt_answer":[],
                             "bt_answer_mask":[],
                            }
            for example in batch:
                collated_dict["bt_qid"].append(example[0])
                collated_dict["bt_question"].append(example[1])
                collated_dict["bt_question_mask"].append(example[2])
                collated_dict["bt_answer"].append(example[3])
                collated_dict["bt_answer_mask"].append(example[4])
                

            not_need_to_tensor_keys = set(["bt_qid"])

            for key in collated_dict:
                if key not in not_need_to_tensor_keys:
                    collated_dict[key] = torch.tensor(collated_dict[key], dtype=torch.long)
            return collated_dict

        return collate_fn

def padding_seq_to_same_length(input_ids, max_pad_length, pad_token = 0):
    padding_length = max_pad_length - len(input_ids)
    padding_ids = [pad_token] * padding_length
    attention_mask = []

    if padding_length <= 0:
        attention_mask = [1] * max_pad_length
        input_ids = input_ids[:max_pad_length]
    else:
        attention_mask = [1] * len(input_ids) + [0] * padding_length
        input_ids = input_ids + padding_ids
            
    assert len(input_ids) == max_pad_length
    assert len(attention_mask) == max_pad_length
  
    return input_ids, attention_mask

class conversation_format(Dataset):
    def __init__(self, args, tokenizer, filename):
        self.tokenizer = tokenizer
        self.data = []
        self.max_length = 1024
        with open(filename, encoding="utf-8") as f:
            self.data = f.readlines()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        record = json.loads(self.data[idx])
        qid = record["sample_id"]
        question = record["cur_utt_text"]
        context = record["ctx_utts_text"]
        gold_answers = record["answers"]

        context_parts = []
        for i, cont in enumerate(context):
            if i % 2 == 1: # response
                turn = f"Response_{i // 2 + 1}:"
            else:
                turn = f"Question_{i // 2 + 1}:"
            context_parts.append(f"{turn} {cont}")
        
        history = "\n".join(context_parts)
        messages = [
            {"role": "system", "content": "You are a helpful assistant for query-centric summarization."},
            {"role": "user", "content": f"Given a multiple turns conversation with question and answer pairs, please summarize the useful information of the previous turns. The generated summarization should be related and helpful to answer the current question.\nThe previous turns are:\n{history}\n\nThe current question is:{question}\n. Do not give any explanation or say you cannot summarize it. The generated summarization are: "},
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
        return {
            "qid": qid,
            "question": question,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "gold_answers": gold_answers
        }



        
