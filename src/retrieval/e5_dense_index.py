import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

import os
import sys
sys.path.append('..')
sys.path.append('.')
import time
import json
import array
import pickle
import argparse
import numpy as np
from os.path import join as oj
from tqdm import tqdm
import random

import torch
from torch import nn
from torch.utils.data import DataLoader, IterableDataset
from transformers import (RobertaConfig, RobertaModel,
                          RobertaForSequenceClassification, RobertaTokenizer,
                          AutoTokenizer, AutoModel, AutoModelForCausalLM)

#from utils import set_seed, pstore
#os.environ['CUDA_VISIBLE_DEVICES'] = '6'

def pstore(x, path, high_protocol = False):
    with open(path, 'wb') as f:
        if high_protocol:  
            pickle.dump(x, f, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            pickle.dump(x, f)
    print('store object in path = {} ok'.format(path))

class StreamIndexDataset(IterableDataset):
    def __init__(self, collection_path):
        super().__init__()
        self.collection_path = collection_path

    def __iter__(self):
        with open(self.collection_path, "r") as f:
            first_line = True
            for line in f:
                if first_line:
                    first_line = False
                    continue
                line = line.strip().split('\t')
                if len(line) == 1:
                    line.append("")
                yield line[0], line[1]

class CollateClass:
    def __init__(self, args, tokenizer, prefix=""):
        self.args = args
        self.doc_prefix = prefix
        self.tokenizer = tokenizer
        self.max_doc_length = args.max_doc_length

    def collate_fn(self, batch):
        """
        batch is a list of tuples, each tuple has 2 (text) items (id_, doc)
        """
        id_, docs = zip(*batch)
        docs = list(docs)
        if len(self.doc_prefix) > 0:
            for i in range(len(docs)):
                docs[i] = self.doc_prefix + docs[i]
        
        tokenized_docs = self.tokenizer(docs,
                                        add_special_tokens=True,
                                        padding="longest",  # pad to max sequence length in batch
                                        truncation="longest_first",  # truncates to self.max_length
                                        max_length=self.max_doc_length,
                                        return_attention_mask=True)
        return {**{k: torch.tensor(v) for k, v in tokenized_docs.items()},
                "id": id_}

def dense_indexing(args):
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_doc_encoder_path)
    model = AutoModel.from_pretrained(args.pretrained_doc_encoder_path).to(args.device)
    model.to(args.device)
    
    indexing_batch_size = args.per_gpu_index_batch_size
    indexing_dataset = StreamIndexDataset(args.collection_path)
    prefix = ""
    collate_func = CollateClass(args, tokenizer, prefix=prefix)
    index_dataloader =  DataLoader(indexing_dataset, 
                                   batch_size=indexing_batch_size, 
                                   collate_fn=collate_func.collate_fn)

    doc_ids = []
    doc_embeddings = []
    cur_block_id = 0
    num_doc_embs = 0
    num_per_block_docs = 5000000 # 3844000 is ~6.9GB
    with torch.no_grad():
        model.eval()
        for batch in tqdm(index_dataloader, desc="Dense Indexing", position=0, leave=True):
      
            inputs = {k: v.to(args.device) for k, v in batch.items() if k not in {"id"}}

            # Mean Pooling for E5
            batch_doc_embs = model(**inputs)
            attention_mask = inputs['attention_mask']
            last_hidden = batch_doc_embs.last_hidden_state
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
            batch_doc_embs = torch.sum(last_hidden * input_mask_expanded, 1) / input_mask_expanded.sum(1)
            batch_doc_embs = batch_doc_embs.detach().cpu().numpy()
            doc_embeddings.append(batch_doc_embs)
     
            for doc_id in batch["id"]:
                #doc_ids.append(int(doc_id))
                doc_ids.append(doc_id)
        
            if len(doc_ids) >= num_per_block_docs:
                doc_embeddings = np.concatenate(doc_embeddings, axis=0)
                doc_ids = np.array(doc_ids)
                emb_output_path = oj(args.output_index_dir_path, "doc_emb_block.{}.pb".format(cur_block_id))
                embid_output_path = oj(args.output_index_dir_path, "doc_embid_block.{}.pb".format(cur_block_id))
                pstore(doc_embeddings, emb_output_path, high_protocol=True)
                pstore(doc_ids, embid_output_path, high_protocol=True)
                
                num_doc_embs += len(doc_ids)
                doc_ids = []
                doc_embeddings = []
                cur_block_id += 1
    
    if len(doc_ids) > 0:
        doc_embeddings = np.concatenate(doc_embeddings, axis=0)
        doc_ids = np.array(doc_ids)
        emb_output_path = oj(args.output_index_dir_path, "doc_emb_block.{}.pb".format(cur_block_id))
        embid_output_path = oj(args.output_index_dir_path, "doc_embid_block.{}.pb".format(cur_block_id))
        pstore(doc_embeddings, emb_output_path, high_protocol=True)
        pstore(doc_ids, embid_output_path, high_protocol=True)    

        num_doc_embs += len(doc_ids)
        doc_ids = []
        doc_embeddings = []
        cur_block_id += 1
    
    print("Totally {} docs in {} blocks are stored.".format(num_doc_embs, cur_block_id))



def get_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--collection_path", type=str, default="../../datasets/wikipedia/psgs_w100.tsv")
    parser.add_argument("--pretrained_doc_encoder_path", type=str, default="../../checkpoints/e5-base-v2")
    
    parser.add_argument("--output_index_dir_path", type=str, default="../../datasets/wikipedia/index")
    parser.add_argument("--force_emptying_dir", action="store_true", default=True)

    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--use_data_percent", type=float, default=1.0, help="Percent of samples to use. Faciliating the debugging.")
    parser.add_argument("--per_gpu_index_batch_size", type=int, default=500)

    parser.add_argument("--max_doc_length", type=int, default=256, help="Max doc length")
    

    args = parser.parse_args()
    # pytorch parallel gpu
    device = torch.device("cuda:7" if torch.cuda.is_available() else "cpu")
    args.device = device
    args.start_running_time = time.asctime(time.localtime(time.time()))
    logger.info("---------------------The arguments are:---------------------")
    logger.info(args)
        
    return args


if __name__ == "__main__":
    args = get_args()
    #set_seed(args)

    dense_indexing(args)
