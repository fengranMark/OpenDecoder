import csv, json
import pickle
import random
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F

def pload(path):
    with open(path, 'rb') as f:
        res = pickle.load(f)
    print(f'load path = {path} object')
    return res
  
def construct_RAG_input(data_file, retrieval_file, output_file, collection, model_path):
    qid2topk_pid, pid_list, qid2scores, score_list = {}, [], {}, []
    
    with open(retrieval_file, 'r') as f:
        ret_data = f.readlines()
    
    for idx in range(len(ret_data)):
        line = ret_data[idx].strip().split()
        qid, pid, rank, rel_score = line[0], int(line[2]), int(line[3]), float(line[-1])
        pid_list.append(pid)
        score_list.append(rel_score)
        if rank == 100:
            qid2topk_pid[qid] = pid_list
            qid2scores[qid] = score_list
            pid_list, score_list = [], []
          
    with open(RAG_file, 'w') as g:
        record = {}
        for qid, topk in qid2topk_pid.items():
            record["id"] = qid
            #for pid in topk:
            record["top_pid"] = topk
            record["top_pid_score"] = qid2scores[qid]
            g.write(json.dumps(record) + "\n")
    

retrieval_file = "nq_train_e5.trec"
RAG_file = "RAG_train_input.jsonl"
construct_RAG_input(retrieval_file, RAG_file)
retrieval_file = "nq_test_e5.trec"
RAG_file = "RAG_test_input.jsonl"
construct_RAG_input(retrieval_file, RAG_file)

def get_embedding(text, tokenizer, model, device):
    tokens = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(device)
    with torch.no_grad():
        output = model(**tokens)
        token_embeddings = output.last_hidden_state  # [batch, seq, hidden]
        attention_mask = tokens['attention_mask']
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size())
        return (token_embeddings * input_mask_expanded).sum(1) / input_mask_expanded.sum(1)  # mean pooling


def calculate_similarity(model, tokenizer, query, passage, device):
    query = "query: " + query
    passage = "passage: " + passage

    query_emb = get_embedding(query, tokenizer, model, device)      # [1, hidden]
    passage_emb = get_embedding(passage, tokenizer, model, device)  # [1, hidden]
    similarity = torch.matmul(query_emb, passage_emb.T)  # [1, 1]
    return round(similarity.item(), 4)

def construct_noisy_evaluation(collection, input_file, input_file_2, output_file, model_path):
    pid2passage = pload(collection)
    len_collection = len(pid2passage)

    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModel.from_pretrained(model_path).to(device)
    model.eval()

    with open(input_file, 'r') as f1, open(input_file_2, 'r') as f2, open(output_file, 'w') as g:
        data, data_2 = f1.readlines(), f2.readlines()
        for i in range(len(data)):
            record, record_2 = json.loads(data[i]), json.loads(data_2[i])
            RAG_pid, RAG_score = record_2["top_pid"], record_2["top_pid_score"]
            query = record["question"]
            assert record["id"] == record_2["id"]

            irrel_pid, irrel_pid_score = [], []
            while len(irrel_pid) < 10:
                irrel_int = random.randint(1, len_collection)
                if irrel_int not in RAG_pid:
                    passage = pid2passage[irrel_int]
                    score = calculate_similarity(model, tokenizer, query, passage, device)
                    irrel_pid.append(irrel_int)
                    irrel_pid_score.append(score)

            record_2["irrel_pid"] = irrel_pid
            record_2["irrel_pid_score"] = irrel_pid_score
            print("processed", i)
            g.write(json.dumps(record_2) + '\n')

collection = "wikipedia/pid2psg.pkl"
input_file = "train.jsonl"
input_file_2 = "RAG_train_input.jsonl"
output_file = "RAG_train_input.jsonl"
model_path = "e5-base-v2"
construct_noisy_evaluation(collection, input_file, input_file_2, output_file, model_path)
input_file = "train.jsonl"
input_file_2 = "RAG_test_input.jsonl"
output_file = "RAG_test_input.jsonl"
construct_noisy_evaluation(collection, input_file, input_file_2, output_file, model_path)
