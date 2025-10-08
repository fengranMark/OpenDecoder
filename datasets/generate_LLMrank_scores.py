import torch
from peft import PeftModel, PeftConfig
import os
import json
import jsonlines
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel, AutoModelForSequenceClassification
from tqdm import tqdm
import pickle


def get_model(peft_model_name):
    config = PeftConfig.from_pretrained(peft_model_name)
    base_model = AutoModelForSequenceClassification.from_pretrained(config.base_model_name_or_path, num_labels=1)
    model = PeftModel.from_pretrained(base_model, peft_model_name)
    model = model.merge_and_unload()
    model.eval()
    return model

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b-hf')
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token 
  
model = get_model('rankllama-v1-7b-lora-doc').to(DEVICE)
model.config.pad_token_id = tokenizer.pad_token_id



def pload(path):
	with open(path, 'rb') as f:
		res = pickle.load(f)
	print('load path = {} object'.format(path))
	return res

def compute_ranking_scores(pairs, batch_size=16):
    scores = []

    for i in tqdm(range(0, len(pairs), batch_size), desc="ranking score"):
        batch_qd = pairs[i:i+batch_size]
        texts = [f"query: {q} document: {d}" for q, d in batch_qd]

        enc = tokenizer(texts, padding=True, truncation=True, max_length=256, return_tensors="pt").to(DEVICE)

        with torch.no_grad():
            outputs = model(**enc)
            logits = outputs.logits.squeeze(-1)  # shape: (batch_size,)
            #probs = torch.sigmoid(logits)
            batch_scores = logits
        scores.extend(batch_scores.cpu().tolist())

    return scores

def main(queries_path, qid_map_path, RAG_text_path, output_path, batch_size=16):
    pid2passage = pload(RAG_text_path)

    with open(qid_map_path, "r", encoding="utf-8") as f:
        qid_retrieval_map = f.readlines()

    qid2pid = {}
    for line in qid_retrieval_map:
        q_sample = json.loads(line)
        rel_items = [q_sample["top_pid"][i] for i in [29, 49, 99]]
        irrel_items = q_sample["irrel_pid"]
        qid2pid[q_sample["id"]] = q_sample["top_pid"][:10] + rel_items + irrel_items[:2]
        #qid2pid[q_sample["id"]] = q_sample["top_pid"][5:10]
        
    queries_data = []
    with jsonlines.open(queries_path, "r") as reader:
        for obj in reader:
            queries_data.append(obj)

    with jsonlines.open(output_path, "a") as writer:
        for qobj in tqdm(queries_data, desc="处理 queries"):
            q = qobj["question"]
            doc_ids = qid2pid[qobj["id"]]

            batch_pairs = [(q, pid2passage[doc_id]) for doc_id in doc_ids]
            scores = compute_ranking_scores(batch_pairs, batch_size=batch_size)

            writer.write({
                "id": qobj["id"],
                "doc_ids": doc_ids,
                "LLM_rank_scores": scores
            })



    print(f"Save to {output_path}")
if __name__ == "__main__":
    queries_path, qid_map_path, RAG_text_path, output_path
    main(
        queries_path="datasets/nq/test.jsonl",        
        qid_map_path="datasets/nq/RAG_test_input.jsonl",
        RAG_text_path="datasets/wikipedia/pid2psg.pkl",     
        output_path="datasets/nq/test_LLM_rank_score.jsonl",
        batch_size=8
    )
    main(
        queries_path="datasets/nq/train.jsonl",        
        qid_map_path="datasets/nq/RAG_train_input.jsonl",
        RAG_text_path="datasets/wikipedia/pid2psg.pkl",     
        output_path="datasets/nq/test_LLM_rank_score.jsonl",
        batch_size=8
    )
