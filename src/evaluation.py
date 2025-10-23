#from src.eval.eval_metric import f1_score, acc_score
import json, string, re
from collections import Counter, defaultdict
from argparse import ArgumentParser

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)
    def white_space_fix(text):
        return ' '.join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))

def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    #breakpoint()
    return f1

def acc_score(prediction, ground_truth):
    prediction = normalize_answer(prediction)
    ground_truth = normalize_answer(ground_truth)
    if ground_truth in prediction:
        return 1
    return 0

def exact_match_score(prediction, ground_truth):
    return (normalize_answer(prediction) == normalize_answer(ground_truth))

def calculate_metric_hotpotqa(pred_path, gold_path, result_path, start=None, end=None):
    if start is not None:
        with open(pred_path, 'r') as f1, open(gold_path, 'r') as f2:
            pred_data, gold_data = f1.readlines()[start:end], f2.readlines()
    else:
        with open(pred_path, 'r') as f1, open(gold_path, 'r') as f2:
            pred_data, gold_data = f1.readlines(), f2.readlines()
    with open(result_path, 'w') as fout:
        f1, acc, em = [], [], []
        for predication, golds in zip(pred_data, gold_data):
            pred_record = json.loads(predication)
            gold_record = json.loads(golds)
            pred = pred_record["pred"].replace("<|im_start|>\n", "").replace("system\n", "").replace("<|im_start|>", "")
            gold_ans = gold_record["answer"]
            best_f1 = f1_score(pred, gold_ans)
            best_acc = acc_score(pred, gold_ans)
            best_em = exact_match_score(pred, gold_ans)
            f1.append(best_f1)
            acc.append(best_acc)
            em.append(best_em)
            #breakpoint()
        avg_f1 = sum(f1) / len(f1)
        fout.write(f"\nAverage F1 score over evaluation set: {avg_f1:.4f}\n")
        #print(f"\nAverage F1 score over evaluation set: {avg_f1:.4f}") 
        avg_em = sum(em) / len(em)
        #print(f"\nAverage EM. score over evaluation set: {avg_em:.4f}") 
        fout.write(f"\nAverage EM. score over evaluation set: {avg_em:.4f}")
        avg_acc = sum(acc) / len(acc)
        #print(f"\nAverage ACC. score over evaluation set: {avg_acc:.4f}") 
        fout.write(f"\nAverage ACC. score over evaluation set: {avg_acc:.4f}")
        print(f"{avg_f1*100:.2f} & {avg_em*100:.2f} & {avg_acc*100:.2f}")

def calculate_metric(pred_path, gold_path, result_path, start=None, end=None):
    if start is not None:
        with open(pred_path, 'r') as f1, open(gold_path, 'r') as f2:
            pred_data, gold_data = f1.readlines()[start:end], f2.readlines()
    else:
        with open(pred_path, 'r') as f1, open(gold_path, 'r') as f2:
            pred_data, gold_data = f1.readlines(), f2.readlines()

    with open(result_path, 'w') as fout:
        f1, acc, em = [], [], []
        for predication, golds in zip(pred_data, gold_data):
            pred_record = json.loads(predication)
            gold_record = json.loads(golds)
            #breakpoint()
            pred = pred_record["pred"].replace("<|im_start|>\n", "").replace("system\n", "").replace("<|im_start|>", "")
            if "golden_answers" in gold_record:
                gold_ans = gold_record["golden_answers"]
            elif "answers" in gold_record:
                gold_ans = gold_record["answers"]
            best_f1, best_acc, best_em = 0.0, 0.0, 0.0
            cur_f1, cur_acc, cur_em = 0.0, 0.0, 0.0
            for gold in gold_ans:
                cur_f1 = f1_score(pred, gold)
                cur_acc = acc_score(pred, gold)
                cur_em = exact_match_score(pred, gold)
                best_f1 = max(best_f1, cur_f1)
                best_acc = max(best_acc, cur_acc)
                best_em = max(best_em, cur_em)
                #breakpoint()
            f1.append(best_f1)
            acc.append(best_acc)
            em.append(best_em)
            #breakpoint()
        avg_f1 = sum(f1) / len(f1)
        fout.write(f"\nAverage F1 score over evaluation set: {avg_f1:.4f}\n")
        #print(f"\nAverage F1 score over evaluation set: {avg_f1:.4f}") 
        avg_em = sum(em) / len(em)
        #print(f"\nAverage EM. score over evaluation set: {avg_em:.4f}") 
        fout.write(f"\nAverage EM. score over evaluation set: {avg_em:.4f}")
        avg_acc = sum(acc) / len(acc)
        #print(f"\nAverage ACC. score over evaluation set: {avg_acc:.4f}") 
        fout.write(f"\nAverage ACC. score over evaluation set: {avg_acc:.4f}")
        print(f"{avg_f1*100:.2f} & {avg_em*100:.2f} & {avg_acc*100:.2f}")

# data_path = "./noisy_output/Qwen2.5-1.5B-Instruct_nq_hotpotqa_open_top10_irrel"
# dataset_path = "./datasets"
# pred_path = f"{data_path}/all/pred.json"
# gold_path = f"{dataset_path}/nq/test.jsonl"
# result_path = f"{data_path}/nq_result.txt"
# calculate_metric(pred_path, gold_path, result_path, 0, 1500)

# pred_path = f"{data_path}/all/pred.json"
# gold_path = f"{dataset_path}/trivialqa/trivialqa_test.jsonl"
# result_path = f"{data_path}/trivialqa_result.txt"
# calculate_metric(pred_path, gold_path, result_path, 4000, 5500)

# pred_path = f"{data_path}/all/pred.json"
# gold_path = f"{dataset_path}/popqa/popqa_longtail.jsonl"
# result_path = f"{data_path}/popqa_result.txt"
# calculate_metric(pred_path, gold_path, result_path, 3000, 4000)

# pred_path = f"{data_path}/all/pred.json"
# gold_path = f"{dataset_path}/hotpotqa/hotpot_dev.jsonl"
# result_path = f"{data_path}/hotpotqa_result.txt"
# calculate_metric_hotpotqa(pred_path, gold_path, result_path, 1500, 3000)

# pred_path = f"{data_path}/all/pred.json"
# gold_path = f"{dataset_path}/2wiki/2wiki_dev.jsonl"
# result_path = f"{data_path}/2wiki_result.txt"
# calculate_metric(pred_path, gold_path, result_path, 5500, 7000)


data_path = "./output/Qwen2.5-1.5B-Instruct_RAG_top10"
dataset_path = "./datasets"
pred_path = f"{data_path}/nq/pred.json"
gold_path = f"{dataset_path}/nq/test.jsonl"
result_path = f"{data_path}/nq/result.txt"
calculate_metric(pred_path, gold_path, result_path)

pred_path = f"{data_path}/trivialqa/pred.json"
gold_path = f"{dataset_path}/trivialqa/trivialqa_test.jsonl"
result_path = f"{data_path}/trivialqa/result.txt"
calculate_metric(pred_path, gold_path, result_path)

pred_path = f"{data_path}/popqa/pred.json"
gold_path = f"{dataset_path}/popqa/popqa_longtail.jsonl"
result_path = f"{data_path}/popqa/result.txt"
calculate_metric(pred_path, gold_path, result_path)

pred_path = f"{data_path}/hotpotqa/pred.json"
gold_path = f"{dataset_path}/hotpotqa/hotpot_dev.jsonl"
result_path = f"{data_path}/hotpotqa/result.txt"
calculate_metric_hotpotqa(pred_path, gold_path, result_path)

pred_path = f"{data_path}/2wiki/pred.json"
gold_path = f"{dataset_path}/2wiki/2wiki_dev.jsonl"
result_path = f"{data_path}/2wiki/result.txt"
calculate_metric(pred_path, gold_path, result_path)
