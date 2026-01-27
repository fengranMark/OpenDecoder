import torch
from transformers import AutoTokenizer
from src.model.qwen_decoder.modeling import IModelForCausalLM
from src.model.qwen_decoder.configuration import IConfig

device = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_PATH = "/data/rech/mofengra/opendecoder/ckpts/Qwen2.5-3B-Instruct_nq_hotpotqa_open_top10_irrel_shuffle/checkpoint-21000"

# ------------------
# Load model/tokenizer
# ------------------
config = IConfig.from_pretrained(MODEL_PATH)
model = IModelForCausalLM.from_pretrained(MODEL_PATH, config=config).to(device).eval()

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True,
    padding_side="left",
)

# Add Passage tokens (must match training)
special_passage_tokens = [f"Passage_{i+1}:" for i in range(20)]
tokenizer.add_special_tokens({"additional_special_tokens": special_passage_tokens})
model.resize_token_embeddings(len(tokenizer))

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.unk_token
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id


# ------------------
# Example input
# ------------------
question = "Who wrote the novel The Old Man and the Sea?"

documents = [
    "The Old Man and the Sea is a short novel written by Ernest Hemingway in 1951.",
    "Ernest Hemingway was an American novelist and short-story writer.",
    "The book won the Pulitzer Prize for Fiction in 1953.",
    "It tells the story of an aging Cuban fisherman.",
    "Hemingway also wrote For Whom the Bell Tolls.",
    "The novella was published in Life magazine.",
    "It contributed to Hemingway winning the Nobel Prize.",
    "The protagonist is named Santiago.",
    "The story is set in the Gulf Stream.",
    "The work is considered one of Hemingway's classics."
]

# document-level relevance (length = 10)
doc_scores = [0.95, 0.9, 0.4, 0.2, 0.1, 0.1, 0.05, 0.05, 0.05, 0.05]

# normalize exactly like your dataset (normal mode)
mx = max(doc_scores)
norm_scores = [s / mx for s in doc_scores]


# ------------------
# Build RAG prompt
# ------------------
context_parts = []
for i, doc in enumerate(documents):
    context_parts.append(f"Passage_{i+1}: {doc}")

context = "\n".join(context_parts)

messages = [
    {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
    {
        "role": "user",
        "content": (
            "You should answer the question by referring to the knowledge provided below and integrating "
            "the usefulness of your own knowledge. Just directly answer it in several words as a short answer "
            "without any explanation.\n"
            f"{context}\n\nQuestion:{question}\n"
        ),
    },
]

prompt = tokenizer.apply_chat_template(messages, tokenize=False)

tokenized = tokenizer(
    prompt,
    return_tensors="pt",
    padding="max_length",
    truncation=True,
    max_length=4096,
)

input_ids = tokenized["input_ids"].to(device)
attention_mask = tokenized["attention_mask"].to(device)

seq_len = input_ids.shape[1]


# ------------------
# Build token-level relevance_scores
# ------------------
relevance_scores = torch.ones(seq_len, dtype=torch.float)

# find Passage_i token positions
passage_starts = []
for i in range(len(documents)):
    tok = f"Passage_{i+1}:"
    tok_id = tokenizer.convert_tokens_to_ids(tok)
    matches = (input_ids[0] == tok_id).nonzero(as_tuple=True)[0]
    passage_starts.append(matches[0].item())

# find assistant start (same logic as dataset)
im_start = tokenizer.convert_tokens_to_ids("<|im_start|>")
assistant = tokenizer.convert_tokens_to_ids("assistant")

label_start = seq_len
positions = (input_ids[0] == im_start).nonzero(as_tuple=True)[0].tolist()
for p in reversed(positions):
    if input_ids[0][p + 1] == assistant:
        label_start = p
        break

# compute passage spans
spans = []
for i in range(len(passage_starts)):
    s = passage_starts[i]
    e = passage_starts[i + 1] if i < len(passage_starts) - 1 else label_start - 1
    spans.append((s, e))

# assign relevance per token
for i, (s, e) in enumerate(spans):
    relevance_scores[s:e] = norm_scores[i]

relevance_scores = relevance_scores.unsqueeze(0).to(device)


# ------------------
# Generate
# ------------------
with torch.no_grad():
    outputs = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        relevant_scores=relevance_scores,
        max_new_tokens=64,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

answer = tokenizer.decode(
    outputs[0][input_ids.shape[-1]:],
    skip_special_tokens=True,
).strip().replace("assistant", "").replace("<|im_start|>\n", "").replace("system\n", "")

print("Answer:", answer)
