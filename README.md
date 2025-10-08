# OpenDecoder
A repository of the OpenDecoder framework: Open Large Language Model Decoding to Incorporate Document Quality in RAG.

# Running Steps
## 1. Download data and Preprocessing

The used RAG datasets can be downloaded via [FlashRAG](https://huggingface.co/datasets/RUC-NLPIR/FlashRAG_datasets) to obtain the processed datasets. The e5 retriever can be loaded via the [checkpoint](https://huggingface.co/intfloat/e5-base-v2), and the collection for retrieval can be downloaded via the following script.

    wget https://dl.fbaipublicfiles.com/dpr/wikipedia_split/psgs_w100.tsv.gz

Then, the pre-computed embeddings can be established via the script

    python datasets/e5_dense_index.py

## 2. Searching External Information and Construct Quality Indicators

The first step of RAG is to retrieve relevant documents, which is achieved by the script as below, and obtain the retrieved top-k list in TREC format.

    python datasets/test_e5_retrieval.py

Since we need to use the relevance scores as document quality indicators in OpenDecoder, we store the top-k documents' ID and scores, as well as the sampled irrelevant ones, for robust training via the script below, and produce the result file RAG_train/test_input.jsonl.

    python datasets/construct_indicators.py

The format of the result file is

    {"id": "", "top_pid": [], "top_pid_score": [], "irrel_pid": [], "irrel_pid_score": []}

For generating LLM-rank score and QPP score, please run the script as below to produce the result file with the same format.

    python datasets/generate_LLMrank_scores.py
    python datasets/generate_QPP_scores.py

## 3. Open the LLM to Modulate the Computation of the Decoder

### 3.1 
