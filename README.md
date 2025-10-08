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

### 3.1 Access the LLM
Since the OpenDecoder requires modifying the original attention network computation, the first step is to access the LLM by downloading the Qwen-2.5-3B-instruct [checkpoint](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct) to the path ./checkpoint. As the original official source code does not support additional input to inject relevant indicators, we need to load the initial LLM's weight into a modified architecture of LLM with relevance features as one of the input arguments via the script.

    # Remember to adjust Arguments according to the used version of the backbone model
    bash ./utils/iniModel.sh 

### 3.2 Modulate Computation
The modified architecture of LLM is under the path with the corresponding configuration.

    ./src/model/qwen_decoder/modeling.py

Within the architecture, we modify the computation of the function "eager_attention_forward" with 

    if kwargs.get("relevant_scores", None) is not None: 
        relevant_scores = kwargs["relevant_scores"].unsqueeze(1).unsqueeze(-1).to(query.dtype)
        query = query * relevant_scores

## 4. OpenDecoder
(1) Train OpenDecoder:

    bash train.sh

The used indicator features and robust training are controlled by 

    --add_irrelevant_psg True/False \ # whether add noisy doc for Robust Rraining
    --add_LLM_scores True/False \ # whether add LLM-rank scores
    --add_QPP_scores True/False \ # whether add QPP scores

(2) Inference via OpenDecoder:

    bash inference.sh

The evaluation settings are controlled by 

    --add_irrelevant_psg True/False \ # evaluate in noisy setting
    --full_irrelevant_psg True/False \ # evaluate in extreme noisy setting


    
