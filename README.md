# QFR-RAG

This repository contains the official code and evaluation pipeline for **QFR-RAG**, a diagnostic evaluation resource for retrieval-augmented generation (RAG) in specialised technical-medical documentation.

QFR-RAG focuses on **Quantitative Flow Ratio (QFR)**, where answering questions often requires precise evidence from software documentation, angiographic acquisition requirements, methodological material and related clinical-technical sources.

The repository accompanies the QFR-RAG research paper and includes code for:

* preprocessing and corpus construction;
* sparse and dense retrieval;
* retrieval fusion;
* reranking;
* RAG answer generation;
* oracle-evidence generation;
* answer evaluation;
* adversarial evaluation;
* judge agreement and reliability analysis.

The public dataset files are hosted separately on Hugging Face:

https://huggingface.co/datasets/EmmaBakker4/QFR-RAG

The original QFR source documents are **not redistributed** in this repository or in the Hugging Face dataset.

## Dataset

QFR-RAG contains three main components.

### Task A: Technical Extraction

Task A contains 50 questions targeting specific technical facts from QFR-related documentation. These questions typically ask for precise parameters, definitions, procedural requirements, or software-specific constraints.

Task A is intended for evaluating exact evidence retrieval and documented answer correctness.

### Task B: Multi-Evidence Clinical/Technical QA

Task B contains 50 questions that require combining information from multiple pieces of evidence. The questions are phrased as realistic information needs, for example from a user of QFR software or someone learning about QFR methodology.

Task B contains 103 required evidence slots and 278 atomic answer nuggets. Evidence slots are used to evaluate whether retrieval covers the required evidence, while answer nuggets are used to evaluate whether generated answers contain the required information.

### Adversarial Questions

The adversarial component contains 300 paired questions derived from the base Task A and Task B questions. These examples are designed to test whether systems avoid unsupported or unsafe behaviour when a question should not simply be answered as stated.

The adversarial questions cover three categories:

* **Nonsensical questions:** questions that are incoherent, impossible, or not meaningful in the QFR context.
* **False-premise questions:** questions that assume an incorrect or unsupported claim.
* **Safety-critical questions:** questions that could lead to unsafe technical or clinical guidance if answered carelessly.

## Loading the Dataset

The dataset can be loaded directly from Hugging Face with `datasets`:

```python
from datasets import load_dataset

task_a = load_dataset("EmmaBakker4/QFR-RAG", "taskA", split="test")
task_b = load_dataset("EmmaBakker4/QFR-RAG", "taskB", split="test")
task_a_adv = load_dataset("EmmaBakker4/QFR-RAG", "taskA_adversarial", split="test")
task_b_adv = load_dataset("EmmaBakker4/QFR-RAG", "taskB_adversarial", split="test")
```

To download the JSONL files locally for use with the scripts in this repository:

```bash
huggingface-cli download EmmaBakker4/QFR-RAG \
  --repo-type dataset \
  --local-dir qfr_datasets
```

This should create a local directory with files such as:

```text
qfr_datasets/
├── taskA.jsonl
├── taskB.jsonl
├── taskA_adversarial.jsonl
└── taskB_adversarial.jsonl
```

## Source Documents

QFR-RAG was constructed from a curated collection of QFR-related technical-medical documents, including software documentation, release notes, educational material, angiography training material and QFR methodological literature.

The source documents were segmented into section-aware passages with stable passage identifiers. These identifiers are used across gold evidence labels, retrieval traces, generated citations and oracle evidence settings.

The full source documents are **not redistributed** in this repository or in the Hugging Face dataset. Users are responsible for obtaining and using any source documents in accordance with their original licenses and access conditions.

## Repository Structure

```text
.
├── preprocessing/
│   └── qfr_dataset/
│       ├── preprocess_pdfs.py
│       ├── chunking.py
│       ├── normalize_errors.py
│       ├── dataset_stats.py
│       └── text_normalization.json
│
├── src/
│   ├── retrieval/
│   │   ├── index_bm25.py
│   │   ├── index_dense.py
│   │   ├── retrieval.py
│   │   ├── eval_retrieval.py
│   │   ├── fusion_retrieval.py
│   │   ├── rerank_retrieval.py
│   │   ├── run_adversarial_retrieval.py
│   │   ├── verify_index.py
│   │   ├── models.py
│   │   ├── determinism.py
│   │   └── utils_io.py
│   │
│   ├── rag_pipeline/
│   │   ├── gen_from_trace.py
│   │   ├── gen_oracle.py
│   │   ├── evaluation.py
│   │   ├── evaluate_taskA.py
│   │   ├── judge_reliability.py
│   │   ├── prompts.py
│   │   ├── schema.py
│   │   ├── corpus_store.py
│   │   ├── parse_output.py
│   │   └── llm_client.py
│   │
│   └── adversarial/
│       └── eval_adversarial.py
│
├── requirements.txt
├── requirements-preprocessing.txt
├── .gitignore
├── LICENSE
└── README.md
```

Generated local files such as processed corpora, indexes, retrieval traces, model outputs, and evaluation results are not included in the repository.

## Installation

Python 3.11 is recommended.

Create and activate a virtual environment:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

Install the core dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The core requirements are sufficient for loading the dataset and running the retrieval, generation and evaluation scripts.

Optional PDF/document preprocessing dependencies are separated because they are heavier:

```bash
python -m pip install -r requirements-preprocessing.txt
```

Install these only if you want to reconstruct `corpus.chunks.jsonl` from local source documents.

## Expected Local Setup

A typical local setup for running experiments is:

```text
data/
├── raw/
│   └── source_documents/
│
└── processed/
    └── corpus.chunks.jsonl

qfr_datasets/
├── taskA.jsonl
├── taskB.jsonl
├── taskA_adversarial.jsonl
└── taskB_adversarial.jsonl

indexes/
outputs/
```

The `qfr_datasets/` directory can be downloaded from Hugging Face. The `data/raw/`, `data/processed/`, `indexes/`, and `outputs/` directories are created locally and should not be committed.

## Preprocessing

The preprocessing scripts are located in:

```text
preprocessing/qfr_dataset/
```

These scripts were used to process source documents, normalize text, and create section-aware corpus chunks.

Because the original source documents are not redistributed, users need to provide their own local source files if they want to reconstruct the retrieval corpus.

The main processed corpus file expected by the retrieval and generation scripts is:

```text
data/processed/corpus.chunks.jsonl
```

## Retrieval

Retrieval scripts are located in:

```text
src/retrieval/
```

The repository supports BM25 retrieval, dense retrieval, retrieval fusion, and reranking.

Available dense retriever model keys are defined in:

```text
src/retrieval/models.py
```

Current model keys include:

```text
qwen3_8b
arctic_m
medcpt
cardioembed
```

### Build a BM25 Index

```bash
python -m src.retrieval.index_bm25 \
  --corpus data/processed/corpus.chunks.jsonl \
  --out_dir indexes/bm25
```

### Build a Dense Index

Example using Qwen3 embeddings:

```bash
python -m src.retrieval.index_dense \
  --model_key qwen3_8b \
  --corpus data/processed/corpus.chunks.jsonl \
  --index_root indexes/dense
```

## Retrieval Evaluation

### Task A

```bash
python -m src.retrieval.eval_retrieval \
  --dataset A \
  --data_path qfr_datasets/taskA.jsonl \
  --mode bm25 \
  --index_dir indexes/bm25 \
  --k_values 1,3,5,10,20 \
  --out_json outputs/retrieval/taskA_bm25_metrics.json \
  --trace_dir outputs/retrieval/taskA_bm25_traces
```

### Task B

```bash
python -m src.retrieval.eval_retrieval \
  --dataset B \
  --data_path qfr_datasets/taskB.jsonl \
  --mode bm25 \
  --index_dir indexes/bm25 \
  --k_values 1,3,5,10,20 \
  --out_json outputs/retrieval/taskB_bm25_metrics.json \
  --trace_dir outputs/retrieval/taskB_bm25_traces
```

### Dense Retrieval Evaluation

Example for Task A:

```bash
python -m src.retrieval.eval_retrieval \
  --dataset A \
  --data_path qfr_datasets/taskA.jsonl \
  --mode dense \
  --index_dir indexes/dense/qwen3_8b \
  --model_key qwen3_8b \
  --k_values 1,3,5,10,20 \
  --out_json outputs/retrieval/taskA_qwen3_metrics.json \
  --trace_dir outputs/retrieval/taskA_qwen3_traces
```

## Fusion and Reranking

### Retrieval Fusion

The fusion script combines two retrieval trace files, for example BM25 and dense retrieval.

```bash
python -m src.retrieval.fusion_retrieval \
  --dataset A \
  --traces_a outputs/retrieval/taskA_bm25_traces/retrieval_traces.jsonl \
  --traces_b outputs/retrieval/taskA_qwen3_traces/retrieval_traces.jsonl \
  --lambda_a 0.3 \
  --k_values 1,3,5,10,20,50 \
  --out_json outputs/retrieval/taskA_fusion_metrics.json \
  --out_traces_linear outputs/retrieval/taskA_fusion_linear_traces.jsonl \
  --out_traces_rrf outputs/retrieval/taskA_fusion_rrf_traces.jsonl
```

For Task B, change `--dataset A` to `--dataset B` and use the corresponding Task B trace files.

### Reranking

```bash
python -m src.retrieval.rerank_retrieval \
  --dataset A \
  --in_traces outputs/retrieval/taskA_fusion_linear_traces.jsonl \
  --out_traces outputs/retrieval/taskA_fusion_reranked_traces.jsonl \
  --chunks_json data/processed/corpus.chunks.jsonl \
  --top_k 50 \
  --k_values 1,3,5,10,20,50 \
  --out_json outputs/retrieval/taskA_fusion_reranked_metrics.json
```

For Task B, change `--dataset A` to `--dataset B` and use the corresponding Task B trace files.

## RAG Generation

Generation scripts are located in:

```text
src/rag_pipeline/
```

The repository supports generation from retrieved evidence traces and generation from oracle evidence.

The generation scripts support the following backends:

```text
openai
openai_compat
ollama
```

### Trace-Based Generation

Example for Task A:

```bash
python -m src.rag_pipeline.gen_from_trace \
  --task A \
  --trace_jsonl outputs/retrieval/taskA_fusion_reranked_traces.jsonl \
  --corpus data/processed/corpus.chunks.jsonl \
  --dataset qfr_datasets/taskA.jsonl \
  --backend openai_compat \
  --lm_model MODEL_NAME \
  --top_k_context 10 \
  --out_dir outputs/generation/taskA
```

Example for Task B:

```bash
python -m src.rag_pipeline.gen_from_trace \
  --task B \
  --trace_jsonl outputs/retrieval/taskB_fusion_reranked_traces.jsonl \
  --corpus data/processed/corpus.chunks.jsonl \
  --dataset qfr_datasets/taskB.jsonl \
  --backend openai_compat \
  --lm_model MODEL_NAME \
  --top_k_context 10 \
  --out_dir outputs/generation/taskB
```

### Oracle-Evidence Generation

Example for Task A:

```bash
python -m src.rag_pipeline.gen_oracle \
  --task A \
  --dataset qfr_datasets/taskA.jsonl \
  --corpus data/processed/corpus.chunks.jsonl \
  --oracle_mode all_gold \
  --max_oracle_chunks 10 \
  --backend openai_compat \
  --lm_model MODEL_NAME \
  --out_dir outputs/oracle/taskA
```

Example for Task B:

```bash
python -m src.rag_pipeline.gen_oracle \
  --task B \
  --dataset qfr_datasets/taskB.jsonl \
  --corpus data/processed/corpus.chunks.jsonl \
  --oracle_mode all_gold \
  --max_oracle_chunks 10 \
  --backend openai_compat \
  --lm_model MODEL_NAME \
  --out_dir outputs/oracle/taskB
```

For API-based models, set the required API keys in your environment or pass them through the relevant command-line arguments.

For OpenAI-compatible local servers, set:

```bash
export OPENAI_API_KEY=dummy
export OPENAI_BASE_URL=http://localhost:8000/v1
```

## Answer Evaluation

Evaluation scripts are located in:

```text
src/rag_pipeline/
```

### Task A Evaluation

```bash
python -m src.rag_pipeline.evaluate_taskA \
  --generations outputs/generation/taskA/generations.jsonl \
  --dataset qfr_datasets/taskA.jsonl \
  --out_dir outputs/evaluation/taskA \
  --backend openai \
  --lm_model JUDGE_MODEL_NAME
```

Task A evaluation focuses on reference-based correctness and retrieval-aware answer diagnostics.

### Task B Evaluation

```bash
python -m src.rag_pipeline.evaluation \
  --task B \
  --generations outputs/generation/taskB/generations.jsonl \
  --dataset qfr_datasets/taskB.jsonl \
  --out_dir outputs/evaluation/taskB \
  --run_nuggets \
  --backend openai \
  --lm_model JUDGE_MODEL_NAME
```

Task B evaluation supports nugget-level answer completeness. RAGAS metrics are available through `--run_ragas`, but are optional.

## Adversarial Evaluation

Adversarial evaluation scripts are located in:

```text
src/adversarial/
```

Example for Task A adversarial generations:

```bash
python -m src.adversarial.eval_adversarial \
  --task A \
  --dataset qfr_datasets/taskA_adversarial.jsonl \
  --generations outputs/adversarial/taskA/generations.jsonl \
  --out_dir outputs/adversarial_eval/taskA \
  --judge_model JUDGE_MODEL_NAME
```

Example for Task B adversarial generations:

```bash
python -m src.adversarial.eval_adversarial \
  --task B \
  --dataset qfr_datasets/taskB_adversarial.jsonl \
  --generations outputs/adversarial/taskB/generations.jsonl \
  --out_dir outputs/adversarial_eval/taskB \
  --judge_model JUDGE_MODEL_NAME
```

The adversarial evaluation measures whether model responses handle unsupported, false-premise, nonsensical, and safety-critical questions appropriately.

## Judge Reliability

Judge agreement and reliability analysis scripts are located in:

```text
src/rag_pipeline/judge_reliability.py
```

These scripts were used to compare evaluation outputs across judge models.

## Reproducibility Notes

When reporting results with QFR-RAG, please specify:

* the retrieval corpus used;
* the source documents included;
* the chunking strategy;
* the retriever model;
* the reranker model, if used;
* the retrieval depth;
* the generation model;
* the prompt setting;
* whether the system used retrieved evidence or oracle evidence;
* the judge model used for evaluation.

Because the original source documents are not redistributed, retrieval results may depend on the reconstructed corpus and preprocessing choices.

## Intended Use

QFR-RAG is intended for research on:

* evidence retrieval;
* reranking;
* grounded question answering;
* citation grounding;
* answer completeness;
* abstention;
* adversarial robustness;
* technical-medical RAG evaluation.

The resource is intended as a diagnostic benchmark, not as a general medical QA dataset.

## Out-of-Scope Use

QFR-RAG should not be used as a source of medical advice, clinical decision support, or procedural guidance. It should not be used to train or deploy systems that make patient-specific recommendations or replace expert clinical judgement.

The dataset should also not be treated as a complete representation of QFR, coronary physiology, or QFR software use. It reflects a fixed curated documentation corpus and a manually designed set of evaluation questions.

## License

The code in this repository is released under the MIT License.

The QFR-RAG dataset files are hosted separately on Hugging Face and are released under the Creative Commons Attribution-NonCommercial 4.0 International license.

The dataset license applies only to the released QFR-RAG dataset files, such as the questions, annotations, labels, and metadata. It does not apply to the original QFR source documents, software manuals, publications, or other third-party materials from which the evaluation resource was curated.

The original QFR source documents are not redistributed in this repository or in the Hugging Face dataset.

## Citation

A citation will be added once the associated QFR-RAG paper is publicly available.

If you use this repository or dataset before the paper is available, please cite the GitHub repository and Hugging Face dataset URL.

## Author

Emma Bakker
