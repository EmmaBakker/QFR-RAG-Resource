# QFR-RAG

**QFR-RAG** is a diagnostic evaluation resource and reproducible experimental pipeline for retrieval-augmented generation (RAG) over specialised technical-medical documentation.

The resource focuses on **Quantitative Flow Ratio (QFR)**, a documentation-heavy setting where correct answers often require precise evidence about software use, angiographic acquisition constraints, methodological assumptions, and related coronary physiology concepts. The goal of this repository is not to build a clinical assistant, but to support controlled evaluation of retrieval, evidence use, grounded generation, abstention behaviour, citation grounding, and adversarial robustness.

This repository accompanies the thesis:

> **QFR-RAG: A Multi-Level Evaluation Resource for Grounded RAG over Technical-Medical Documentation**

The public dataset files are hosted separately on Hugging Face:

```text
https://huggingface.co/datasets/EmmaBakker4/QFR-RAG
```

The original QFR source documents are **not redistributed** in this repository or in the Hugging Face dataset.

## What is included

This repository contains code for:

- preprocessing local source documents into section-aware corpus chunks;
- building BM25 and dense retrieval indexes;
- evaluating retrieval for Task A and Task B;
- combining BM25 and dense retrieval through linear fusion;
- reranking fused retrieval results with a Qwen3 reranker;
- generating standard RAG answers from retrieved traces;
- generating closed-book answers without retrieved evidence;
- generating oracle-evidence answers from annotated gold evidence;
- running full-corpus / citation-alias diagnostic generation;
- evaluating Task A correctness and taxonomy;
- evaluating Task B nugget recall, taxonomy, and optional RAGAS-style metrics;
- evaluating adversarial robustness;
- evaluating BioASQ validation runs;
- computing inter-judge reliability from explicit judge-output pairs.

## Dataset

QFR-RAG contains three main components.

### Task A: Technical Extraction

Task A contains **50 questions** targeting exact technical facts from QFR-related documentation. These questions ask for precise parameters, definitions, procedural requirements, or software-specific constraints. Task A is evaluated with chunk-level gold evidence labels and reference-answer correctness.

### Task B: Multi-Evidence QFR Question Answering

Task B contains **50 questions** that require combining information from multiple pieces of evidence. These questions are phrased as realistic information needs from a QFR software user or someone learning about QFR methodology.

Task B contains:

- **103 required evidence slots** for retrieval sufficiency evaluation;
- **278 atomic answer nuggets** for generation completeness evaluation.

The slot annotations support retrieval-side diagnosis, while the nugget annotations support answer-side diagnosis.

### Adversarial Questions

The adversarial component contains **300 paired questions** derived from the base Task A and Task B questions. These examples test whether systems avoid inappropriate answering when the question should not simply be answered as stated.

The adversarial categories are:

- **Nonsensical questions:** incoherent, impossible, or meaningless questions in the QFR context;
- **False-premise questions:** questions that assume an incorrect or unsupported claim;
- **Safety-critical questions:** questions where careless answering could produce unsafe technical or clinical guidance.

## Loading the dataset

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

Expected local dataset layout:

```text
qfr_datasets/
├── taskA.jsonl
├── taskB.jsonl
├── taskA_adversarial.jsonl
└── taskB_adversarial.jsonl
```

## Source documents

QFR-RAG was constructed from a curated corpus of QFR-related technical-medical material, including software documentation, release notes, educational material, angiography training material, and QFR methodological literature.

The source documents were segmented into section-aware chunks with stable passage identifiers. These identifiers are used across gold evidence labels, retrieval traces, generated citations, and oracle-evidence settings.

The full source documents are **not redistributed**. Users who want to reproduce corpus-dependent retrieval and generation experiments must provide their own local source documents and ensure that their use complies with the original licenses and access conditions.

## Repository structure

```text
.
├── preprocessing/
│   ├── qfr_dataset/
│   │   ├── preprocess_pdfs.py
│   │   ├── chunking.py
│   │   ├── normalize_errors.py
│   │   ├── dataset_stats.py
│   │   └── text_normalization.json
│   └── bioasq_dataset/
│       ├── bioasq_make_subset.py
│       ├── bioasq_fetch_pubmed.py
│       └── chunking.py
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
│   │   ├── gen_closed_book.py
│   │   ├── gen_oracle.py
│   │   ├── gen_full_corpus_alias.py
│   │   ├── bioasq_gen_from_trace.py
│   │   ├── evaluate_taskA.py
│   │   ├── evaluation.py
│   │   ├── eval_bioasq.py
│   │   ├── inter_judge_reliability.py
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

Generated local files such as processed corpora, indexes, retrieval traces, model outputs, and evaluation results are intentionally not included in the repository.

## Installation

Python 3.11 is recommended.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Optional PDF/document preprocessing dependencies are separated because they are heavier:

```bash
python -m pip install -r requirements-preprocessing.txt
```

Install the preprocessing requirements only if you want to reconstruct `corpus.chunks.jsonl` from local source documents.

## Expected local layout

A typical local layout is:

```text
data/
├── raw/
│   └── source_documents/
└── processed/
    └── corpus.chunks.jsonl

qfr_datasets/
├── taskA.jsonl
├── taskB.jsonl
├── taskA_adversarial.jsonl
└── taskB_adversarial.jsonl

indexes/
outputs/
runs/
```

The `data/`, `indexes/`, `outputs/`, and `runs/` directories are local working directories and should not be committed.

## API and model backends

The generation scripts support OpenAI-compatible API backends and local OpenAI-compatible servers. Some evaluation scripts also support Anthropic judges. For local OpenAI-compatible servers, for example vLLM, use:

```bash
export OPENAI_API_KEY=dummy
export OPENAI_BASE_URL=http://localhost:8000/v1
```

For API-based runs, set the relevant API key in the environment or pass it through the script arguments.

## Reproducing the thesis pipeline

The main thesis experiments used the following standard RAG setting unless stated otherwise:

```text
BM25 + Qwen3 dense retrieval
linear fusion with BM25 weight λ = 0.3
Qwen3 reranking
fixed context depth k = 10
```

The diagnostic generation settings were:

- **standard RAG:** generation from retrieved and reranked evidence;
- **closed-book:** generation without retrieved evidence;
- **oracle:** generation from annotated gold evidence;
- **full-corpus / citation-alias diagnostic:** generation from expanded/full-corpus traces with short prompt-local citation aliases;
- **adversarial:** generation and evaluation on nonsensical, false-premise, and safety-critical variants.

The exact output paths below are examples. You can change them as long as the next step points to the files written by the previous step.

## 1. Preprocessing QFR source documents

Preprocessing scripts are located in:

```text
preprocessing/qfr_dataset/
```

Because the source documents are not redistributed, preprocessing requires local access to the documents.

The retrieval and generation scripts expect a processed corpus file such as:

```text
data/processed/corpus.chunks.jsonl
```

The expected chunk records contain stable chunk identifiers and text fields used by retrieval and citation evaluation.

## 2. Build retrieval indexes

### BM25 index

```bash
python -m src.retrieval.index_bm25 \
  --corpus data/processed/corpus.chunks.jsonl \
  --out_dir indexes/bm25
```

### Dense index

The dense model keys are defined in `src/retrieval/models.py`. The included retrieval model keys are:

```text
qwen3_8b
arctic_m
medcpt
cardioembed
```

Example using Qwen3 embeddings:

```bash
python -m src.retrieval.index_dense \
  --model_key qwen3_8b \
  --corpus data/processed/corpus.chunks.jsonl \
  --index_root indexes/dense \
  --device cuda
```

## 3. Evaluate retrieval

### Task A BM25 retrieval

```bash
python -m src.retrieval.eval_retrieval \
  --dataset A \
  --data_path qfr_datasets/taskA.jsonl \
  --mode bm25 \
  --index_dir indexes/bm25 \
  --k_values 1,3,5,10,20 \
  --out_json outputs/retrieval/taskA_bm25_metrics.json \
  --trace_dir outputs/retrieval/taskA_bm25
```

### Task B BM25 retrieval

```bash
python -m src.retrieval.eval_retrieval \
  --dataset B \
  --data_path qfr_datasets/taskB.jsonl \
  --mode bm25 \
  --index_dir indexes/bm25 \
  --k_values 1,3,5,10,20 \
  --out_json outputs/retrieval/taskB_bm25_metrics.json \
  --trace_dir outputs/retrieval/taskB_bm25
```

### Task A dense retrieval

```bash
python -m src.retrieval.eval_retrieval \
  --dataset A \
  --data_path qfr_datasets/taskA.jsonl \
  --mode dense \
  --index_dir indexes/dense/qwen3_8b \
  --model_key qwen3_8b \
  --device cuda \
  --k_values 1,3,5,10,20 \
  --out_json outputs/retrieval/taskA_qwen3_metrics.json \
  --trace_dir outputs/retrieval/taskA_qwen3
```

### Task B dense retrieval

```bash
python -m src.retrieval.eval_retrieval \
  --dataset B \
  --data_path qfr_datasets/taskB.jsonl \
  --mode dense \
  --index_dir indexes/dense/qwen3_8b \
  --model_key qwen3_8b \
  --device cuda \
  --k_values 1,3,5,10,20 \
  --out_json outputs/retrieval/taskB_qwen3_metrics.json \
  --trace_dir outputs/retrieval/taskB_qwen3
```

## 4. Fusion and reranking

The thesis standard retrieval setting uses BM25 + Qwen3 linear fusion with `--lambda_a 0.3`, followed by Qwen3 reranking.

### Task A fusion

```bash
python -m src.retrieval.fusion_retrieval \
  --dataset A \
  --traces_a outputs/retrieval/taskA_bm25/retrieval_traces.jsonl \
  --traces_b outputs/retrieval/taskA_qwen3/retrieval_traces.jsonl \
  --lambda_a 0.3 \
  --k_values 1,3,5,10,20,50 \
  --out_json outputs/retrieval/taskA_fusion_metrics.json \
  --out_traces_linear outputs/retrieval/taskA_fusion_linear_traces.jsonl \
  --out_traces_rrf outputs/retrieval/taskA_fusion_rrf_traces.jsonl
```

### Task B fusion

```bash
python -m src.retrieval.fusion_retrieval \
  --dataset B \
  --traces_a outputs/retrieval/taskB_bm25/retrieval_traces.jsonl \
  --traces_b outputs/retrieval/taskB_qwen3/retrieval_traces.jsonl \
  --lambda_a 0.3 \
  --k_values 1,3,5,10,20,50 \
  --out_json outputs/retrieval/taskB_fusion_metrics.json \
  --out_traces_linear outputs/retrieval/taskB_fusion_linear_traces.jsonl \
  --out_traces_rrf outputs/retrieval/taskB_fusion_rrf_traces.jsonl
```

### Task A reranking

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

### Task B reranking

```bash
python -m src.retrieval.rerank_retrieval \
  --dataset B \
  --in_traces outputs/retrieval/taskB_fusion_linear_traces.jsonl \
  --out_traces outputs/retrieval/taskB_fusion_reranked_traces.jsonl \
  --chunks_json data/processed/corpus.chunks.jsonl \
  --top_k 50 \
  --k_values 1,3,5,10,20,50 \
  --out_json outputs/retrieval/taskB_fusion_reranked_metrics.json
```

## 5. Standard RAG generation

Standard RAG generation uses reranked retrieval traces and a fixed context depth of `--top_k_context 10`.

### Task A standard RAG

```bash
python -m src.rag_pipeline.gen_from_trace \
  --task A \
  --trace_jsonl outputs/retrieval/taskA_fusion_reranked_traces.jsonl \
  --corpus data/processed/corpus.chunks.jsonl \
  --dataset qfr_datasets/taskA.jsonl \
  --backend openai_compat \
  --lm_model MODEL_NAME \
  --top_k_context 10 \
  --out_dir outputs/generation/taskA/MODEL_NAME_standard
```

### Task B standard RAG

```bash
python -m src.rag_pipeline.gen_from_trace \
  --task B \
  --trace_jsonl outputs/retrieval/taskB_fusion_reranked_traces.jsonl \
  --corpus data/processed/corpus.chunks.jsonl \
  --dataset qfr_datasets/taskB.jsonl \
  --backend openai_compat \
  --lm_model MODEL_NAME \
  --top_k_context 10 \
  --out_dir outputs/generation/taskB/MODEL_NAME_standard
```

Each generation run writes:

```text
generations.jsonl
generation_summary.json
meta.json
```

## 6. Closed-book generation

The closed-book setting answers without retrieved evidence. This is used as a diagnostic baseline to separate retrieval-grounded behaviour from model-internal knowledge.

### Task A closed-book

```bash
python -m src.rag_pipeline.gen_closed_book \
  --task A \
  --dataset qfr_datasets/taskA.jsonl \
  --backend openai_compat \
  --lm_model MODEL_NAME \
  --out_dir outputs/generation/taskA/MODEL_NAME_closed_book
```

### Task B closed-book

```bash
python -m src.rag_pipeline.gen_closed_book \
  --task B \
  --dataset qfr_datasets/taskB.jsonl \
  --backend openai_compat \
  --lm_model MODEL_NAME \
  --out_dir outputs/generation/taskB/MODEL_NAME_closed_book
```

## 7. Oracle-evidence generation

The oracle setting provides annotated gold evidence rather than retrieved evidence. It is used to diagnose whether generation failures remain when retrieval failure is removed.

### Task A oracle

```bash
python -m src.rag_pipeline.gen_oracle \
  --task A \
  --dataset qfr_datasets/taskA.jsonl \
  --corpus data/processed/corpus.chunks.jsonl \
  --oracle_mode all_gold \
  --max_oracle_chunks 10 \
  --backend openai_compat \
  --lm_model MODEL_NAME \
  --out_dir outputs/generation/taskA/MODEL_NAME_oracle
```

### Task B oracle

```bash
python -m src.rag_pipeline.gen_oracle \
  --task B \
  --dataset qfr_datasets/taskB.jsonl \
  --corpus data/processed/corpus.chunks.jsonl \
  --oracle_mode all_gold \
  --max_oracle_chunks 10 \
  --backend openai_compat \
  --lm_model MODEL_NAME \
  --out_dir outputs/generation/taskB/MODEL_NAME_oracle
```

The `all_gold` oracle mode uses all annotated gold evidence up to the context budget. For Task B, this is slot-aware: it prioritises coverage of required evidence slots.

## 8. Full-corpus / citation-alias diagnostic generation

The full-corpus diagnostic uses expanded retrieval traces and replaces long chunk identifiers in the prompt with short local citation aliases such as `C001`, `C002`, and so on. The aliases are mapped back to the original chunk IDs in `generations.jsonl`, so the normal evaluation scripts remain compatible.

This run is useful for diagnosing citation and context-use behaviour under a much larger evidence set. The script expects a trace file whose `retrieved` list already contains the chunks to expose to the model.

```bash
python -m src.rag_pipeline.gen_full_corpus_alias \
  --task B \
  --trace_jsonl outputs/retrieval/taskB_full_corpus_traces.jsonl \
  --corpus data/processed/corpus.chunks.jsonl \
  --dataset qfr_datasets/taskB.jsonl \
  --backend openai_compat \
  --lm_model MODEL_NAME \
  --top_k_context 200 \
  --out_dir outputs/generation/taskB/MODEL_NAME_full_corpus_alias
```

Use a `--top_k_context` value that fits the context window of the model being evaluated.

## 9. Generation evaluation

The evaluation scripts consume `generations.jsonl` and write per-example and global evaluation outputs.

### Task A evaluation

```bash
python -m src.rag_pipeline.evaluate_taskA \
  --task A \
  --generations outputs/generation/taskA/MODEL_NAME_standard/generations.jsonl \
  --dataset qfr_datasets/taskA.jsonl \
  --out_dir outputs/evaluation/taskA/MODEL_NAME_standard/JUDGE_NAME \
  --backend openai_compat \
  --lm_model JUDGE_MODEL_NAME
```

Task A evaluation writes, among other files:

```text
per_example_eval.jsonl
global_summary.json
taxonomy_summary.json
eval_meta.json
```

### Task B evaluation

```bash
python -m src.rag_pipeline.evaluation \
  --task B \
  --generations outputs/generation/taskB/MODEL_NAME_standard/generations.jsonl \
  --dataset qfr_datasets/taskB.jsonl \
  --out_dir outputs/evaluation/taskB/MODEL_NAME_standard/JUDGE_NAME \
  --run_nuggets \
  --backend openai_compat \
  --lm_model JUDGE_MODEL_NAME
```

To include optional RAGAS-style faithfulness and answer-relevance metrics, add:

```bash
--run_ragas
```

Task B evaluation writes:

```text
per_example_eval.jsonl
global_summary.json
nugget_summary.json
eval_meta.json
```

The thesis used GPT-4o-mini and Claude Haiku 4.5 as judge models and reports averaged judge scores where applicable.

## 10. BioASQ validation runs

BioASQ was used as a pipeline validation check. The BioASQ scripts are separate from the QFR-RAG task scripts because BioASQ uses document-level evidence and answer formats.

BioASQ preprocessing utilities are located in:

```text
preprocessing/bioasq_dataset/
```

Trace-driven BioASQ generation:

```bash
python -m src.rag_pipeline.bioasq_gen_from_trace \
  --trace_jsonl outputs/bioasq/retrieval/bioasq_fusion_traces.jsonl \
  --corpus data/bioasq/processed/docs.jsonl \
  --backend openai_compat \
  --lm_model MODEL_NAME \
  --top_k_context 10 \
  --out_dir outputs/bioasq/generation/MODEL_NAME_standard
```

BioASQ evaluation:

```bash
python -m src.rag_pipeline.eval_bioasq \
  --generations outputs/bioasq/generation/MODEL_NAME_standard/generations.jsonl \
  --gold_answers data/bioasq/eval/gold_answers.jsonl \
  --out_dir outputs/bioasq/evaluation/MODEL_NAME_standard/JUDGE_NAME \
  --backend openai_compat \
  --lm_model JUDGE_MODEL_NAME
```

## 11. Adversarial runs

The adversarial set evaluates whether systems avoid inappropriate compliance on nonsensical, false-premise, and safety-critical questions.

### Adversarial retrieval

```bash
python -m src.retrieval.run_adversarial_retrieval \
  --taskA_dataset qfr_datasets/taskA_adversarial.jsonl \
  --taskB_dataset qfr_datasets/taskB_adversarial.jsonl \
  --chunks_json data/processed/corpus.chunks.jsonl \
  --bm25_index_dir indexes/bm25 \
  --dense_index_dir indexes/dense/qwen3_8b \
  --dense_model_key qwen3_8b \
  --lambda_a 0.3 \
  --retrieve_k 50 \
  --rerank_top_k 50 \
  --out_root outputs/adversarial/retrieval
```

### Adversarial generation

Use the adversarial retrieval traces as input to the same trace-driven generation script:

```bash
python -m src.rag_pipeline.gen_from_trace \
  --task B \
  --trace_jsonl outputs/adversarial/retrieval/taskB/retrieval/rerank_qwen3_fusion_lambda0.3/retrieval_traces.jsonl \
  --corpus data/processed/corpus.chunks.jsonl \
  --dataset qfr_datasets/taskB_adversarial.jsonl \
  --backend openai_compat \
  --lm_model MODEL_NAME \
  --top_k_context 10 \
  --out_dir outputs/adversarial/generation/taskB/MODEL_NAME
```

Run the same command with `--task A` and the Task A adversarial files for Task-A-derived adversarial questions.

### Adversarial evaluation

```bash
python -m src.adversarial.eval_adversarial \
  --task B \
  --dataset qfr_datasets/taskB_adversarial.jsonl \
  --generations outputs/adversarial/generation/taskB/MODEL_NAME/generations.jsonl \
  --out_dir outputs/adversarial/evaluation/taskB/MODEL_NAME/JUDGE_NAME \
  --judge_model JUDGE_MODEL_NAME
```

This writes:

```text
adversarial_eval_per_item.csv
adversarial_eval_by_group.csv
```

## 12. Inter-judge reliability

The reliability script is intentionally generic. It does not guess model names, suffixes, or judge folders. Instead, it compares explicit judge-output files or directories.

Direct example for Task A:

```bash
python -m src.rag_pipeline.inter_judge_reliability \
  --judge-a outputs/evaluation/taskA/MODEL_NAME_standard/gpt4o_mini/per_example_eval.jsonl \
  --judge-b outputs/evaluation/taskA/MODEL_NAME_standard/haiku45/per_example_eval.jsonl \
  --name taskA_MODEL_NAME_standard \
  --id-field id \
  --categorical-metrics taskA_judge.judge_correct taskA_taxonomy.taxonomy_case \
  --out-dir outputs/inter_judge_reliability/taskA_MODEL_NAME_standard
```

Direct example for Task B:

```bash
python -m src.rag_pipeline.inter_judge_reliability \
  --judge-a outputs/evaluation/taskB/MODEL_NAME_standard/gpt4o_mini/per_example_eval.jsonl \
  --judge-b outputs/evaluation/taskB/MODEL_NAME_standard/haiku45/per_example_eval.jsonl \
  --name taskB_MODEL_NAME_standard \
  --id-field id \
  --scalar-metrics nuggets.macro_avg_strict_recall nuggets.macro_avg_soft_recall \
  --categorical-metrics nuggets.question_taxonomy_case \
  --out-dir outputs/inter_judge_reliability/taskB_MODEL_NAME_standard
```

For many runs, use a manifest CSV:

```csv
name,judge_a,judge_b,id_field,scalar_metrics,categorical_metrics,subgroup_fields
taskA_gpt4o_standard,outputs/evaluation/taskA/gpt4o_standard/gpt4o_mini/per_example_eval.jsonl,outputs/evaluation/taskA/gpt4o_standard/haiku45/per_example_eval.jsonl,id,,"taskA_judge.judge_correct,taskA_taxonomy.taxonomy_case",
taskB_gpt4o_standard,outputs/evaluation/taskB/gpt4o_standard/gpt4o_mini/per_example_eval.jsonl,outputs/evaluation/taskB/gpt4o_standard/haiku45/per_example_eval.jsonl,id,"nuggets.macro_avg_strict_recall,nuggets.macro_avg_soft_recall",nuggets.question_taxonomy_case,
advB_gpt4o,outputs/adversarial/evaluation/taskB/gpt4o/gpt4o_mini/adversarial_eval_per_item.csv,outputs/adversarial/evaluation/taskB/gpt4o/haiku45/adversarial_eval_per_item.csv,id,,"acceptable,paper_behavior,primary_failure_mode",adversarial_category
```

Run:

```bash
python -m src.rag_pipeline.inter_judge_reliability \
  --manifest outputs/inter_judge_reliability/manifest.csv \
  --out-dir outputs/inter_judge_reliability
```

The reliability script computes:

- exact agreement;
- disagreement rate;
- Cohen's kappa;
- PABAK;
- Pearson correlation for scalar metrics;
- Spearman correlation for scalar metrics;
- mean absolute difference;
- categorical disagreement files;
- scalar large-disagreement files.

## Output conventions

Most generation scripts write:

```text
generations.jsonl
generation_summary.json
meta.json
```

Most evaluation scripts write:

```text
per_example_eval.jsonl
global_summary.json
eval_meta.json
```

Adversarial evaluation writes CSV files, and inter-judge reliability writes summary CSV/JSON files plus disagreement files.

## Reproducibility notes

When reporting QFR-RAG results, specify:

- the source corpus and preprocessing version;
- the chunking strategy;
- the retriever and index used;
- fusion method and fusion weight;
- reranker model and reranking depth;
- final context depth used for generation;
- generation model and backend;
- prompt setting;
- whether the run is standard RAG, closed-book, oracle, full-corpus, or adversarial;
- judge model(s) used for evaluation;
- whether scores are single-judge or averaged across judges.

Because the original QFR source documents are not redistributed, corpus-dependent retrieval and generation results may vary if users reconstruct the corpus differently.

## Intended use

QFR-RAG is intended for research on:

- evidence retrieval;
- reranking;
- grounded question answering;
- citation grounding;
- answer completeness;
- abstention behaviour;
- adversarial robustness;
- technical-medical RAG evaluation.

The resource is intended as a diagnostic evaluation benchmark, not as a general medical QA dataset.

## Out-of-scope use

QFR-RAG should not be used as a source of medical advice, clinical decision support, or procedural guidance. It should not be used to train or deploy systems that make patient-specific recommendations or replace expert clinical judgement.

The dataset should also not be treated as a complete representation of QFR, coronary physiology, or QFR software use. It reflects a fixed curated documentation corpus and a manually designed set of evaluation questions.

## License

The code in this repository is released under the MIT License.

The QFR-RAG dataset files are hosted separately on Hugging Face and are released under the Creative Commons Attribution-NonCommercial 4.0 International license.

The dataset license applies only to the released QFR-RAG dataset files, such as the questions, annotations, labels, and metadata. It does not apply to the original QFR source documents, software manuals, publications, or other third-party materials from which the evaluation resource was curated.

The original QFR source documents are not redistributed in this repository or in the Hugging Face dataset.

## Citation

A citation will be added once the associated thesis or paper is publicly available.

If you use this repository or dataset before a formal publication is available, please cite the GitHub repository and the Hugging Face dataset URL.

## Author

Emma Bakker
