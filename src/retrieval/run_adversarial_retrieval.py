#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np

from .determinism import set_deterministic
from .retrieval import BM25Retriever, DenseRetriever
from .models import Qwen3Reranker

logger = logging.getLogger(__name__)


# IO

def read_text_robust(path: str | Path) -> str:
    """Read text robustly. Keeps scripts usable when a JSONL has cp1252 smart quotes."""
    p = Path(path)
    raw = p.read_bytes()
    for enc in ("utf-8", "utf-8-sig", "cp1252", "latin-1"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    # latin-1 should always decode, but keep a final fallback.
    return raw.decode("utf-8", errors="replace")


def iter_jsonl(path: str | Path) -> Iterable[Dict[str, Any]]:
    text = read_text_robust(path)
    for line_no, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            yield json.loads(line)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in {path} at line {line_no}: {e}") from e


def write_jsonl(path: str | Path, rows: Iterable[Dict[str, Any]]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: str | Path, obj: Dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def load_chunk_texts(path: str | Path) -> Dict[str, str]:
    p = Path(path)
    texts: Dict[str, str] = {}

    if p.suffix.lower() == ".jsonl":
        for obj in iter_jsonl(p):
            cid = obj.get("chunk_id")
            txt = obj.get("text")
            if cid is not None and txt is not None:
                texts[str(cid)] = str(txt)
    else:
        obj = json.loads(read_text_robust(p))
        if not isinstance(obj, dict):
            raise ValueError(f"Expected JSON object mapping chunk_id -> text in {p}")
        for cid, txt in obj.items():
            texts[str(cid)] = str(txt)

    if not texts:
        raise ValueError(f"No chunk texts loaded from {p}")
    return texts


# Fusion

def normalize_scores(d: Dict[str, float]) -> Dict[str, float]:
    if not d:
        return {}
    vals = np.array(list(d.values()), dtype=float)
    vmin = float(vals.min())
    vmax = float(vals.max())
    if vmax <= vmin:
        return {cid: 0.0 for cid in d.keys()}
    return {cid: float((score - vmin) / (vmax - vmin)) for cid, score in d.items()}


def linear_fusion(
    hits_a: List[Tuple[str, float]],
    hits_b: List[Tuple[str, float]],
    lambda_a: float,
    k: int,
) -> List[Tuple[str, float]]:
    scores_a = {str(cid): float(score) for cid, score in hits_a}
    scores_b = {str(cid): float(score) for cid, score in hits_b}

    norm_a = normalize_scores(scores_a)
    norm_b = normalize_scores(scores_b)

    fused: List[Tuple[str, float]] = []
    for cid in set(norm_a.keys()) | set(norm_b.keys()):
        score = lambda_a * norm_a.get(cid, 0.0) + (1.0 - lambda_a) * norm_b.get(cid, 0.0)
        fused.append((cid, float(score)))

    fused.sort(key=lambda x: x[1], reverse=True)
    return fused[:k]


# Trace construction

def build_trace_record(
    dataset_row: Dict[str, Any],
    run_name: str,
    mode: str,
    model_key: str | None,
    index_dir: str | None,
    seed: int,
    retrieve_k: int,
    retrieved_hits: List[Tuple[str, float]],
) -> Dict[str, Any]:
    """
    Keep the adversarial row metadata in the trace.
    This is useful later for generation/error analysis, but does not require base datasets.
    """
    qid = str(dataset_row.get("id", "")).strip()
    question = str(dataset_row.get("question", "")).strip()

    rec: Dict[str, Any] = dict(dataset_row)
    rec.update(
        {
            "query_id": qid,
            "question": question,
            "run": run_name,
            "mode": mode,
            "model_key": model_key,
            "index_dir": index_dir,
            "seed": int(seed),
            "retrieve_k": int(retrieve_k),
            "retrieved": [
                {"rank": i + 1, "chunk_id": str(cid), "score": float(score)}
                for i, (cid, score) in enumerate(retrieved_hits)
            ],
        }
    )
    return rec


def run_retrieval_for_dataset(
    dataset_path: str,
    retriever,
    run_name: str,
    mode: str,
    model_key: str | None,
    index_dir: str | None,
    seed: int,
    retrieve_k: int,
) -> List[Dict[str, Any]]:
    rows = list(iter_jsonl(dataset_path))
    traces: List[Dict[str, Any]] = []

    logger.info("Loaded %d rows from %s", len(rows), dataset_path)

    for row in rows:
        qid = str(row.get("id", "")).strip()
        question = str(row.get("question", "")).strip()
        if not qid or not question:
            logger.warning("Skipping row without id/question: %s", row)
            continue

        hits = retriever.search(question, k=retrieve_k)
        traces.append(
            build_trace_record(
                dataset_row=row,
                run_name=run_name,
                mode=mode,
                model_key=model_key,
                index_dir=index_dir,
                seed=seed,
                retrieve_k=retrieve_k,
                retrieved_hits=hits,
            )
        )

    return traces


def fuse_trace_lists(
    traces_a: List[Dict[str, Any]],
    traces_b: List[Dict[str, Any]],
    lambda_a: float,
    retrieve_k: int,
    seed: int,
) -> List[Dict[str, Any]]:
    by_id_a = {str(x["query_id"]): x for x in traces_a}
    by_id_b = {str(x["query_id"]): x for x in traces_b}
    common_ids = sorted(set(by_id_a.keys()) & set(by_id_b.keys()))

    fused_rows: List[Dict[str, Any]] = []
    for qid in common_ids:
        ta = by_id_a[qid]
        tb = by_id_b[qid]

        hits_a = [(r["chunk_id"], float(r["score"])) for r in ta.get("retrieved", [])]
        hits_b = [(r["chunk_id"], float(r["score"])) for r in tb.get("retrieved", [])]

        fused = linear_fusion(hits_a=hits_a, hits_b=hits_b, lambda_a=lambda_a, k=retrieve_k)

        # Preserve adversarial metadata from the BM25 trace row, overwrite retrieval-run fields.
        out: Dict[str, Any] = dict(ta)
        out.update(
            {
                "query_id": qid,
                "question": ta.get("question", ""),
                "run": f"linear_fusion::bm25_qwen3_lambda{lambda_a}",
                "mode": "fusion_linear",
                "model_key": None,
                "index_dir": None,
                "seed": int(seed),
                "retrieve_k": int(retrieve_k),
                "fusion": {
                    "method": "linear_minmax",
                    "lambda_a": float(lambda_a),
                    "a": "bm25",
                    "b": "qwen3",
                },
                "retrieved": [
                    {"rank": i + 1, "chunk_id": str(cid), "score": float(score)}
                    for i, (cid, score) in enumerate(fused)
                ],
            }
        )
        fused_rows.append(out)

    if len(common_ids) != len(traces_a) or len(common_ids) != len(traces_b):
        logger.warning(
            "Fusion used %d common query_ids, with %d BM25 traces and %d dense traces",
            len(common_ids),
            len(traces_a),
            len(traces_b),
        )

    return fused_rows


# Reranking traces only: no gold labels, no metrics, no base datasets.

def rerank_trace_rows(
    traces: List[Dict[str, Any]],
    chunk_texts: Dict[str, str],
    reranker: Qwen3Reranker,
    top_k: int,
    batch_size: int,
    task_name: str,
    lambda_a: float,
) -> List[Dict[str, Any]]:
    out_rows: List[Dict[str, Any]] = []
    latencies: List[float] = []

    for i, rec in enumerate(traces, start=1):
        question = str(rec.get("question", "")).strip()
        retrieved = rec.get("retrieved", []) or []

        if not question or not retrieved:
            logger.warning("Skipping rerank row without question/retrieved: %s", rec.get("query_id"))
            continue

        top_hits = retrieved[:top_k]
        queries: List[str] = []
        passages: List[str] = []
        candidate_rows: List[Dict[str, Any]] = []

        for hit in top_hits:
            cid = str(hit.get("chunk_id", ""))
            text = chunk_texts.get(cid)
            if not cid or text is None:
                continue
            queries.append(question)
            passages.append(text)
            candidate_rows.append(dict(hit))

        if not candidate_rows:
            logger.warning("No rerankable candidates for query_id=%s", rec.get("query_id"))
            continue

        t0 = time.time()
        scores = reranker.score_pairs(queries, passages, batch_size=batch_size)
        elapsed = time.time() - t0
        latencies.append(float(elapsed))

        scored = []
        for hit, score in zip(candidate_rows, scores):
            cid = str(hit["chunk_id"])
            scored.append((cid, float(score), hit))
        scored.sort(key=lambda x: x[1], reverse=True)

        reranked = []
        for rank, (cid, score, old_hit) in enumerate(scored, start=1):
            new_hit = dict(old_hit)
            new_hit.update(
                {
                    "rank": int(rank),
                    "chunk_id": cid,
                    "score": float(score),
                    "score_type": "qwen3_reranker_yes_probability",
                    "pre_rerank_rank": old_hit.get("rank"),
                    "pre_rerank_score": old_hit.get("score"),
                }
            )
            reranked.append(new_hit)

        out = dict(rec)
        out.update(
            {
                "run": f"qwen3_rerank::fusion_lambda{lambda_a}",
                "mode": "rerank_qwen3_over_fusion",
                "model_key": "qwen3_reranker",
                "retrieved": reranked,
                "rerank": {
                    "input_run": rec.get("run"),
                    "input_mode": rec.get("mode"),
                    "reranker": "Qwen/Qwen3-Reranker-0.6B",
                    "top_k": int(top_k),
                    "batch_size": int(batch_size),
                    "n_candidates_scored": int(len(reranked)),
                    "latency_sec": float(elapsed),
                },
            }
        )
        out_rows.append(out)

        if i % 10 == 0:
            logger.info("Reranked %s: %d/%d queries", task_name, i, len(traces))

    if latencies:
        logger.info(
            "Reranked %s: %d queries, mean latency %.4f sec/query",
            task_name,
            len(out_rows),
            float(np.mean(latencies)),
        )

    return out_rows


# Main

def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Run adversarial retrieval traces for Task A and Task B: BM25, Qwen3 dense, "
            "lambda fusion, and Qwen3 reranked fusion. This script does NOT require base datasets."
        )
    )
    ap.add_argument("--taskA_dataset", required=True, help="Task A adversarial JSONL")
    ap.add_argument(
        "--taskB_dataset",
        default=None,
        help="Task B adversarial JSONL",
    )
    ap.add_argument(
        "--taskB_dataset",
        default=None,
        help="[DEPRECATED] Alias for --taskB_dataset. Use --taskB_dataset instead.",
    )
    ap.add_argument("--chunks_json", required=True, help="Corpus chunks JSONL with chunk_id and text")

    ap.add_argument("--bm25_index_dir", required=True, help="BM25 index directory")
    ap.add_argument("--dense_index_dir", required=True, help="Dense Qwen3 index directory")
    ap.add_argument("--dense_model_key", default="qwen3_8b", help="Dense model key")

    ap.add_argument("--out_root", required=True, help="Root output directory")
    ap.add_argument("--retrieve_k", type=int, default=50, help="Top-k retrieval depth")
    ap.add_argument("--rerank_top_k", type=int, default=50, help="How many fusion candidates to rerank")
    ap.add_argument("--lambda_a", type=float, default=0.3, help="Weight for BM25 in linear fusion")
    ap.add_argument("--seed", type=int, default=224)
    ap.add_argument("--device", default=None, help="cpu or cuda for dense retrieval")
    ap.add_argument("--rerank_device", default="cuda", help="cpu or cuda for reranker")
    ap.add_argument("--batch_size", type=int, default=8, help="Batch size for reranker")
    ap.add_argument("--skip_rerank", action="store_true", help="Only write BM25, Qwen3 and fusion traces")
    args = ap.parse_args()

    # Resolve the Task B dataset path, accepting the deprecated --taskB_dataset alias.
    taskB_dataset: str | None = args.taskB_dataset
    if taskB_dataset is None and args.taskB_dataset is not None:
        logger.warning(
            "--taskB_dataset is deprecated; please use --taskB_dataset instead."
        )
        taskB_dataset = args.taskB_dataset
    if taskB_dataset is None:
        ap.error("--taskB_dataset is required (or its deprecated alias --taskB_dataset)")

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    set_deterministic(args.seed, deterministic_torch=False)

    out_root = Path(args.out_root)

    logger.info("Loading BM25 retriever")
    bm25 = BM25Retriever(args.bm25_index_dir)

    logger.info("Loading dense retriever")
    dense = DenseRetriever(
        model_key=args.dense_model_key,
        index_dir=args.dense_index_dir,
        device=args.device,
    )

    chunk_texts: Dict[str, str] = {}
    reranker = None
    if not args.skip_rerank:
        logger.info("Loading chunk texts from %s", args.chunks_json)
        chunk_texts = load_chunk_texts(args.chunks_json)
        logger.info("Loading Qwen3 reranker")
        reranker = Qwen3Reranker(device=args.rerank_device)

    configs = [
        ("taskA", args.taskA_dataset),
        ("taskB", taskB_dataset),
    ]

    summary: Dict[str, Any] = {
        "taskA_dataset": args.taskA_dataset,
        "taskB_dataset": taskB_dataset,
        "chunks_json": args.chunks_json,
        "bm25_index_dir": args.bm25_index_dir,
        "dense_index_dir": args.dense_index_dir,
        "dense_model_key": args.dense_model_key,
        "retrieve_k": int(args.retrieve_k),
        "rerank_top_k": int(args.rerank_top_k),
        "lambda_a": float(args.lambda_a),
        "seed": int(args.seed),
        "outputs": {},
    }

    for task_name, dataset_path in configs:
        logger.info("=== Running adversarial retrieval for %s ===", task_name)

        task_root = out_root / task_name / "retrieval"
        bm25_trace_path = task_root / "bm25" / "retrieval_traces.jsonl"
        dense_trace_path = task_root / "qwen3" / "retrieval_traces.jsonl"
        fused_trace_path = task_root / "fusion" / "linear_traces.jsonl"
        rerank_dir = task_root / f"rerank_qwen3_fusion_lambda{args.lambda_a}"
        rerank_trace_path = rerank_dir / "retrieval_traces.jsonl"

        bm25_traces = run_retrieval_for_dataset(
            dataset_path=dataset_path,
            retriever=bm25,
            run_name=f"bm25::{args.bm25_index_dir}",
            mode="bm25",
            model_key=None,
            index_dir=args.bm25_index_dir,
            seed=args.seed,
            retrieve_k=args.retrieve_k,
        )
        write_jsonl(bm25_trace_path, bm25_traces)
        logger.info("Wrote BM25 traces: %s", bm25_trace_path)

        dense_traces = run_retrieval_for_dataset(
            dataset_path=dataset_path,
            retriever=dense,
            run_name=f"dense::{args.dense_model_key}::{args.dense_index_dir}",
            mode="dense",
            model_key=args.dense_model_key,
            index_dir=args.dense_index_dir,
            seed=args.seed,
            retrieve_k=args.retrieve_k,
        )
        write_jsonl(dense_trace_path, dense_traces)
        logger.info("Wrote dense traces: %s", dense_trace_path)

        fused_traces = fuse_trace_lists(
            traces_a=bm25_traces,
            traces_b=dense_traces,
            lambda_a=args.lambda_a,
            retrieve_k=args.retrieve_k,
            seed=args.seed,
        )
        write_jsonl(fused_trace_path, fused_traces)
        logger.info("Wrote linear fused traces: %s", fused_trace_path)

        task_outputs: Dict[str, Any] = {
            "bm25_traces": bm25_trace_path.as_posix(),
            "dense_traces": dense_trace_path.as_posix(),
            "linear_fused_traces": fused_trace_path.as_posix(),
            "n_queries": len(fused_traces),
        }

        if not args.skip_rerank:
            assert reranker is not None
            reranked_traces = rerank_trace_rows(
                traces=fused_traces,
                chunk_texts=chunk_texts,
                reranker=reranker,
                top_k=args.rerank_top_k,
                batch_size=args.batch_size,
                task_name=task_name,
                lambda_a=args.lambda_a,
            )
            write_jsonl(rerank_trace_path, reranked_traces)
            logger.info("Wrote Qwen3 reranked fusion traces: %s", rerank_trace_path)
            task_outputs["reranked_fusion_traces"] = rerank_trace_path.as_posix()
            task_outputs["n_queries_reranked"] = len(reranked_traces)

        summary["outputs"][task_name] = task_outputs

    summary_path = out_root / "adversarial_retrieval_summary.json"
    write_json(summary_path, summary)
    logger.info("Wrote summary JSON: %s", summary_path)

    print("\n=== Done ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
