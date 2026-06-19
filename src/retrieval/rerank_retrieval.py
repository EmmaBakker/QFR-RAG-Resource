#!/usr/bin/env python3
"""
Examples
--------
# Task A
python -m src.retrieval.rerank_retrieval --dataset A \\
    --in_traces  outputs/qfr/bm25/retrieval_traces.jsonl \\
    --out_traces outputs/qfr/rerank_A/retrieval_traces.jsonl \\
    --chunks_json Data/processed/corpus.chunks.jsonl \\
    --top_k 50 --batch_size 8 \\
    --k_values 1,3,5,10,20,50 \\
    --out_json outputs/qfr/rerank_A/metrics.json

# Task B
python -m src.retrieval.rerank_retrieval --dataset B \\
    --in_traces  outputs/qfr/bm25/retrieval_traces.jsonl \\
    --out_traces outputs/qfr/rerank_B/retrieval_traces.jsonl \\
    --chunks_json Data/processed/corpus.chunks.jsonl \\
    --top_k 50 --batch_size 8 \\
    --k_values 1,3,5,10,20,50 \\
    --out_json outputs/qfr/rerank_B/metrics.json

# BioASQ
python -m src.retrieval.rerank_retrieval --dataset bioasq \\
    --in_traces  outputs/bioasq/bm25/retrieval_traces.jsonl \\
    --out_traces outputs/bioasq/rerank/retrieval_traces.jsonl \\
    --docs_json  Data/bioasq/processed/docs.jsonl \\
    --top_k 50 --batch_size 8 \\
    --k_values 1,3,5,10,20,50 \\
    --out_json outputs/bioasq/rerank/metrics.json
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import numpy as np



# ---------------------------------------------------------------------------
# Shared IO helpers
# ---------------------------------------------------------------------------

def _iter_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _load_traces(path: str) -> List[Dict[str, Any]]:
    """Load any retrieval_traces.jsonl as a plain list of dicts."""
    return list(_iter_jsonl(path))


def _load_chunk_texts(path: str) -> Dict[str, str]:
    """
    Load chunk_id -> text mapping.
    Supports JSONL (fields: chunk_id, text) or flat JSON dict.
    """
    texts: Dict[str, str] = {}
    p = Path(path)
    if p.suffix.lower() == ".jsonl":
        for obj in _iter_jsonl(path):
            cid = obj.get("chunk_id")
            txt = obj.get("text")
            if cid is not None and txt is not None:
                texts[str(cid)] = str(txt)
    else:
        obj = json.loads(p.read_text(encoding="utf-8"))
        for cid, txt in obj.items():
            texts[str(cid)] = str(txt)
    return texts


def _load_doc_texts(path: str) -> Dict[str, str]:
    """
    Load doc_id -> text mapping.
    Supports JSONL (fields: doc_id, text) or flat JSON dict.
    """
    texts: Dict[str, str] = {}
    p = Path(path)
    if p.suffix.lower() == ".jsonl":
        for obj in _iter_jsonl(path):
            did = obj.get("doc_id")
            txt = obj.get("text")
            if did is not None and txt is not None:
                texts[str(did)] = str(txt)
    else:
        obj = json.loads(p.read_text(encoding="utf-8"))
        for did, txt in obj.items():
            texts[str(did)] = str(txt)
    return texts


def _write_json(path: str, obj: Any) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# Shared metric helpers
# ---------------------------------------------------------------------------

def _mrr_at_k(ranked_ids: List[str], gold_set: Set[str], k: int) -> float:
    for i, cid in enumerate(ranked_ids[:k], start=1):
        if cid in gold_set:
            return 1.0 / i
    return 0.0


def _ndcg_at_k_binary(ranked_ids: List[str], gold_set: Set[str], k: int) -> float:
    dcg = 0.0
    for i, cid in enumerate(ranked_ids[:k], start=1):
        if cid in gold_set:
            dcg += 1.0 / np.log2(i + 1)
    m = min(len(gold_set), k)
    if m == 0:
        return 0.0
    idcg = sum(1.0 / np.log2(i + 1) for i in range(1, m + 1))
    return float(dcg / idcg)


# ---------------------------------------------------------------------------
# Task A
# ---------------------------------------------------------------------------

def _hit_recall_at_k(ranked_ids: List[str], gold_set: Set[str], k: int) -> float:
    for cid in ranked_ids[:k]:
        if cid in gold_set:
            return 1.0
    return 0.0


def _set_recall_at_k(ranked_ids: List[str], gold_set: Set[str], k: int) -> float:
    if not gold_set:
        return 0.0
    return float(len(set(ranked_ids[:k]).intersection(gold_set)) / len(gold_set))


def rerank_taskA(args: argparse.Namespace) -> None:
    """Replicate rerank_taskA.py behaviour exactly."""
    k_values = sorted({int(x.strip()) for x in args.k_values.split(",") if x.strip()})

    records = _load_traces(args.in_traces)
    chunk_texts = _load_chunk_texts(args.chunks_json)

    from .models import Qwen3Reranker  # deferred: keeps --help torch-free
    reranker = Qwen3Reranker()

    agg: Dict[str, List[float]] = {f"MRR@{k}": [] for k in k_values}
    agg.update({f"HitRecall@{k}": [] for k in k_values})
    agg.update({f"SetRecall@{k}": [] for k in k_values})
    agg.update({f"nDCG@{k}": [] for k in k_values})
    agg["latency_sec"] = []

    out_path = Path(args.out_traces)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_queries = 0
    with out_path.open("w", encoding="utf-8") as fout:
        for rec in records:
            qid = rec.get("query_id")
            question = rec.get("question", "")
            retrieved = rec.get("retrieved", []) or []
            gold: Set[str] = set(rec.get("gold_chunk_ids", []) or [])

            if not qid or not question or not retrieved:
                continue

            top_hits = retrieved[: args.top_k]

            queries: List[str] = []
            passages: List[str] = []
            hit_ids: List[str] = []
            for r in top_hits:
                cid = str(r["chunk_id"])
                text = chunk_texts.get(cid)
                if text is None:
                    continue
                queries.append(question)
                passages.append(text)
                hit_ids.append(cid)

            if not hit_ids:
                continue

            t0 = time.time()
            scores = reranker.score_pairs(queries, passages, batch_size=args.batch_size)
            elapsed = time.time() - t0
            agg["latency_sec"].append(float(elapsed))

            scored_sorted = sorted(zip(hit_ids, scores), key=lambda x: x[1], reverse=True)

            reranked = [
                {"rank": rank, "chunk_id": cid, "score": float(score)}
                for rank, (cid, score) in enumerate(scored_sorted, start=1)
            ]
            ranked_ids = [cid for cid, _ in scored_sorted]

            for k in k_values:
                agg[f"MRR@{k}"].append(_mrr_at_k(ranked_ids, gold, k))
                agg[f"HitRecall@{k}"].append(_hit_recall_at_k(ranked_ids, gold, k))
                agg[f"SetRecall@{k}"].append(_set_recall_at_k(ranked_ids, gold, k))
                agg[f"nDCG@{k}"].append(_ndcg_at_k_binary(ranked_ids, gold, k))

            out_rec: Dict[str, Any] = {
                "query_id": qid,
                "question": question,
                "gold_chunk_ids": list(gold),
                "retrieved": reranked,
            }
            for key, val in rec.items():
                if key not in out_rec:
                    out_rec[key] = val

            fout.write(json.dumps(out_rec, ensure_ascii=False) + "\n")
            n_queries += 1

            if n_queries % 10 == 0:
                print(f"Reranked {n_queries}/{len(records)} queries", flush=True)

    mean_metrics = {k: float(np.mean(v)) if v else 0.0 for k, v in agg.items()}

    print(f"\n=== TASK A RERANK (Qwen3) EVAL ===")
    print(f"Input traces : {args.in_traces}")
    print(f"Output traces: {out_path.as_posix()}")
    print(f"Queries      : {n_queries}")
    print(f"mean latency : {mean_metrics['latency_sec']:.6f} sec/query\n")

    for k in k_values:
        print(
            f"@{k:>2}  "
            f"MRR={mean_metrics[f'MRR@{k}']:.4f}  "
            f"HitRecall={mean_metrics[f'HitRecall@{k}']:.4f}  "
            f"SetRecall={mean_metrics[f'SetRecall@{k}']:.4f}  "
            f"nDCG={mean_metrics[f'nDCG@{k}']:.4f}"
        )

    if args.out_json:
        _write_json(args.out_json, {
            "in_traces": args.in_traces,
            "out_traces": out_path.as_posix(),
            "k_values": k_values,
            "metrics": mean_metrics,
            "n_queries_scored": n_queries,
        })
        print(f"\nWrote metrics JSON: {args.out_json}")


# ---------------------------------------------------------------------------
# Task B
# ---------------------------------------------------------------------------

@dataclass
class _PerKAccum:
    q_recall_sum: float = 0.0
    slot_recall_sum: float = 0.0
    slot_mrr_sum: float = 0.0


def _first_hit_rank_1based(retrieved_ids: List[str], gold_set: Set[str]) -> Optional[int]:
    for i, cid in enumerate(retrieved_ids, start=1):
        if cid in gold_set:
            return i
    return None


def rerank_taskB(args: argparse.Namespace) -> None:
    """Replicate rerank_taskB.py behaviour exactly."""
    k_values = sorted({int(x.strip()) for x in args.k_values.split(",") if x.strip()})

    records = _load_traces(args.in_traces)
    chunk_texts = _load_chunk_texts(args.chunks_json)

    from .models import Qwen3Reranker  # deferred: keeps --help torch-free
    reranker = Qwen3Reranker()

    acc: Dict[int, _PerKAccum] = {k: _PerKAccum() for k in k_values}
    latencies: List[float] = []

    out_path = Path(args.out_traces)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_queries = 0
    with out_path.open("w", encoding="utf-8") as fout:
        for rec in records:
            qid = rec.get("query_id")
            question = rec.get("question", "")
            retrieved = rec.get("retrieved", []) or []
            slots_raw = rec.get("slots", []) or []
            required_slots = rec.get("required_slots", None)

            if not qid or not question or not retrieved:
                continue

            slots: List[Tuple[str, Set[str]]] = []
            for s in slots_raw:
                lab = s.get("label")
                gold_ids = s.get("gold_ids", []) or []
                if lab and gold_ids:
                    slots.append((lab, set(str(x) for x in gold_ids)))

            if required_slots is not None:
                try:
                    required_slots_n = int(required_slots)
                    slots = slots[:required_slots_n]
                except Exception:
                    pass

            if not slots:
                continue

            top_hits = retrieved[: args.top_k]

            queries: List[str] = []
            passages: List[str] = []
            hit_ids: List[str] = []
            for r in top_hits:
                cid = str(r["chunk_id"])
                text = chunk_texts.get(cid)
                if text is None:
                    continue
                queries.append(question)
                passages.append(text)
                hit_ids.append(cid)

            if not hit_ids:
                continue

            t0 = time.time()
            scores = reranker.score_pairs(queries, passages, batch_size=args.batch_size)
            elapsed = time.time() - t0
            latencies.append(float(elapsed))

            scored_sorted = sorted(zip(hit_ids, scores), key=lambda x: x[1], reverse=True)

            reranked = [
                {"rank": rank, "chunk_id": cid, "score": float(score)}
                for rank, (cid, score) in enumerate(scored_sorted, start=1)
            ]
            ranked_ids = [cid for cid, _ in scored_sorted]

            for k in k_values:
                retrieved_ids = ranked_ids[:k]
                satisfied = 0
                rr_sum = 0.0
                nslots = len(slots)
                for _lab, gold_set in slots:
                    r = _first_hit_rank_1based(retrieved_ids, gold_set)
                    if r is not None:
                        satisfied += 1
                        rr_sum += 1.0 / float(r)
                slot_recall = satisfied / float(nslots) if nslots > 0 else 0.0
                slot_mrr = rr_sum / float(nslots) if nslots > 0 else 0.0
                q_recall = 1.0 if satisfied == nslots and nslots > 0 else 0.0
                acc[k].q_recall_sum += q_recall
                acc[k].slot_recall_sum += slot_recall
                acc[k].slot_mrr_sum += slot_mrr

            out_rec: Dict[str, Any] = {
                "query_id": qid,
                "question": question,
                "slots": slots_raw,
                "required_slots": required_slots,
                "retrieved": reranked,
            }
            for key, val in rec.items():
                if key not in out_rec:
                    out_rec[key] = val

            fout.write(json.dumps(out_rec, ensure_ascii=False) + "\n")
            n_queries += 1

            if n_queries % 10 == 0:
                print(f"Reranked {n_queries}/{len(records)} queries", flush=True)

    if n_queries == 0:
        raise SystemExit("No valid Task B queries reranked.")

    metrics: Dict[int, Dict[str, float]] = {
        k: {
            "QuestionRecall": acc[k].q_recall_sum / float(n_queries),
            "SlotRecall": acc[k].slot_recall_sum / float(n_queries),
            "SlotMRR": acc[k].slot_mrr_sum / float(n_queries),
        }
        for k in k_values
    }
    mean_latency = float(np.mean(latencies)) if latencies else 0.0

    print(f"\n=== TASK B RERANK (Qwen3) EVAL ===")
    print(f"Input traces : {args.in_traces}")
    print(f"Output traces: {out_path.as_posix()}")
    print(f"Queries      : {n_queries}")
    print(f"mean latency : {mean_latency:.6f} sec/query\n")

    for k in k_values:
        m = metrics[k]
        print(
            f"@{k:<2d}  "
            f"QuestionRecall={m['QuestionRecall']:.4f}  "
            f"SlotRecall={m['SlotRecall']:.4f}  "
            f"SlotMRR={m['SlotMRR']:.4f}"
        )

    if args.out_json:
        _write_json(args.out_json, {
            "in_traces": args.in_traces,
            "out_traces": out_path.as_posix(),
            "k_values": k_values,
            "metrics": metrics,
            "mean_latency_sec": mean_latency,
            "n_queries_scored": n_queries,
        })
        print(f"\nWrote metrics JSON: {args.out_json}")


# ---------------------------------------------------------------------------
# BioASQ
# ---------------------------------------------------------------------------

def _recall_at_k_bioasq(ranked_ids: List[str], gold_set: Set[str], k: int) -> float:
    if not gold_set:
        return 0.0
    return float(len(set(ranked_ids[:k]).intersection(gold_set)) / len(gold_set))


def rerank_bioasq(args: argparse.Namespace) -> None:
    """Replicate rerank_bioasq.py behaviour exactly."""
    k_values = sorted({int(x.strip()) for x in args.k_values.split(",") if x.strip()})

    records = _load_traces(args.in_traces)
    doc_texts = _load_doc_texts(args.docs_json)

    from .models import Qwen3Reranker  # deferred: keeps --help torch-free
    reranker = Qwen3Reranker()

    agg: Dict[str, List[float]] = {f"MRR@{k}": [] for k in k_values}
    agg.update({f"Recall@{k}": [] for k in k_values})
    agg.update({f"nDCG@{k}": [] for k in k_values})
    agg["latency_sec"] = []

    out_path = Path(args.out_traces)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_queries = 0
    with out_path.open("w", encoding="utf-8") as fout:
        for rec in records:
            qid = rec.get("query_id")
            question = rec.get("question", "")
            gold_docs: Set[str] = set(rec.get("gold_docs", []) or [])
            retrieved = rec.get("retrieved_docs", []) or []

            if not qid or not question or not retrieved or not gold_docs:
                continue

            top_hits = retrieved[: args.top_k]

            queries: List[str] = []
            passages: List[str] = []
            hit_ids: List[str] = []
            for r in top_hits:
                did = str(r["doc_id"])
                text = doc_texts.get(did)
                if text is None:
                    continue
                queries.append(question)
                passages.append(text)
                hit_ids.append(did)

            if not hit_ids:
                continue

            t0 = time.time()
            scores = reranker.score_pairs(queries, passages, batch_size=args.batch_size)
            elapsed = time.time() - t0
            agg["latency_sec"].append(float(elapsed))

            scored_sorted = sorted(zip(hit_ids, scores), key=lambda x: x[1], reverse=True)

            reranked = [
                {"rank": rank, "doc_id": did, "score": float(score)}
                for rank, (did, score) in enumerate(scored_sorted, start=1)
            ]
            ranked_ids = [did for did, _ in scored_sorted]

            for k in k_values:
                agg[f"MRR@{k}"].append(_mrr_at_k(ranked_ids, gold_docs, k))
                agg[f"Recall@{k}"].append(_recall_at_k_bioasq(ranked_ids, gold_docs, k))
                agg[f"nDCG@{k}"].append(_ndcg_at_k_binary(ranked_ids, gold_docs, k))

            out_rec: Dict[str, Any] = {
                "query_id": qid,
                "question": question,
                "gold_docs": list(gold_docs),
                "retrieved_docs": reranked,
            }
            for key, val in rec.items():
                if key not in out_rec:
                    out_rec[key] = val

            fout.write(json.dumps(out_rec, ensure_ascii=False) + "\n")
            n_queries += 1

            if n_queries % 50 == 0:
                print(f"Reranked {n_queries}/{len(records)} queries", flush=True)

    if n_queries == 0:
        raise SystemExit("No valid BioASQ queries reranked.")

    mean_metrics = {k: float(np.mean(v)) if v else 0.0 for k, v in agg.items()}

    print(f"\n=== BIOASQ RERANK (Qwen3) EVAL ===")
    print(f"Input traces : {args.in_traces}")
    print(f"Output traces: {out_path.as_posix()}")
    print(f"Queries      : {n_queries}")
    print(f"mean latency : {mean_metrics['latency_sec']:.6f} sec/query\n")

    for k in k_values:
        print(
            f"@{k:>2}  "
            f"MRR={mean_metrics[f'MRR@{k}']:.4f}  "
            f"Recall={mean_metrics[f'Recall@{k}']:.4f}  "
            f"nDCG={mean_metrics[f'nDCG@{k}']:.4f}"
        )

    if args.out_json:
        _write_json(args.out_json, {
            "in_traces": args.in_traces,
            "out_traces": out_path.as_posix(),
            "k_values": k_values,
            "metrics": mean_metrics,
            "n_queries_scored": n_queries,
        })
        print(f"\nWrote metrics JSON: {args.out_json}")


# ---------------------------------------------------------------------------
# Argument parser & dispatcher
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Unified Qwen3 reranking + evaluation CLI. Select task with --dataset.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples
--------
# Task A
python -m src.retrieval.rerank_retrieval --dataset A \\
    --in_traces  outputs/qfr/bm25/retrieval_traces.jsonl \\
    --out_traces outputs/qfr/rerank_A/retrieval_traces.jsonl \\
    --chunks_json Data/processed/corpus.chunks.jsonl

# Task B
python -m src.retrieval.rerank_retrieval --dataset B \\
    --in_traces  outputs/qfr/bm25/retrieval_traces.jsonl \\
    --out_traces outputs/qfr/rerank_B/retrieval_traces.jsonl \\
    --chunks_json Data/processed/corpus.chunks.jsonl

# BioASQ
python -m src.retrieval.rerank_retrieval --dataset bioasq \\
    --in_traces  outputs/bioasq/bm25/retrieval_traces.jsonl \\
    --out_traces outputs/bioasq/rerank/retrieval_traces.jsonl \\
    --docs_json  Data/bioasq/processed/docs.jsonl
""",
    )

    # ---- Task selector (new) ----
    ap.add_argument(
        "--dataset",
        required=True,
        choices=["A", "B", "bioasq"],
        help="Task to rerank and evaluate: A (Task A chunk-level), B (Task B slot-based), bioasq (doc-level).",
    )

    # ---- Shared arguments (identical across all three originals) ----
    ap.add_argument(
        "--in_traces",
        required=True,
        help="Input retrieval_traces.jsonl (from BM25 / dense / fusion).",
    )
    ap.add_argument(
        "--out_traces",
        required=True,
        help="Output retrieval_traces.jsonl with reranked list.",
    )
    ap.add_argument(
        "--top_k",
        type=int,
        default=50,
        help="How many retrieved candidates per query to rerank (default: 50).",
    )
    ap.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for reranker inference (default: 8).",
    )
    ap.add_argument(
        "--k_values",
        default="1,3,5,10,20,50",
        help="Comma-separated k values for evaluation (default: 1,3,5,10,20,50).",
    )
    ap.add_argument(
        "--out_json",
        default=None,
        help="Optional path to write metrics JSON.",
    )

    # ---- Corpus text arguments (dataset-specific, validated at runtime) ----
    ap.add_argument(
        "--chunks_json",
        default=None,
        help=(
            "Path to corpus chunks JSONL or JSON mapping chunk_id -> text. "
            "Required when --dataset is A or B."
        ),
    )
    ap.add_argument(
        "--docs_json",
        default=None,
        help=(
            "Path to BioASQ corpus JSONL or JSON mapping doc_id -> text. "
            "Required when --dataset is bioasq."
        ),
    )

    return ap


def main() -> None:
    ap = build_parser()
    args = ap.parse_args()

    # Validate dataset-specific required corpus argument
    if args.dataset in ("A", "B"):
        if not args.chunks_json:
            ap.error("--chunks_json is required when --dataset is A or B.")
    elif args.dataset == "bioasq":
        if not args.docs_json:
            ap.error("--docs_json is required when --dataset is bioasq.")

    if args.dataset == "A":
        rerank_taskA(args)
    elif args.dataset == "B":
        rerank_taskB(args)
    else:
        rerank_bioasq(args)


if __name__ == "__main__":
    main()

