#!/usr/bin/env python3
"""
Examples
--------
# Task A
python -m src.retrieval.fusion_retrieval --dataset A \\
    --traces_a outputs/qfr/bm25/retrieval_traces.jsonl \\
    --traces_b outputs/qfr/dense/retrieval_traces.jsonl \\
    --lambda_a 0.5 --k_values 1,3,5,10,20 \\
    --out_json outputs/qfr/fusion_A.json \\
    --out_traces_linear outputs/qfr/fused_linear_A.jsonl \\
    --out_traces_rrf    outputs/qfr/fused_rrf_A.jsonl

# Task B
python -m src.retrieval.fusion_retrieval --dataset B \\
    --traces_a outputs/qfr/bm25/retrieval_traces.jsonl \\
    --traces_b outputs/qfr/dense/retrieval_traces.jsonl \\
    --out_json outputs/qfr/fusion_B.json

# BioASQ
python -m src.retrieval.fusion_retrieval --dataset bioasq \\
    --traces_a outputs/bioasq/bm25/retrieval_traces.jsonl \\
    --traces_b outputs/bioasq/dense/retrieval_traces.jsonl \\
    --out_json outputs/bioasq/fusion_bioasq.json
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import numpy as np


# Shared IO helpers

def _iter_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _write_jsonl(path: str, records: List[Dict[str, Any]]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _write_json(path: str, obj: Any) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def _parse_k_values(s: str) -> List[int]:
    return sorted({int(x.strip()) for x in s.split(",") if x.strip()})


# Shared fusion logic  (identical across all three original scripts)

def _normalize_scores(d: Dict[str, float]) -> Dict[str, float]:
    if not d:
        return {}
    vals = np.array(list(d.values()), dtype=float)
    vmin = float(vals.min())
    vmax = float(vals.max())
    if vmax <= vmin:
        return {cid: 0.0 for cid in d.keys()}
    return {cid: (s - vmin) / (vmax - vmin) for cid, s in d.items()}


def linear_fusion(
    hits_a: List[Tuple[str, float, int]],
    hits_b: List[Tuple[str, float, int]],
    lambda_a: float,
    k: int,
) -> List[Tuple[str, float]]:
    """
    Linear score fusion over two models.

    hits_*: list of (id, score, rank)
    lambda_a: weight for model A; (1-lambda_a) for model B.
    Returns list of (id, fused_score) sorted desc, truncated to top-k.
    """
    scores_a: Dict[str, float] = {cid: score for cid, score, _ in hits_a}
    scores_b: Dict[str, float] = {cid: score for cid, score, _ in hits_b}

    all_ids = set(scores_a.keys()) | set(scores_b.keys())
    norm_a = _normalize_scores(scores_a)
    norm_b = _normalize_scores(scores_b)

    fused: List[Tuple[str, float]] = [
        (cid, lambda_a * norm_a.get(cid, 0.0) + (1.0 - lambda_a) * norm_b.get(cid, 0.0))
        for cid in all_ids
    ]
    return sorted(fused, key=lambda x: x[1], reverse=True)[:k]


def rrf_fusion(
    hits_a: List[Tuple[str, float, int]],
    hits_b: List[Tuple[str, float, int]],
    k: int,
    k_rrf: int = 60,
) -> List[Tuple[str, float]]:
    """
    Reciprocal Rank Fusion (RRF) over two ranked lists.

    score_rrf(doc) = sum_m 1 / (k_rrf + rank_m(doc))
    Returns list of (id, rrf_score) sorted desc, truncated to top-k.
    """
    rrf_scores: Dict[str, float] = {}
    for cid, _score, rank in hits_a:
        rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / float(k_rrf + rank)
    for cid, _score, rank in hits_b:
        rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / float(k_rrf + rank)
    return sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:k]


# Task A

def _load_taskA_traces(path: str) -> Dict[str, Dict[str, Any]]:
    """Load fusion_taskA.py retrieval_traces.jsonl format."""
    out: Dict[str, Dict[str, Any]] = {}
    for obj in _iter_jsonl(path):
        qid = str(obj.get("query_id", "")).strip()
        if not qid:
            continue
        question = obj.get("question", "")
        gold = obj.get("gold_chunk_ids", []) or []
        retrieved_raw = obj.get("retrieved", []) or []
        retrieved: List[Tuple[str, float, int]] = [
            (r["chunk_id"], float(r["score"]), int(r["rank"]))
            for r in retrieved_raw
        ]
        out[qid] = {
            "question": question,
            "gold_chunk_ids": set(gold),
            "retrieved": retrieved,
        }
    return out


def _mrr_at_k_A(ranked_ids: List[str], gold_set: Set[str], k: int) -> float:
    for i, cid in enumerate(ranked_ids[:k], start=1):
        if cid in gold_set:
            return 1.0 / i
    return 0.0


def _hit_recall_at_k(ranked_ids: List[str], gold_set: Set[str], k: int) -> float:
    for cid in ranked_ids[:k]:
        if cid in gold_set:
            return 1.0
    return 0.0


def _set_recall_at_k(ranked_ids: List[str], gold_set: Set[str], k: int) -> float:
    if not gold_set:
        return 0.0
    return float(len(set(ranked_ids[:k]).intersection(gold_set)) / len(gold_set))


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


def fuse_taskA(args: argparse.Namespace) -> None:
    """Replicate fusion_taskA.py behaviour exactly."""
    k_values = _parse_k_values(args.k_values)
    k_max = max(k_values)

    traces_a = _load_taskA_traces(args.traces_a)
    traces_b = _load_taskA_traces(args.traces_b)

    common_qids = sorted(set(traces_a.keys()) & set(traces_b.keys()))
    print(f"Loaded {len(common_qids)} common queries")

    agg_linear: Dict[str, List[float]] = {f"MRR@{k}": [] for k in k_values}
    agg_linear.update({f"HitRecall@{k}": [] for k in k_values})
    agg_linear.update({f"SetRecall@{k}": [] for k in k_values})
    agg_linear.update({f"nDCG@{k}": [] for k in k_values})
    agg_linear["latency_sec"] = []

    agg_rrf: Dict[str, List[float]] = {f"MRR@{k}": [] for k in k_values}
    agg_rrf.update({f"HitRecall@{k}": [] for k in k_values})
    agg_rrf.update({f"SetRecall@{k}": [] for k in k_values})
    agg_rrf.update({f"nDCG@{k}": [] for k in k_values})
    agg_rrf["latency_sec"] = []

    fused_linear_records: List[Dict[str, Any]] = []
    fused_rrf_records: List[Dict[str, Any]] = []

    for qid in common_qids:
        gold = traces_a[qid]["gold_chunk_ids"]
        hits_a = traces_a[qid]["retrieved"]
        hits_b = traces_b[qid]["retrieved"]
        question = traces_a[qid].get("question", "")

        t0 = time.time()
        fused_linear = linear_fusion(hits_a, hits_b, lambda_a=args.lambda_a, k=k_max)
        elapsed_linear = time.time() - t0

        t1 = time.time()
        fused_rrf = rrf_fusion(hits_a, hits_b, k=k_max, k_rrf=60)
        elapsed_rrf = time.time() - t1

        ranked_linear_ids = [cid for cid, _s in fused_linear]
        ranked_rrf_ids = [cid for cid, _s in fused_rrf]

        for k in k_values:
            for agg, ranked_ids in ((agg_linear, ranked_linear_ids), (agg_rrf, ranked_rrf_ids)):
                agg[f"MRR@{k}"].append(_mrr_at_k_A(ranked_ids, gold, k))
                agg[f"HitRecall@{k}"].append(_hit_recall_at_k(ranked_ids, gold, k))
                agg[f"SetRecall@{k}"].append(_set_recall_at_k(ranked_ids, gold, k))
                agg[f"nDCG@{k}"].append(_ndcg_at_k_binary(ranked_ids, gold, k))

        agg_linear["latency_sec"].append(float(elapsed_linear))
        agg_rrf["latency_sec"].append(float(elapsed_rrf))

        if args.out_traces_linear:
            fused_linear_records.append({
                "query_id": qid,
                "question": question,
                "gold_chunk_ids": list(gold),
                "retrieved": [
                    {"rank": rank, "chunk_id": cid, "score": float(score)}
                    for rank, (cid, score) in enumerate(fused_linear, start=1)
                ],
            })

        if args.out_traces_rrf:
            fused_rrf_records.append({
                "query_id": qid,
                "question": question,
                "gold_chunk_ids": list(gold),
                "retrieved": [
                    {"rank": rank, "chunk_id": cid, "score": float(score)}
                    for rank, (cid, score) in enumerate(fused_rrf, start=1)
                ],
            })

    def _mean(agg: Dict[str, List[float]]) -> Dict[str, float]:
        return {k: float(np.mean(v)) if v else 0.0 for k, v in agg.items()}

    mean_linear = _mean(agg_linear)
    mean_rrf = _mean(agg_rrf)

    print("\n=== TASK A FUSION EVAL ===")
    print(f"Traces A : {args.traces_a}")
    print(f"Traces B : {args.traces_b}")
    print(f"Queries  : {len(common_qids)}")
    print(f"lambda_A (linear fusion) = {args.lambda_a}\n")
    print(f"Linear fusion mean latency: {mean_linear['latency_sec']:.6f} sec/query")
    print(f"RRF fusion mean latency   : {mean_rrf['latency_sec']:.6f} sec/query\n")

    print("---- Linear score fusion ----")
    for k in k_values:
        print(
            f"@{k:>2}  "
            f"MRR={mean_linear[f'MRR@{k}']:.4f}  "
            f"HitRecall={mean_linear[f'HitRecall@{k}']:.4f}  "
            f"SetRecall={mean_linear[f'SetRecall@{k}']:.4f}  "
            f"nDCG={mean_linear[f'nDCG@{k}']:.4f}"
        )

    print("\n---- Reciprocal Rank Fusion (RRF) ----")
    for k in k_values:
        print(
            f"@{k:>2}  "
            f"MRR={mean_rrf[f'MRR@{k}']:.4f}  "
            f"HitRecall={mean_rrf[f'HitRecall@{k}']:.4f}  "
            f"SetRecall={mean_rrf[f'SetRecall@{k}']:.4f}  "
            f"nDCG={mean_rrf[f'nDCG@{k}']:.4f}"
        )

    if args.out_json:
        _write_json(args.out_json, {
            "traces_a": args.traces_a,
            "traces_b": args.traces_b,
            "lambda_a": float(args.lambda_a),
            "k_values": k_values,
            "mean_linear": mean_linear,
            "mean_rrf": mean_rrf,
        })
        print(f"\nWrote metrics JSON: {args.out_json}")

    if args.out_traces_linear:
        _write_jsonl(args.out_traces_linear, fused_linear_records)
        print(f"Wrote fused linear traces: {args.out_traces_linear}")

    if args.out_traces_rrf:
        _write_jsonl(args.out_traces_rrf, fused_rrf_records)
        print(f"Wrote fused RRF traces: {args.out_traces_rrf}")


# Task B

@dataclass
class _PerKAccum:
    q_recall_sum: float = 0.0
    slot_recall_sum: float = 0.0
    slot_mrr_sum: float = 0.0


def _load_taskB_traces(path: str) -> Dict[str, Dict[str, Any]]:
    """Load fusion_taskB.py retrieval_traces.jsonl format."""
    out: Dict[str, Dict[str, Any]] = {}
    for obj in _iter_jsonl(path):
        qid = str(obj.get("query_id", "")).strip()
        if not qid:
            continue
        question = obj.get("question", "")
        slots_raw = obj.get("slots", []) or []
        slots: List[Tuple[str, Set[str]]] = []
        for s in slots_raw:
            lab = s.get("label")
            gold_ids = s.get("gold_ids", []) or []
            if lab and gold_ids:
                slots.append((lab, set(gold_ids)))
        required_slots = obj.get("required_slots", None)
        try:
            required_slots_n: Optional[int] = int(required_slots) if required_slots is not None else None
        except Exception:
            required_slots_n = None
        retrieved_raw = obj.get("retrieved", []) or []
        retrieved: List[Tuple[str, float, int]] = [
            (r["chunk_id"], float(r["score"]), int(r["rank"]))
            for r in retrieved_raw
        ]
        out[qid] = {
            "question": question,
            "slots": slots,
            "required_slots_n": required_slots_n,
            "retrieved": retrieved,
        }
    return out


def _first_hit_rank_1based(retrieved_ids: List[str], gold_set: Set[str]) -> Optional[int]:
    for i, cid in enumerate(retrieved_ids, start=1):
        if cid in gold_set:
            return i
    return None


def fuse_taskB(args: argparse.Namespace) -> None:
    """Replicate fusion_taskB.py behaviour exactly."""
    k_values = _parse_k_values(args.k_values)
    max_k = max(k_values)

    traces_a = _load_taskB_traces(args.traces_a)
    traces_b = _load_taskB_traces(args.traces_b)

    common_qids = sorted(set(traces_a.keys()) & set(traces_b.keys()))
    print(f"Loaded {len(common_qids)} common queries")

    acc_linear: Dict[int, _PerKAccum] = {k: _PerKAccum() for k in k_values}
    acc_rrf: Dict[int, _PerKAccum] = {k: _PerKAccum() for k in k_values}
    total_latency_linear = 0.0
    total_latency_rrf = 0.0
    scored = 0

    fused_linear_records: List[Dict[str, Any]] = []
    fused_rrf_records: List[Dict[str, Any]] = []

    for qid in common_qids:
        info_a = traces_a[qid]
        info_b = traces_b[qid]
        question = info_a.get("question", "")

        slots = info_a["slots"]
        required_slots_n = info_a["required_slots_n"]
        if required_slots_n is not None:
            slots = slots[:required_slots_n]

        if not slots:
            continue

        hits_a = info_a["retrieved"]
        hits_b = info_b["retrieved"]

        t0 = time.time()
        fused_linear = linear_fusion(hits_a, hits_b, lambda_a=args.lambda_a, k=max_k)
        elapsed_linear = time.time() - t0

        t1 = time.time()
        fused_rrf = rrf_fusion(hits_a, hits_b, k=max_k, k_rrf=60)
        elapsed_rrf = time.time() - t1

        ranked_linear_ids = [cid for cid, _s in fused_linear]
        ranked_rrf_ids = [cid for cid, _s in fused_rrf]

        for k in k_values:
            for acc, retrieved_ids in (
                (acc_linear, ranked_linear_ids[:k]),
                (acc_rrf, ranked_rrf_ids[:k]),
            ):
                satisfied = 0
                rr_sum = 0.0
                nslots = len(slots)
                for _lab, gold_set in slots:
                    r = _first_hit_rank_1based(retrieved_ids, gold_set)
                    if r is not None:
                        satisfied += 1
                        rr_sum += 1.0 / float(r)
                acc[k].q_recall_sum += 1.0 if satisfied == nslots else 0.0
                acc[k].slot_recall_sum += satisfied / float(nslots)
                acc[k].slot_mrr_sum += rr_sum / float(nslots)

        total_latency_linear += float(elapsed_linear)
        total_latency_rrf += float(elapsed_rrf)
        scored += 1

        if args.out_traces_linear:
            fused_linear_records.append({
                "query_id": qid,
                "question": question,
                "slots": [
                    {"label": lab, "gold_ids": list(gold_ids)}
                    for lab, gold_ids in info_a["slots"]
                ],
                "required_slots": info_a["required_slots_n"],
                "retrieved": [
                    {"rank": rank, "chunk_id": cid, "score": float(score)}
                    for rank, (cid, score) in enumerate(fused_linear, start=1)
                ],
            })

        if args.out_traces_rrf:
            fused_rrf_records.append({
                "query_id": qid,
                "question": question,
                "slots": [
                    {"label": lab, "gold_ids": list(gold_ids)}
                    for lab, gold_ids in info_a["slots"]
                ],
                "required_slots": info_a["required_slots_n"],
                "retrieved": [
                    {"rank": rank, "chunk_id": cid, "score": float(score)}
                    for rank, (cid, score) in enumerate(fused_rrf, start=1)
                ],
            })

    if scored == 0:
        raise SystemExit("No valid common questions scored.")

    def _summarize_B(acc: Dict[int, _PerKAccum]) -> Dict[int, Dict[str, float]]:
        return {
            k: {
                "QuestionRecall": acc[k].q_recall_sum / float(scored),
                "SlotRecall": acc[k].slot_recall_sum / float(scored),
                "SlotMRR": acc[k].slot_mrr_sum / float(scored),
            }
            for k in k_values
        }

    metrics_linear = _summarize_B(acc_linear)
    metrics_rrf = _summarize_B(acc_rrf)
    mean_lat_linear = total_latency_linear / float(scored)
    mean_lat_rrf = total_latency_rrf / float(scored)

    print("\n=== TASK B FUSION EVAL (slot-based) ===")
    print(f"Traces A : {args.traces_a}")
    print(f"Traces B : {args.traces_b}")
    print(f"Queries  : {scored}")
    print(f"lambda_A (linear fusion) = {args.lambda_a}\n")
    print(f"Linear fusion mean latency: {mean_lat_linear:.6f} sec/query")
    print(f"RRF fusion mean latency   : {mean_lat_rrf:.6f} sec/query\n")

    print("---- Linear score fusion ----")
    for k in k_values:
        m = metrics_linear[k]
        print(
            f"@{k:<2d}  "
            f"QuestionRecall={m['QuestionRecall']:.4f}  "
            f"SlotRecall={m['SlotRecall']:.4f}  "
            f"SlotMRR={m['SlotMRR']:.4f}"
        )

    print("\n---- Reciprocal Rank Fusion (RRF) ----")
    for k in k_values:
        m = metrics_rrf[k]
        print(
            f"@{k:<2d}  "
            f"QuestionRecall={m['QuestionRecall']:.4f}  "
            f"SlotRecall={m['SlotRecall']:.4f}  "
            f"SlotMRR={m['SlotMRR']:.4f}"
        )

    if args.out_json:
        _write_json(args.out_json, {
            "traces_a": args.traces_a,
            "traces_b": args.traces_b,
            "lambda_a": float(args.lambda_a),
            "k_values": k_values,
            "metrics_linear": metrics_linear,
            "metrics_rrf": metrics_rrf,
            "mean_latency_linear_sec": mean_lat_linear,
            "mean_latency_rrf_sec": mean_lat_rrf,
            "n_queries_scored": scored,
        })
        print(f"\nWrote metrics JSON: {args.out_json}")

    if args.out_traces_linear:
        _write_jsonl(args.out_traces_linear, fused_linear_records)
        print(f"Wrote fused linear Task B traces: {args.out_traces_linear}")

    if args.out_traces_rrf:
        _write_jsonl(args.out_traces_rrf, fused_rrf_records)
        print(f"Wrote fused RRF Task B traces: {args.out_traces_rrf}")


# BioASQ

def _load_bioasq_traces(path: str) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for obj in _iter_jsonl(path):
        qid = str(obj.get("query_id", "")).strip()
        if not qid:
            continue

        question = obj.get("question", "")
        gold_docs = obj.get("gold_docs", []) or []

        retrieved_chunks_raw = obj.get("retrieved_chunks", []) or []
        retrieved_chunks: List[Tuple[str, float, int]] = [
            (r["chunk_id"], float(r["score"]), int(r["rank"]))
            for r in retrieved_chunks_raw
        ]

        retrieved_docs_raw = obj.get("retrieved_docs", []) or []
        retrieved_docs: List[Tuple[str, float, int]] = [
            (r["doc_id"], float(r["score"]), int(r["rank"]))
            for r in retrieved_docs_raw
        ]

        out[qid] = {
            "question": question,
            "gold_docs": set(gold_docs),
            "retrieved_chunks": retrieved_chunks,
            "retrieved_docs": retrieved_docs,
        }
    return out

def _chunk_id_to_doc_id_bioasq(chunk_id: str) -> str:
    if "_chunk" in chunk_id:
        return chunk_id.split("_chunk", 1)[0]
    if "_" in chunk_id:
        return chunk_id.rsplit("_", 1)[0]
    return chunk_id


def _aggregate_fused_chunks_to_docs(
    hits: List[Tuple[str, float]],
    method: str = "max",
) -> List[Tuple[str, float]]:
    doc_scores: Dict[str, float] = {}

    if method not in {"max", "sum"}:
        raise ValueError(f"Unknown aggregation method: {method}")

    for cid, score in hits:
        did = _chunk_id_to_doc_id_bioasq(cid)
        if method == "max":
            if did not in doc_scores or score > doc_scores[did]:
                doc_scores[did] = float(score)
        else:
            doc_scores[did] = doc_scores.get(did, 0.0) + float(score)

    return sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)

def _mrr_at_k_bioasq(ranked_ids: List[str], gold_set: Set[str], k: int) -> float:
    for i, did in enumerate(ranked_ids[:k], start=1):
        if did in gold_set:
            return 1.0 / i
    return 0.0


def _recall_at_k_bioasq(ranked_ids: List[str], gold_set: Set[str], k: int) -> float:
    if not gold_set:
        return 0.0
    return float(len(set(ranked_ids[:k]).intersection(gold_set)) / len(gold_set))


def _ndcg_at_k_binary_bioasq(ranked_ids: List[str], gold_set: Set[str], k: int) -> float:
    dcg = 0.0
    for i, did in enumerate(ranked_ids[:k], start=1):
        if did in gold_set:
            dcg += 1.0 / np.log2(i + 1)
    m = min(len(gold_set), k)
    if m == 0:
        return 0.0
    idcg = sum(1.0 / np.log2(i + 1) for i in range(1, m + 1))
    return float(dcg / idcg)


def fuse_bioasq(args: argparse.Namespace) -> None:
    k_values = _parse_k_values(args.k_values)
    k_max_docs = max(k_values)

    traces_a = _load_bioasq_traces(args.traces_a)
    traces_b = _load_bioasq_traces(args.traces_b)

    common_qids = sorted(set(traces_a.keys()) & set(traces_b.keys()))
    print(f"Loaded {len(common_qids)} common queries")

    agg_linear: Dict[str, List[float]] = {f"MRR@{k}": [] for k in k_values}
    agg_linear.update({f"Recall@{k}": [] for k in k_values})
    agg_linear.update({f"nDCG@{k}": [] for k in k_values})
    agg_linear["latency_sec"] = []

    agg_rrf: Dict[str, List[float]] = {f"MRR@{k}": [] for k in k_values}
    agg_rrf.update({f"Recall@{k}": [] for k in k_values})
    agg_rrf.update({f"nDCG@{k}": [] for k in k_values})
    agg_rrf["latency_sec"] = []

    fused_linear_records: List[Dict[str, Any]] = []
    fused_rrf_records: List[Dict[str, Any]] = []
    scored = 0

    for qid in common_qids:
        info_a = traces_a[qid]
        info_b = traces_b[qid]
        question = info_a.get("question", "")
        gold = info_a["gold_docs"]

        if not gold:
            continue

        hits_a_chunks = info_a.get("retrieved_chunks", [])
        hits_b_chunks = info_b.get("retrieved_chunks", [])

        if not hits_a_chunks or not hits_b_chunks:
            continue

        # keep full chunk depth from source traces, not just max(k_values)
        fusion_k_chunks = max(len(hits_a_chunks), len(hits_b_chunks))

        t0 = time.time()
        fused_linear_chunks = linear_fusion(
            hits_a_chunks, hits_b_chunks,
            lambda_a=args.lambda_a,
            k=fusion_k_chunks,
        )
        elapsed_linear = time.time() - t0

        t1 = time.time()
        fused_rrf_chunks = rrf_fusion(
            hits_a_chunks, hits_b_chunks,
            k=fusion_k_chunks,
            k_rrf=60,
        )
        elapsed_rrf = time.time() - t1

        fused_linear_docs = _aggregate_fused_chunks_to_docs(
            fused_linear_chunks, method=args.agg
        )
        fused_rrf_docs = _aggregate_fused_chunks_to_docs(
            fused_rrf_chunks, method=args.agg
        )

        ranked_linear_doc_ids = [did for did, _s in fused_linear_docs]
        ranked_rrf_doc_ids = [did for did, _s in fused_rrf_docs]

        for k in k_values:
            for agg, ranked_ids in (
                (agg_linear, ranked_linear_doc_ids),
                (agg_rrf, ranked_rrf_doc_ids),
            ):
                agg[f"MRR@{k}"].append(_mrr_at_k_bioasq(ranked_ids, gold, k))
                agg[f"Recall@{k}"].append(_recall_at_k_bioasq(ranked_ids, gold, k))
                agg[f"nDCG@{k}"].append(_ndcg_at_k_binary_bioasq(ranked_ids, gold, k))

        agg_linear["latency_sec"].append(float(elapsed_linear))
        agg_rrf["latency_sec"].append(float(elapsed_rrf))
        scored += 1

        if args.out_traces_linear:
            fused_linear_records.append({
                "query_id": qid,
                "question": question,
                "gold_docs": list(gold),
                "retrieved_chunks": [
                    {
                        "rank": rank,
                        "chunk_id": cid,
                        "doc_id": _chunk_id_to_doc_id_bioasq(cid),
                        "score": float(score),
                    }
                    for rank, (cid, score) in enumerate(fused_linear_chunks, start=1)
                ],
                "retrieved_docs": [
                    {"rank": rank, "doc_id": did, "score": float(score)}
                    for rank, (did, score) in enumerate(fused_linear_docs, start=1)
                ],
            })

        if args.out_traces_rrf:
            fused_rrf_records.append({
                "query_id": qid,
                "question": question,
                "gold_docs": list(gold),
                "retrieved_chunks": [
                    {
                        "rank": rank,
                        "chunk_id": cid,
                        "doc_id": _chunk_id_to_doc_id_bioasq(cid),
                        "score": float(score),
                    }
                    for rank, (cid, score) in enumerate(fused_rrf_chunks, start=1)
                ],
                "retrieved_docs": [
                    {"rank": rank, "doc_id": did, "score": float(score)}
                    for rank, (did, score) in enumerate(fused_rrf_docs, start=1)
                ],
            })

    if scored == 0:
        raise SystemExit("No valid common queries scored (no gold docs or no retrieved_chunks).")

    def _mean(agg: Dict[str, List[float]]) -> Dict[str, float]:
        return {k: float(np.mean(v)) if v else 0.0 for k, v in agg.items()}

    mean_linear = _mean(agg_linear)
    mean_rrf = _mean(agg_rrf)

    print("\n=== BIOASQ FUSION EVAL (chunk-fused -> doc-evaluated) ===")
    print(f"Traces A : {args.traces_a}")
    print(f"Traces B : {args.traces_b}")
    print(f"Queries  : {scored}")
    print(f"lambda_A (linear fusion) = {args.lambda_a}")
    print(f"agg      : {args.agg}\n")
    print(f"Linear fusion mean latency: {mean_linear['latency_sec']:.6f} sec/query")
    print(f"RRF fusion mean latency   : {mean_rrf['latency_sec']:.6f} sec/query\n")

    print("---- Linear score fusion ----")
    for k in k_values:
        print(
            f"@{k:>3}  "
            f"MRR={mean_linear[f'MRR@{k}']:.4f}  "
            f"Recall={mean_linear[f'Recall@{k}']:.4f}  "
            f"nDCG={mean_linear[f'nDCG@{k}']:.4f}"
        )

    print("\n---- Reciprocal Rank Fusion (RRF) ----")
    for k in k_values:
        print(
            f"@{k:>3}  "
            f"MRR={mean_rrf[f'MRR@{k}']:.4f}  "
            f"Recall={mean_rrf[f'Recall@{k}']:.4f}  "
            f"nDCG={mean_rrf[f'nDCG@{k}']:.4f}"
        )

    if args.out_json:
        _write_json(args.out_json, {
            "traces_a": args.traces_a,
            "traces_b": args.traces_b,
            "lambda_a": float(args.lambda_a),
            "agg": args.agg,
            "k_values": k_values,
            "mean_linear": mean_linear,
            "mean_rrf": mean_rrf,
            "n_queries_scored": scored,
        })
        print(f"\nWrote metrics JSON: {args.out_json}")

    if args.out_traces_linear:
        _write_jsonl(args.out_traces_linear, fused_linear_records)
        print(f"Wrote fused linear BioASQ traces: {args.out_traces_linear}")

    if args.out_traces_rrf:
        _write_jsonl(args.out_traces_rrf, fused_rrf_records)
        print(f"Wrote fused RRF BioASQ traces: {args.out_traces_rrf}")


# Argument parser & dispatcher

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Unified fusion evaluation CLI (linear + RRF). Select task with --dataset.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples
--------
# Task A
python -m src.retrieval.fusion_retrieval --dataset A \\
    --traces_a outputs/qfr/bm25/retrieval_traces.jsonl \\
    --traces_b outputs/qfr/dense/retrieval_traces.jsonl

# Task B
python -m src.retrieval.fusion_retrieval --dataset B \\
    --traces_a outputs/qfr/bm25/retrieval_traces.jsonl \\
    --traces_b outputs/qfr/dense/retrieval_traces.jsonl

# BioASQ
python -m src.retrieval.fusion_retrieval --dataset bioasq \\
    --traces_a outputs/bioasq/bm25/retrieval_traces.jsonl \\
    --traces_b outputs/bioasq/dense/retrieval_traces.jsonl
""",
    )

    ap.add_argument(
        "--dataset",
        required=True,
        choices=["A", "B", "bioasq"],
        help="Task to evaluate: A (Task A chunk-level), B (Task B slot-based), bioasq (doc-level).",
    )

    # Shared arguments (identical across all three original scripts)
    ap.add_argument("--traces_a", required=True,
                    help="retrieval_traces.jsonl for model A (e.g., BM25).")
    ap.add_argument("--traces_b", required=True,
                    help="retrieval_traces.jsonl for model B (e.g., dense).")
    ap.add_argument("--lambda_a", type=float, default=0.5,
                    help="Weight for model A in linear fusion (default: 0.5).")
    ap.add_argument("--k_values", default="1,3,5,10,20,50",
                    help="Comma-separated k values (default: 1,3,5,10,20,50).")
    ap.add_argument("--out_json", default=None,
                    help="Optional path to write fusion metrics JSON.")
    ap.add_argument("--out_traces_linear", default=None,
                    help="Optional path to write fused linear retrieval_traces.jsonl.")
    ap.add_argument("--out_traces_rrf", default=None,
                    help="Optional path to write fused RRF retrieval_traces.jsonl.")
    ap.add_argument("--agg", choices=["max", "sum"], default="max",
                    help="BioASQ only: aggregate fused chunk scores to docs (default: max).")

    return ap


def main() -> None:
    ap = build_parser()
    args = ap.parse_args()

    if args.dataset == "A":
        fuse_taskA(args)
    elif args.dataset == "B":
        fuse_taskB(args)
    else:
        fuse_bioasq(args)


if __name__ == "__main__":
    main()

