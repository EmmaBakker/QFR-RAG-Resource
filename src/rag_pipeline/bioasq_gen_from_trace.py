#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import platform
import re
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.retrieval.determinism import set_deterministic
from src.retrieval.utils_io import iter_jsonl

from .corpus_store import CorpusStore
from .llm_client import LLMConfig, _safe_llm_meta, generate_with_info
from .parse_output import parse_all
from .prompts import build_prompt
from .schema import (
    NIC_TOKEN,
    FINAL_STATUS_ANSWERED,
    FINAL_STATUS_EXPLICIT_NIC,
    FINAL_STATUS_FORCED_NIC_EMPTY,
    FINAL_STATUS_FORCED_NIC_NO_CIT,
    FINAL_STATUS_ERROR,
    GenerationRecord,
)

logger = logging.getLogger(__name__)


def _collect_env_metadata() -> Dict[str, Any]:
    try:
        import torch
        torch_version = torch.__version__
        cuda_available = torch.cuda.is_available()
    except Exception:
        torch_version = None
        cuda_available = None

    return {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "python_version": sys.version,
        "platform": platform.platform(),
        "torch_version": torch_version,
        "cuda_available": cuda_available,
    }


def _count_words(s: str) -> int:
    return 0 if not s else len(s.split())


def _token_estimate_from_text(s: str) -> int:
    if not s:
        return 0
    return max(1, int(len(s) / 4))


def _normalize_doc_id(x: Any) -> str:
    s = str(x or "").strip()
    if not s:
        return ""
    if s.startswith("pmid:"):
        return s
    if s.isdigit():
        return f"pmid:{s}"
    return s


def _infer_doc_id_from_chunk_id(chunk_id: str) -> str:
    cid = str(chunk_id or "").strip()
    if not cid:
        return ""

    if cid.startswith("pmid:"):
        return cid.split("::", 1)[0].split("#", 1)[0]

    m = re.match(r"^(pmid:\d+)", cid, flags=re.I)
    if m:
        return m.group(1).lower()

    m = re.match(r"^(\d+)(?:[:_#-]|$)", cid)
    if m:
        return f"pmid:{m.group(1)}"

    return ""


def _doc_id_from_corpus_row(row: Dict[str, Any]) -> str:
    for key in ("doc_id", "document_id", "pmid", "source_id", "paper_id", "article_id"):
        val = row.get(key)
        if val not in (None, ""):
            return _normalize_doc_id(val)

    return _infer_doc_id_from_chunk_id(str(row.get("chunk_id", "")))


def _chunk_sort_key(chunk_id: str) -> Tuple[int, int, str]:
    cid = str(chunk_id or "")
    m = re.search(r"(?:chunk|passage|part|seg|section)[_:\-]?(\d+)", cid, flags=re.I)
    if m:
        return (0, int(m.group(1)), cid)

    nums = re.findall(r"\d+", cid)
    if nums:
        return (1, int(nums[-1]), cid)

    return (2, 0, cid)


def _build_doc_to_chunks_index(corpus_path: str) -> Dict[str, List[str]]:
    doc_to_chunks: Dict[str, List[str]] = defaultdict(list)
    total_rows = 0
    indexed_rows = 0

    for row in iter_jsonl(corpus_path):
        total_rows += 1
        chunk_id = str(row.get("chunk_id", "")).strip()
        if not chunk_id:
            continue

        doc_id = _doc_id_from_corpus_row(row)
        if not doc_id:
            continue

        doc_to_chunks[doc_id].append(chunk_id)
        indexed_rows += 1

    for doc_id in doc_to_chunks:
        doc_to_chunks[doc_id].sort(key=_chunk_sort_key)

    logger.info(
        "Built BioASQ doc->chunks index: docs=%d indexed_chunks=%d total_rows=%d",
        len(doc_to_chunks), indexed_rows, total_rows
    )
    return dict(doc_to_chunks)


def _expand_retrieved_docs_to_chunks(
    trace_row: Dict[str, Any],
    doc_to_chunks: Dict[str, List[str]],
    store: CorpusStore,
    top_k_context: int,
) -> Tuple[List[Dict[str, Any]], int]:
    """
    Convert document-level retrieved_docs into chunk-level contexts.

    Strategy:
    - preserve retrieved_docs ranking
    - distribute context budget round-robin across docs
    - all chunks from same doc inherit the doc score
    """
    doc_hits: List[Dict[str, Any]] = []
    missing_docs = 0

    for item in trace_row.get("retrieved_docs") or []:
        doc_id = _normalize_doc_id(item.get("doc_id"))
        if not doc_id:
            continue

        chunk_ids = doc_to_chunks.get(doc_id)
        if not chunk_ids:
            missing_docs += 1
            continue

        score = float(item.get("score", 0.0) or 0.0)
        doc_hits.append({
            "doc_id": doc_id,
            "score": score,
            "chunk_ids": chunk_ids,
        })

    retrieved: List[Dict[str, Any]] = []
    depth = 0

    while len(retrieved) < int(top_k_context):
        added_any = False

        for hit in doc_hits:
            if depth < len(hit["chunk_ids"]):
                cid = hit["chunk_ids"][depth]
                txt = store.get_text(cid)
                if txt is None:
                    txt = ""
                retrieved.append({
                    "chunk_id": cid,
                    "score": hit["score"],
                    "text": txt,
                })
                added_any = True

                if len(retrieved) >= int(top_k_context):
                    break

        if not added_any:
            break

        depth += 1

    return retrieved, missing_docs


def _make_forced_nic_empty_record(qid: str, question: str) -> GenerationRecord:
    return GenerationRecord(
        id=qid,
        question=question,
        prompt=None,
        raw_output="",
        answer_raw="",
        citations_raw=[],
        pred_answer=NIC_TOKEN,
        citations=[],
        invalid_citations=[],
        num_citations=0,
        explicit_nic=False,
        forced_nic_empty_retrieval=True,
        forced_nic_no_valid_citations=False,
        forced_not_in_context=True,
        is_grounded=False,
        final_status=FINAL_STATUS_FORCED_NIC_EMPTY,
        retrieved=[],
        contexts=[],
        context_ids=[],
        context_scores=[],
        prompt_chars=0,
        prompt_words=0,
        prompt_token_est=0,
        llm_latency_s=None,
        llm_info={},
        error=None,
    )


def _classify_llm_output(
    qid: str,
    question: str,
    prompt: str,
    raw: str,
    llm_info: Dict[str, Any],
    retrieved: List[Dict[str, Any]],
    used_chunk_ids: set,
) -> GenerationRecord:
    parsed = parse_all(raw)
    ans_raw = parsed.answer
    pred = ans_raw.strip() if ans_raw else NIC_TOKEN
    citations_raw = parsed.citations

    valid_citations = [c for c in citations_raw if c in used_chunk_ids]
    invalid_citations = [c for c in citations_raw if c not in used_chunk_ids]

    explicit_nic = pred.upper() == NIC_TOKEN.upper()
    forced_nic_no_valid_citations = False

    if explicit_nic:
        final_status = FINAL_STATUS_EXPLICIT_NIC
        valid_citations = []
        invalid_citations = []

    elif not valid_citations:
        pred = NIC_TOKEN
        forced_nic_no_valid_citations = True
        final_status = FINAL_STATUS_FORCED_NIC_NO_CIT
        valid_citations = []
        invalid_citations = []

    else:
        final_status = FINAL_STATUS_ANSWERED

    forced_not_in_context = forced_nic_no_valid_citations
    is_grounded = final_status == FINAL_STATUS_ANSWERED

    return GenerationRecord(
        id=qid,
        question=question,
        prompt=prompt,
        raw_output=raw,
        answer_raw=ans_raw,
        citations_raw=citations_raw,
        pred_answer=pred,
        citations=valid_citations,
        invalid_citations=invalid_citations,
        num_citations=len(valid_citations),
        explicit_nic=explicit_nic,
        forced_nic_empty_retrieval=False,
        forced_nic_no_valid_citations=forced_nic_no_valid_citations,
        forced_not_in_context=forced_not_in_context,
        is_grounded=is_grounded,
        final_status=final_status,
        retrieved=[{"chunk_id": r["chunk_id"], "score": r["score"]} for r in retrieved],
        contexts=[r["text"] for r in retrieved],
        context_ids=[r["chunk_id"] for r in retrieved],
        context_scores=[r["score"] for r in retrieved],
        prompt_chars=len(prompt),
        prompt_words=_count_words(prompt),
        prompt_token_est=_token_estimate_from_text(prompt),
        llm_latency_s=llm_info.get("latency_s"),
        llm_info=llm_info,
        error=None,
    )


def _make_error_record(
    qid: str,
    question: str,
    prompt: str,
    retrieved: List[Dict[str, Any]],
    error: str,
) -> GenerationRecord:
    return GenerationRecord(
        id=qid,
        question=question,
        prompt=prompt,
        raw_output="",
        answer_raw="",
        citations_raw=[],
        pred_answer="",
        citations=[],
        invalid_citations=[],
        num_citations=0,
        explicit_nic=False,
        forced_nic_empty_retrieval=False,
        forced_nic_no_valid_citations=False,
        forced_not_in_context=False,
        is_grounded=False,
        final_status=FINAL_STATUS_ERROR,
        retrieved=[{"chunk_id": r["chunk_id"], "score": r["score"]} for r in retrieved],
        contexts=[r["text"] for r in retrieved],
        context_ids=[r["chunk_id"] for r in retrieved],
        context_scores=[r["score"] for r in retrieved],
        prompt_chars=len(prompt),
        prompt_words=_count_words(prompt),
        prompt_token_est=_token_estimate_from_text(prompt),
        llm_latency_s=None,
        llm_info={},
        error=error,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="BioASQ trace-driven generation from document-level traces")
    ap.add_argument("--trace_jsonl", required=True)
    ap.add_argument("--corpus", required=True)

    ap.add_argument(
        "--backend",
        choices=["openai", "openai_compat", "ollama"],
        default="openai_compat",
    )
    ap.add_argument("--lm_model", required=True)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max_tokens", type=int, default=600)
    ap.add_argument("--timeout_s", type=int, default=600)
    ap.add_argument("--openai_api_key", default=None)
    ap.add_argument("--openai_base_url", default=None)

    ap.add_argument("--top_k_context", type=int, default=10)
    ap.add_argument("--per_doc_max_chunks", type=int, default=50)
    ap.add_argument("--per_chunk_max_chars", type=int, default=2000)
    ap.add_argument("--model_ctx_tokens", type=int, default=None)
    ap.add_argument("--seed", type=int, default=224)
    ap.add_argument("--out_dir", required=True)

    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        force=True,
    )
    set_deterministic(args.seed, deterministic_torch=False)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    store = CorpusStore(args.corpus, strict=True)
    store.load()

    doc_to_chunks = _build_doc_to_chunks_index(args.corpus)
    if args.per_doc_max_chunks is not None and args.per_doc_max_chunks > 0:
        doc_to_chunks = {
            doc_id: chunk_ids[: args.per_doc_max_chunks]
            for doc_id, chunk_ids in doc_to_chunks.items()
        }

    llm_cfg = LLMConfig(
        backend=args.backend,
        model=args.lm_model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        timeout_s=args.timeout_s,
        api_key=args.openai_api_key,
        base_url=args.openai_base_url,
        num_ctx=args.model_ctx_tokens,
    )

    meta = {
        "task": "bioasq",
        "trace_jsonl": args.trace_jsonl,
        "corpus": args.corpus,
        "llm": _safe_llm_meta(llm_cfg),
        "top_k_context": args.top_k_context,
        "per_doc_max_chunks": args.per_doc_max_chunks,
        "per_chunk_max_chars": args.per_chunk_max_chars,
        "model_ctx_tokens": args.model_ctx_tokens,
        "seed": args.seed,
        "environment": _collect_env_metadata(),
        "trace_mode": "doc_level_retrieved_docs_expanded_to_chunks_round_robin",
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    out_path = out_dir / "generations.jsonl"

    n_total = 0
    n_skipped = 0
    n_no_retrieved = 0
    n_llm_calls = 0
    n_missing_text = 0
    n_explicit_nic = 0
    n_forced_nic_empty_retrieval = 0
    n_forced_nic_no_valid_citations = 0
    n_answered = 0
    n_error = 0
    latencies: List[float] = []

    with out_path.open("w", encoding="utf-8") as f:
        for tr in iter_jsonl(args.trace_jsonl):
            n_total += 1

            qid = str(tr.get("query_id") or tr.get("id") or "").strip()
            question = str(tr.get("question") or tr.get("query") or "").strip()

            if not qid or not question:
                n_skipped += 1
                continue

            retrieved, missing = _expand_retrieved_docs_to_chunks(
                tr,
                doc_to_chunks,
                store,
                args.top_k_context,
            )
            n_missing_text += missing

            used_chunk_ids = {r["chunk_id"] for r in retrieved}

            if not retrieved:
                n_no_retrieved += 1
                n_forced_nic_empty_retrieval += 1
                record = _make_forced_nic_empty_record(qid, question)
                f.write(record.to_json() + "\n")
                continue

            prompt = build_prompt(
                task="bioasq",
                question=question,
                retrieved=retrieved,
                per_chunk_max_chars=args.per_chunk_max_chars,
            )

            n_llm_calls += 1
            try:
                raw_output, llm_info = generate_with_info(prompt, llm_cfg)
                if llm_info.get("latency_s") is not None:
                    latencies.append(float(llm_info["latency_s"]))
            except Exception as exc:
                n_error += 1
                error_msg = str(exc)
                logger.warning("LLM error for id=%s: %s", qid, error_msg)
                record = _make_error_record(qid, question, prompt, retrieved, error_msg)
                f.write(record.to_json() + "\n")
                continue

            record = _classify_llm_output(
                qid=qid,
                question=question,
                prompt=prompt,
                raw=raw_output,
                llm_info=llm_info,
                retrieved=retrieved,
                used_chunk_ids=used_chunk_ids,
            )

            if record.final_status == FINAL_STATUS_EXPLICIT_NIC:
                n_explicit_nic += 1
            elif record.final_status == FINAL_STATUS_FORCED_NIC_NO_CIT:
                n_forced_nic_no_valid_citations += 1
            else:
                n_answered += 1

            f.write(record.to_json() + "\n")

    n_valid = n_total - n_skipped
    n_nic_total = (
        n_explicit_nic
        + n_forced_nic_empty_retrieval
        + n_forced_nic_no_valid_citations
    )

    generation_summary = {
        "task": "bioasq",
        "model": args.lm_model,
        "backend": args.backend,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "counts": {
            "total_trace_rows": n_total,
            "skipped_no_id_or_question": n_skipped,
            "valid_evaluated": n_valid,
            "llm_calls": n_llm_calls,
            "missing_doc_to_chunk_mappings": n_missing_text,
        },
        "outcome_counts": {
            "answered": n_answered,
            "explicit_nic": n_explicit_nic,
            "forced_nic_empty_retrieval": n_forced_nic_empty_retrieval,
            "forced_nic_no_valid_citations": n_forced_nic_no_valid_citations,
            "nic_total": n_nic_total,
            "error": n_error,
        },
        "outcome_rates": {
            "answered_rate": n_answered / n_valid if n_valid else None,
            "explicit_nic_rate": n_explicit_nic / n_valid if n_valid else None,
            "forced_nic_empty_retrieval_rate": n_forced_nic_empty_retrieval / n_valid if n_valid else None,
            "forced_nic_no_valid_citations_rate": n_forced_nic_no_valid_citations / n_valid if n_valid else None,
            "nic_total_rate": n_nic_total / n_valid if n_valid else None,
            "error_rate": n_error / n_valid if n_valid else None,
        },
        "latency_stats_s": {
            "n": len(latencies),
            "mean": statistics.mean(latencies) if latencies else None,
            "median": statistics.median(latencies) if latencies else None,
            "min": min(latencies) if latencies else None,
            "max": max(latencies) if latencies else None,
            "stdev": statistics.stdev(latencies) if len(latencies) > 1 else None,
        },
    }
    (out_dir / "generation_summary.json").write_text(
        json.dumps(generation_summary, indent=2), encoding="utf-8"
    )

    logger.info(
        "Finished. out=%s total=%d skipped=%d valid=%d answered=%d "
        "explicit_nic=%d forced_empty=%d forced_no_cit=%d error=%d llm_calls=%d",
        out_path.as_posix(),
        n_total,
        n_skipped,
        n_valid,
        n_answered,
        n_explicit_nic,
        n_forced_nic_empty_retrieval,
        n_forced_nic_no_valid_citations,
        n_error,
        n_llm_calls,
    )


if __name__ == "__main__":
    main()