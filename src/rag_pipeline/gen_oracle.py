#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import platform
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.retrieval.determinism import set_deterministic
from src.retrieval.utils_io import iter_jsonl

from .corpus_store import CorpusStore
from .llm_client import LLMConfig, _safe_llm_meta, generate_with_info
from .parse_output import parse_all
from .prompts import build_prompt
from .schema import (
    FINAL_STATUS_ANSWERED,
    FINAL_STATUS_ERROR,
    FINAL_STATUS_EXPLICIT_NIC,
    FINAL_STATUS_FORCED_NIC_EMPTY,
    FINAL_STATUS_FORCED_NIC_NO_CIT,
    GenerationRecord,
    NIC_TOKEN,
)

logger = logging.getLogger(__name__)


# General helpers

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


def _load_dataset_by_id(dataset_path: str) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for row in iter_jsonl(dataset_path):
        rid = str(row.get("id", "")).strip()
        if rid:
            out[rid] = row
    return out


def _dedupe_keep_order(items: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for item in items:
        item = str(item).strip()
        if not item or item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def _safe_mean(xs: List[float]) -> Optional[float]:
    return statistics.mean(xs) if xs else None


def _safe_median(xs: List[float]) -> Optional[float]:
    return statistics.median(xs) if xs else None


def _safe_stdev(xs: List[float]) -> Optional[float]:
    return statistics.stdev(xs) if len(xs) > 1 else None


def _clip_to_k(items: List[str], k: Optional[int]) -> List[str]:
    if k is None or k <= 0:
        return items
    return items[:k]

# Oracle chunk resolution


def _gold_ev_to_chunk_id(ev: Dict[str, Any]) -> Optional[str]:
    doc_id = str(ev.get("doc_id", "")).strip()
    chunk_id = str(ev.get("chunk_id", "")).strip()
    if not doc_id or not chunk_id:
        return None
    return f"{doc_id}_{chunk_id}"


def _ordered_oracle_chunk_ids_taskA(
    row: Dict[str, Any],
    mode: str,
    max_oracle_chunks: Optional[int],
) -> List[str]:
    gold_evidence = row.get("gold_evidence") or []
    if not isinstance(gold_evidence, list):
        return []

    chunk_ids: List[str] = []
    for ev in gold_evidence:
        if not isinstance(ev, dict):
            continue
        cid = _gold_ev_to_chunk_id(ev)
        if cid:
            chunk_ids.append(cid)

    chunk_ids = _dedupe_keep_order(chunk_ids)

    if mode == "one_per_slot":
        chunk_ids = chunk_ids[:1]

    return _clip_to_k(chunk_ids, max_oracle_chunks)


def _ordered_oracle_chunk_ids_taskB(
    row: Dict[str, Any],
    mode: str,
    max_oracle_chunks: Optional[int],
) -> List[str]:
    required_slots = row.get("required_slots")
    try:
        required_slots_n = int(required_slots) if required_slots is not None else 3
    except Exception:
        required_slots_n = 3

    if required_slots_n < 1:
        required_slots_n = 1
    if required_slots_n > 3:
        logger.warning(
            "qid=%s has required_slots=%s but only slot_A/B/C are supported; capping at 3",
            row.get("id", "?"),
            required_slots,
        )
        required_slots_n = 3

    slot_names = ["slot_A", "slot_B", "slot_C"][:required_slots_n]

    slot_lists: List[List[str]] = []
    for slot_name in slot_names:
        vals = row.get(slot_name) or []
        if not isinstance(vals, list):
            slot_lists.append([])
            continue
        slot_lists.append(_dedupe_keep_order([str(x) for x in vals]))

    if mode == "one_per_slot":
        chosen: List[str] = []
        for vals in slot_lists:
            if vals:
                chosen.append(vals[0])
        chosen = _dedupe_keep_order(chosen)
        return _clip_to_k(chosen, max_oracle_chunks)

    # all_gold + slot-aware fill:
    # 1) guarantee at least one chunk per required non-empty slot
    # 2) fill remaining budget with the rest in original slot order
    chosen: List[str] = []
    remainder: List[str] = []

    for vals in slot_lists:
        if not vals:
            continue
        chosen.append(vals[0])
        remainder.extend(vals[1:])

    chosen = _dedupe_keep_order(chosen)
    chosen_set = set(chosen)

    remainder = _dedupe_keep_order([x for x in remainder if x not in chosen_set])

    if max_oracle_chunks is None or max_oracle_chunks <= 0:
        return _dedupe_keep_order(chosen + remainder)

    if len(chosen) >= max_oracle_chunks:
        return chosen[:max_oracle_chunks]

    room = max_oracle_chunks - len(chosen)
    return _dedupe_keep_order(chosen + remainder[:room])



# Oracle text attachment


def _attach_oracle_texts(
    store: CorpusStore,
    chunk_ids: List[str],
    qid: str = "?",
) -> Tuple[List[Dict[str, Any]], List[str], List[str]]:
    retrieved: List[Dict[str, Any]] = []
    found_ids: List[str] = []
    missing_ids: List[str] = []
    n_requested = len(chunk_ids)

    for rank, cid in enumerate(chunk_ids, start=1):
        txt = store.get_text(cid)
        if txt is None:
            missing_ids.append(cid)
            logger.warning(
                "qid=%s: oracle chunk '%s' not found in corpus store (rank=%d/%d)",
                qid, cid, rank, n_requested,
            )
            continue

        found_ids.append(cid)
        retrieved.append({
            "chunk_id": cid,
            "score": float(n_requested - rank + 1),
            "text": txt,
        })

    return retrieved, found_ids, missing_ids

# Record builders


def _build_oracle_meta(
    oracle_mode: str,
    oracle_chunk_ids: List[str],
    found_ids: List[str],
    missing_ids: List[str],
    max_oracle_chunks: Optional[int],
) -> Dict[str, Any]:
    return {
        "oracle_mode": oracle_mode,
        "max_oracle_chunks": max_oracle_chunks,
        "oracle_chunk_ids_requested": oracle_chunk_ids,
        "oracle_chunk_ids_found": found_ids,
        "oracle_chunk_ids_missing": missing_ids,
        "oracle_requested_n": len(oracle_chunk_ids),
        "oracle_found_n": len(found_ids),
        "oracle_missing_n": len(missing_ids),
    }


def _make_forced_nic_empty_record(
    qid: str,
    question: str,
    llm_info: Optional[Dict[str, Any]] = None,
) -> GenerationRecord:
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
        llm_info=llm_info or {},
        error=None,
    )


def _make_error_record(
    qid: str,
    question: str,
    prompt: str,
    retrieved: List[Dict[str, Any]],
    error: str,
    llm_info: Optional[Dict[str, Any]] = None,
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
        llm_info=llm_info or {},
        error=error,
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



# Main


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Oracle generation from gold evidence for Task A or Task B"
    )
    ap.add_argument("--task", choices=["A", "B"], required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--corpus", required=True)
    ap.add_argument(
        "--oracle_mode",
        choices=["all_gold", "one_per_slot"],
        default="all_gold",
        help=(
            "Task A: all_gold = all annotated gold chunks, capped by --max_oracle_chunks; "
            "one_per_slot = first gold chunk only. "
            "Task B: all_gold = slot-aware oracle (at least one chunk per required slot, then fill); "
            "one_per_slot = first gold chunk per required slot."
        ),
    )
    ap.add_argument(
        "--max_oracle_chunks",
        type=int,
        default=10,
        help="Maximum number of oracle chunks to pass to the prompt. Use 10 to match the main RAG setup.",
    )
    ap.add_argument("--backend", choices=["openai", "openai_compat", "ollama"], default="openai_compat")
    ap.add_argument("--lm_model", required=True)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max_tokens", type=int, default=600)
    ap.add_argument("--timeout_s", type=int, default=600)
    ap.add_argument("--openai_api_key", default=None)
    ap.add_argument("--openai_base_url", default=None)
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

    dataset_by_id = _load_dataset_by_id(args.dataset)
    task_prompt_name = "taskA" if args.task == "A" else "taskB"

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
        "task": args.task,
        "mode": "oracle",
        "dataset": args.dataset,
        "corpus": args.corpus,
        "oracle_mode": args.oracle_mode,
        "max_oracle_chunks": args.max_oracle_chunks,
        "llm": _safe_llm_meta(llm_cfg),
        "per_chunk_max_chars": args.per_chunk_max_chars,
        "model_ctx_tokens": args.model_ctx_tokens,
        "seed": args.seed,
        "environment": _collect_env_metadata(),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    out_path = out_dir / "generations.jsonl"

    n_total = 0
    n_skipped = 0
    n_no_retrieved = 0
    n_llm_calls = 0
    n_questions_with_missing_oracle = 0
    n_missing_text_total = 0
    n_explicit_nic = 0
    n_forced_nic_empty_retrieval = 0
    n_forced_nic_no_valid_citations = 0
    n_answered = 0
    n_error = 0

    latencies: List[float] = []
    oracle_requested_sizes: List[int] = []
    oracle_found_sizes: List[int] = []
    oracle_missing_sizes: List[int] = []

    with out_path.open("w", encoding="utf-8") as f:
        for qid, row in sorted(dataset_by_id.items()):
            n_total += 1
            question = str(row.get("question", "")).strip()

            if not question:
                n_skipped += 1
                logger.warning("qid=%s: empty question, skipping", qid)
                continue

            if args.task == "A":
                oracle_chunk_ids = _ordered_oracle_chunk_ids_taskA(
                    row=row,
                    mode=args.oracle_mode,
                    max_oracle_chunks=args.max_oracle_chunks,
                )
            else:
                oracle_chunk_ids = _ordered_oracle_chunk_ids_taskB(
                    row=row,
                    mode=args.oracle_mode,
                    max_oracle_chunks=args.max_oracle_chunks,
                )

            oracle_chunk_ids = _dedupe_keep_order(oracle_chunk_ids)

            retrieved, found_ids, missing_ids = _attach_oracle_texts(
                store=store,
                chunk_ids=oracle_chunk_ids,
                qid=qid,
            )

            oracle_requested_sizes.append(len(oracle_chunk_ids))
            oracle_found_sizes.append(len(found_ids))
            oracle_missing_sizes.append(len(missing_ids))

            if missing_ids:
                n_questions_with_missing_oracle += 1
            n_missing_text_total += len(missing_ids)

            oracle_meta = _build_oracle_meta(
                oracle_mode=args.oracle_mode,
                oracle_chunk_ids=oracle_chunk_ids,
                found_ids=found_ids,
                missing_ids=missing_ids,
                max_oracle_chunks=args.max_oracle_chunks,
            )

            used_chunk_ids = {r["chunk_id"] for r in retrieved}

            if not retrieved:
                n_no_retrieved += 1
                n_forced_nic_empty_retrieval += 1
                record = _make_forced_nic_empty_record(
                    qid=qid,
                    question=question,
                    llm_info=oracle_meta,
                )
                f.write(record.to_json() + "\n")
                continue

            prompt = build_prompt(
                task=task_prompt_name,
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
                logger.warning("LLM error for id=%s: %s", qid, str(exc))
                record = _make_error_record(
                    qid=qid,
                    question=question,
                    prompt=prompt,
                    retrieved=retrieved,
                    error=str(exc),
                    llm_info=oracle_meta,
                )
                f.write(record.to_json() + "\n")
                continue

            record = _classify_llm_output(
                qid=qid,
                question=question,
                prompt=prompt,
                raw=raw_output,
                llm_info={**llm_info, **oracle_meta},
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
        "task": args.task,
        "mode": "oracle",
        "oracle_mode": args.oracle_mode,
        "max_oracle_chunks": args.max_oracle_chunks,
        "model": args.lm_model,
        "backend": args.backend,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "counts": {
            "total_dataset_rows": n_total,
            "skipped_no_question": n_skipped,
            "valid_evaluated": n_valid,
            "llm_calls": n_llm_calls,
            "questions_with_no_found_oracle_chunks": n_no_retrieved,
            "questions_with_missing_oracle_chunks": n_questions_with_missing_oracle,
            "missing_oracle_chunk_texts_total": n_missing_text_total,
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
        "oracle_context_stats": {
            "n_questions": len(oracle_requested_sizes),
            "requested": {
                "mean": _safe_mean([float(x) for x in oracle_requested_sizes]),
                "median": _safe_median([float(x) for x in oracle_requested_sizes]),
                "min": min(oracle_requested_sizes) if oracle_requested_sizes else None,
                "max": max(oracle_requested_sizes) if oracle_requested_sizes else None,
            },
            "found": {
                "mean": _safe_mean([float(x) for x in oracle_found_sizes]),
                "median": _safe_median([float(x) for x in oracle_found_sizes]),
                "min": min(oracle_found_sizes) if oracle_found_sizes else None,
                "max": max(oracle_found_sizes) if oracle_found_sizes else None,
            },
            "missing": {
                "mean": _safe_mean([float(x) for x in oracle_missing_sizes]),
                "median": _safe_median([float(x) for x in oracle_missing_sizes]),
                "min": min(oracle_missing_sizes) if oracle_missing_sizes else None,
                "max": max(oracle_missing_sizes) if oracle_missing_sizes else None,
            },
        },
        "latency_stats_s": {
            "n": len(latencies),
            "mean": _safe_mean(latencies),
            "median": _safe_median(latencies),
            "min": min(latencies) if latencies else None,
            "max": max(latencies) if latencies else None,
            "stdev": _safe_stdev(latencies),
        },
    }

    (out_dir / "generation_summary.json").write_text(
        json.dumps(generation_summary, indent=2),
        encoding="utf-8",
    )

    logger.info(
        "Finished oracle %s generation. out=%s total=%d skipped=%d valid=%d "
        "answered=%d explicit_nic=%d forced_empty=%d forced_no_cit=%d error=%d",
        args.task,
        out_path.as_posix(),
        n_total,
        n_skipped,
        n_valid,
        n_answered,
        n_explicit_nic,
        n_forced_nic_empty_retrieval,
        n_forced_nic_no_valid_citations,
        n_error,
    )


if __name__ == "__main__":
    main()