#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import platform
import statistics
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
Task = str  # "A" | "B" | "bioasq"


def _load_dataset_by_id(dataset_path: Optional[str]) -> Dict[str, Dict[str, Any]]:
    if not dataset_path:
        return {}
    out: Dict[str, Dict[str, Any]] = {}
    for row in iter_jsonl(dataset_path):
        rid = str(row.get("id", "")).strip()
        if rid:
            out[rid] = row
    return out


def _select_retrieved_list(task: Task, trace_row: Dict[str, Any]) -> List[Dict[str, Any]]:
    if task.lower() == "bioasq":
        return trace_row.get("retrieved_chunks") or trace_row.get("retrieved") or []
    return trace_row.get("retrieved") or []


def _task_name_for_prompt(task: Task) -> str:
    if task.upper() == "A":
        return "taskA"
    if task.upper() == "B":
        return "taskB"
    return "bioasq"


def _attach_text(
    store: CorpusStore,
    retrieved_list: List[Dict[str, Any]],
    top_k_context: int,
) -> Tuple[List[Dict[str, Any]], int]:
    selected = retrieved_list[: int(top_k_context)]
    retrieved: List[Dict[str, Any]] = []
    missing = 0

    for item in selected:
        cid = str(item.get("chunk_id", "")).strip()
        if not cid:
            continue
        score = float(item.get("score", 0.0) or 0.0)
        txt = store.get_text(cid)
        if txt is None:
            missing += 1
            txt = ""
        retrieved.append({"chunk_id": cid, "score": score, "text": txt})

    return retrieved, missing


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


def _make_forced_nic_empty_record(
    qid: str, question: str
) -> GenerationRecord:
    """Case 1 — no chunks retrieved; LLM never called."""
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
    """
    Parse raw model output and classify outcome (Cases 2–4).
    Invoked only after a successful LLM call.
    """
    # parse_all calls _try_parse_json exactly once (vs. three separate calls)
    parsed = parse_all(raw)
    ans_raw = parsed.answer
    pred = ans_raw.strip() if ans_raw else NIC_TOKEN
    citations_raw = parsed.citations

    valid_citations = [c for c in citations_raw if c in used_chunk_ids]
    invalid_citations = [c for c in citations_raw if c not in used_chunk_ids]

    # Classify outcome.  Mutually exclusive, evaluated in order:
    #   explicit_nic           — model returned NIC_TOKEN
    #   forced_nic_no_cit      — answered but no valid citations
    #   answered               — answered with valid citations
    #
    # No implicit normalization: semantic refusal phrases are not mapped to NIC.
    explicit_nic = pred.upper() == NIC_TOKEN.upper()
    forced_nic_no_valid_citations = False

    if explicit_nic:
        # Case 2: model explicitly abstained
        final_status = FINAL_STATUS_EXPLICIT_NIC
        valid_citations = []
        invalid_citations = []

    elif not valid_citations:
        # Case 3: model answered but grounding failed — citation-level forced NIC
        pred = NIC_TOKEN
        forced_nic_no_valid_citations = True
        final_status = FINAL_STATUS_FORCED_NIC_NO_CIT
        valid_citations = []
        invalid_citations = []

    else:
        # Case 4: model answered with valid citations
        final_status = FINAL_STATUS_ANSWERED

    # forced_not_in_context covers both forced flags but NOT explicit_nic:
    # explicit_nic is a model decision, not a system enforcement.
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
    """
    Case 5 — LLM call raised an exception.

    This is NOT classified as explicit_nic.  The model never ran; treating
    the failure as a deliberate abstention would corrupt outcome counts and
    prevent diagnosis of infrastructure problems.
    """
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
    ap = argparse.ArgumentParser(description="Trace-driven generation")
    ap.add_argument("--task", choices=["A", "B", "bioasq"], required=True)
    ap.add_argument("--trace_jsonl", required=True)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--dataset", default=None)

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

    if args.task.upper() == "B" and not args.dataset:
        raise SystemExit("Task B requires --dataset")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    store = CorpusStore(args.corpus, strict=True)
    store.load()

    _dataset_by_id = _load_dataset_by_id(args.dataset)  # reserved for Task B metadata

    llm_cfg = LLMConfig(
        backend=args.backend,
        model=args.lm_model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        timeout_s=args.timeout_s,
        api_key=args.openai_api_key,
        base_url=args.openai_base_url,
        num_ctx=args.model_ctx_tokens,  # None → Ollama uses model default
    )

    meta = {
        "task": args.task,
        "trace_jsonl": args.trace_jsonl,
        "corpus": args.corpus,
        "dataset": args.dataset,
        "llm": _safe_llm_meta(llm_cfg),
        "top_k_context": args.top_k_context,
        "per_chunk_max_chars": args.per_chunk_max_chars,
        "model_ctx_tokens": args.model_ctx_tokens,
        "seed": args.seed,
        "environment": _collect_env_metadata(),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    out_path = out_dir / "generations.jsonl"
    task_prompt_name = _task_name_for_prompt(args.task)

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
            question = (
                tr.get("question")
                or tr.get("query")
                or tr.get("text")
                or tr.get("prompt")
                or ""
            )
            question = str(question).strip()

            if not qid or not question:
                n_skipped += 1
                continue

            retrieved_raw = _select_retrieved_list(args.task, tr)
            retrieved, missing = _attach_text(store, retrieved_raw, args.top_k_context)
            n_missing_text += missing

            used_chunk_ids = {r["chunk_id"] for r in retrieved}

            # ── Case 1: No retrieved chunks ────────────────────────────────────
            # The LLM is not called.  explicit_nic is False because the model
            # never ran; this is a retrieval failure, not a model decision.
            if not retrieved:
                n_no_retrieved += 1
                n_forced_nic_empty_retrieval += 1
                record = _make_forced_nic_empty_record(qid, question)
                f.write(record.to_json() + "\n")
                continue

            prompt = build_prompt(
                task=task_prompt_name,
                question=question,
                retrieved=retrieved,
                per_chunk_max_chars=args.per_chunk_max_chars,
            )

            # ── LLM call ───────────────────────────────────────────────────────
            n_llm_calls += 1
            raw_output: str
            llm_info: Dict[str, Any]
            try:
                raw_output, llm_info = generate_with_info(prompt, llm_cfg)
                if llm_info.get("latency_s") is not None:
                    latencies.append(float(llm_info["latency_s"]))
            except Exception as exc:
                # ── Case 5: LLM call failed ────────────────────────────────────
                # Recorded as final_status="error".  NOT classified as
                # explicit_nic — errors must stay distinguishable from deliberate
                # model abstentions so outcome counts stay meaningful.
                n_error += 1
                error_msg = str(exc)
                logger.warning("LLM error for id=%s: %s", qid, error_msg)
                record = _make_error_record(qid, question, prompt, retrieved, error_msg)
                f.write(record.to_json() + "\n")
                continue

            # ── Cases 2–4: classify successful LLM response ────────────────────
            record = _classify_llm_output(
                qid, question, prompt, raw_output, llm_info,
                retrieved, used_chunk_ids,
            )

            if record.final_status == FINAL_STATUS_EXPLICIT_NIC:
                n_explicit_nic += 1
            elif record.final_status == FINAL_STATUS_FORCED_NIC_NO_CIT:
                n_forced_nic_no_valid_citations += 1
            else:
                n_answered += 1

            f.write(record.to_json() + "\n")

    # generation_summary.json — makes generation a reportable experimental stage.
    # Rates use n_valid (non-skipped) as denominator; skipped rows were never
    # evaluated and including them distorts outcome rates.
    n_valid = n_total - n_skipped
    n_nic_total = n_explicit_nic + n_forced_nic_empty_retrieval + n_forced_nic_no_valid_citations

    generation_summary = {
        "task": args.task,
        "model": args.lm_model,
        "backend": args.backend,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "counts": {
            "total_trace_rows": n_total,
            "skipped_no_id_or_question": n_skipped,
            "valid_evaluated": n_valid,
            "llm_calls": n_llm_calls,
            "missing_chunk_texts": n_missing_text,
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

