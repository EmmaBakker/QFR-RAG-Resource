#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import os
import platform
import re
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.retrieval.determinism import set_deterministic
from src.retrieval.utils_io import iter_jsonl

from .llm_client import LLMConfig, generate_with_info
from .schema import FINAL_STATUS_ANSWERED, FINAL_STATUS_ERROR

logger = logging.getLogger(__name__)

UNKNOWN_TOKEN = "UNKNOWN"
PROMPT_VERSION = "closed_book_direct_v1"

_FENCED_JSON_RX = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.IGNORECASE | re.DOTALL)


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


def _count_words(text: str) -> int:
    return 0 if not text else len(text.split())


def _token_estimate_from_text(text: str) -> int:
    if not text:
        return 0
    return max(1, int(len(text) / 4))


def _load_dataset(dataset_path: str, task: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    seen_ids: set[str] = set()

    for row in iter_jsonl(dataset_path):
        if task.upper() == "A":
            task_value = str(row.get("task", "A") or "A").strip().upper()
            if task_value != "A":
                continue

        qid = str(row.get("id") or row.get("query_id") or row.get("question_id") or "").strip()
        question = str(row.get("question") or row.get("query") or row.get("text") or row.get("prompt") or "").strip()

        if not qid or not question:
            logger.warning("Skipping row without id/question: %s", row)
            continue
        if qid in seen_ids:
            logger.warning("Skipping duplicate id=%s", qid)
            continue

        seen_ids.add(qid)
        rows.append(row)

    return rows


def build_closed_book_prompt(question: str, unknown_token: str = UNKNOWN_TOKEN) -> str:
    """
    Closed-book prompt: no context, no citations, no external tools.

    The key design choice is that uncertainty maps to UNKNOWN, not NOT IN CONTEXT.
    NOT IN CONTEXT is a RAG/corpus-grounding concept; UNKNOWN is a closed-book
    abstention concept.
    """
    return f"""
    Answer the following question as accurately and concisely as possible.

    Return ONLY valid JSON with exactly this schema:
    {{
      "answer": "<answer>"
    }}

    Question:
    {question}
    """.strip()


def _extract_fenced_or_raw(text: str) -> str:
    t = (text or "").strip()
    if not t:
        return ""
    m = _FENCED_JSON_RX.search(t)
    return (m.group(1) if m else t).strip()


def _extract_first_balanced_json_object(text: str) -> str:
    t = _extract_fenced_or_raw(text)
    if not t:
        return ""

    start = t.find("{")
    if start < 0:
        return ""

    depth = 0
    in_str = False
    escape = False

    for i in range(start, len(t)):
        ch = t[i]

        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
            continue

        if ch == '"':
            in_str = True
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return t[start:i + 1]

    return ""


def _try_parse_json_obj(raw: str) -> Optional[Dict[str, Any]]:
    candidate = _extract_fenced_or_raw(raw)
    if not candidate:
        return None

    try:
        obj = json.loads(candidate)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    obj_str = _extract_first_balanced_json_object(candidate)
    if not obj_str:
        return None

    try:
        obj2 = json.loads(obj_str)
        return obj2 if isinstance(obj2, dict) else None
    except Exception:
        return None


def _first_nonempty_line(raw: str) -> str:
    for line in (raw or "").splitlines():
        line = line.strip()
        if line:
            return line
    return ""


def _canonicalize_unknown(answer: str, unknown_token: str = UNKNOWN_TOKEN) -> str:
    stripped = (answer or "").strip()
    if not stripped:
        return ""

    comparable = stripped.strip().strip(" .!?:;\"'").upper()
    if comparable == unknown_token.upper():
        return unknown_token
    return stripped


def parse_closed_book_answer(
    raw: str,
    unknown_token: str = UNKNOWN_TOKEN,
) -> Tuple[str, str]:
    raw = raw or ""
    if not raw.strip():
        return "", "empty_output"

    obj = _try_parse_json_obj(raw)
    if obj is not None:
        answer = str(obj.get("answer", "") or "").strip()
        if answer:
            return _canonicalize_unknown(answer, unknown_token), "json_ok"
        return "", "json_missing_answer"

    fallback = _first_nonempty_line(raw)
    if fallback:
        return _canonicalize_unknown(fallback, unknown_token), "plaintext_fallback"

    return "", "empty_output"


def _is_unknown(answer: str, unknown_token: str = UNKNOWN_TOKEN) -> bool:
    return (answer or "").strip().upper() == unknown_token.upper()


def _make_generation_record(
    *,
    qid: str,
    question: str,
    prompt: str,
    raw_output: str,
    answer_raw: str,
    pred_answer: str,
    llm_info: Dict[str, Any],
    parse_status: str,
    unknown_token: str,
    task: str,
) -> Dict[str, Any]:
    is_unknown = _is_unknown(pred_answer, unknown_token)

    return {
        "id": qid,
        "question": question,
        "prompt": prompt,
        "raw_output": raw_output,
        "answer_raw": answer_raw,
        "citations_raw": [],
        "pred_answer": pred_answer,
        "citations": [],
        "invalid_citations": [],
        "num_citations": 0,

        "explicit_nic": False,
        "forced_nic_empty_retrieval": False,
        "forced_nic_no_valid_citations": False,
        "forced_not_in_context": False,
        "is_grounded": False,
        "final_status": FINAL_STATUS_ANSWERED,

        "retrieved": [],
        "contexts": [],
        "context_ids": [],
        "context_scores": [],

        "prompt_chars": len(prompt),
        "prompt_words": _count_words(prompt),
        "prompt_token_est": _token_estimate_from_text(prompt),
        "llm_latency_s": llm_info.get("latency_s"),
        "llm_info": llm_info,
        "error": None,

        "closed_book": True,
        "closed_book_task": task.upper(),
        "closed_book_prompt_version": PROMPT_VERSION,
        "closed_book_unknown_token": unknown_token,
        "closed_book_parse_status": parse_status,
        "closed_book_answer_status": "unknown_abstention" if is_unknown else "answered",
    }


def _make_error_record(
    *,
    qid: str,
    question: str,
    prompt: str,
    error: str,
    unknown_token: str,
    task: str,
) -> Dict[str, Any]:
    return {
        "id": qid,
        "question": question,
        "prompt": prompt,
        "raw_output": "",
        "answer_raw": "",
        "citations_raw": [],
        "pred_answer": "",
        "citations": [],
        "invalid_citations": [],
        "num_citations": 0,

        "explicit_nic": False,
        "forced_nic_empty_retrieval": False,
        "forced_nic_no_valid_citations": False,
        "forced_not_in_context": False,
        "is_grounded": False,
        "final_status": FINAL_STATUS_ERROR,

        "retrieved": [],
        "contexts": [],
        "context_ids": [],
        "context_scores": [],

        "prompt_chars": len(prompt),
        "prompt_words": _count_words(prompt),
        "prompt_token_est": _token_estimate_from_text(prompt),
        "llm_latency_s": None,
        "llm_info": {},
        "error": error,

        "closed_book": True,
        "closed_book_task": task.upper(),
        "closed_book_prompt_version": PROMPT_VERSION,
        "closed_book_unknown_token": unknown_token,
        "closed_book_parse_status": "generation_error",
        "closed_book_answer_status": "generation_error",
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Closed-book / direct-inference generation for Task A or Task B."
    )
    ap.add_argument("--task", choices=["A", "B"], required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out_dir", required=True)

    ap.add_argument(
        "--backend",
        choices=["openai", "openai_compat", "ollama", "anthropic"],
        default="openai_compat",
    )
    ap.add_argument("--lm_model", required=True)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max_tokens", type=int, default=300)
    ap.add_argument("--timeout_s", type=int, default=600)
    ap.add_argument("--openai_api_key", default=None)
    ap.add_argument("--openai_base_url", default=None)
    ap.add_argument("--anthropic_api_key", default=None)
    ap.add_argument("--model_ctx_tokens", type=int, default=None)
    ap.add_argument("--seed", type=int, default=224)
    ap.add_argument("--unknown_token", default=UNKNOWN_TOKEN)
    ap.add_argument("--max_examples", type=int, default=None)

    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        force=True,
    )

    set_deterministic(args.seed, deterministic_torch=False)

    task = args.task.upper()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset_rows = _load_dataset(args.dataset, task)
    if args.max_examples is not None:
        dataset_rows = dataset_rows[: int(args.max_examples)]

    if not dataset_rows:
        raise SystemExit(f"No Task {task} examples found in dataset: {args.dataset}")

    api_key = args.anthropic_api_key if args.backend == "anthropic" else args.openai_api_key
    if args.backend == "anthropic" and api_key is None:
        api_key = os.getenv("ANTHROPIC_API_KEY")
    if args.backend in {"openai", "openai_compat"} and api_key is None:
        api_key = os.getenv("OPENAI_API_KEY")

    llm_cfg = LLMConfig(
        backend=args.backend,
        model=args.lm_model,
        temperature=float(args.temperature),
        max_tokens=int(args.max_tokens),
        timeout_s=int(args.timeout_s),
        api_key=api_key,
        base_url=args.openai_base_url if args.backend == "openai_compat" else None,
        num_ctx=args.model_ctx_tokens,
    )

    meta = {
        "task": task,
        "condition": "closed_book_direct_inference",
        "dataset": args.dataset,
        "output_file": "generations.jsonl",
        "prompt_version": PROMPT_VERSION,
        "unknown_token": args.unknown_token,
        "interpretation": (
            "This condition uses no retrieved chunks and no citations. It estimates "
            f"how often the model can answer Task {task} from internal/parametric knowledge "
            "alone. UNKNOWN is a closed-book abstention, not a RAG NOT IN CONTEXT decision."
        ),
        "llm": {
            "backend": llm_cfg.backend,
            "model": llm_cfg.model,
            "temperature": llm_cfg.temperature,
            "max_tokens": llm_cfg.max_tokens,
            "timeout_s": llm_cfg.timeout_s,
            "base_url": llm_cfg.base_url,
            "model_ctx_tokens": llm_cfg.num_ctx,
        },
        "seed": args.seed,
        "environment": _collect_env_metadata(),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    out_path = out_dir / "generations.jsonl"

    n_total = 0
    n_answered = 0
    n_unknown = 0
    n_error = 0
    parse_status_counts: Dict[str, int] = {}
    latencies: List[float] = []

    with out_path.open("w", encoding="utf-8") as f:
        for row in dataset_rows:
            n_total += 1
            qid = str(row.get("id") or row.get("query_id") or row.get("question_id") or "").strip()
            question = str(row.get("question") or row.get("query") or row.get("text") or row.get("prompt") or "").strip()
            prompt = build_closed_book_prompt(question, args.unknown_token)

            try:
                raw_output, llm_info = generate_with_info(prompt, llm_cfg)
                if llm_info.get("latency_s") is not None:
                    latencies.append(float(llm_info["latency_s"]))

                answer, parse_status = parse_closed_book_answer(
                    raw=raw_output,
                    unknown_token=args.unknown_token,
                )
                parse_status_counts[parse_status] = parse_status_counts.get(parse_status, 0) + 1

                if not answer:
                    n_error += 1
                    record = _make_error_record(
                        qid=qid,
                        question=question,
                        prompt=prompt,
                        error=f"No usable answer parsed from output; parse_status={parse_status}",
                        unknown_token=args.unknown_token,
                        task=task,
                    )
                else:
                    if _is_unknown(answer, args.unknown_token):
                        n_unknown += 1
                    else:
                        n_answered += 1

                    record = _make_generation_record(
                        qid=qid,
                        question=question,
                        prompt=prompt,
                        raw_output=raw_output,
                        answer_raw=answer,
                        pred_answer=answer,
                        llm_info=llm_info,
                        parse_status=parse_status,
                        unknown_token=args.unknown_token,
                        task=task,
                    )

            except Exception as exc:
                n_error += 1
                logger.warning("LLM error for id=%s: %s", qid, exc)
                record = _make_error_record(
                    qid=qid,
                    question=question,
                    prompt=prompt,
                    error=str(exc),
                    unknown_token=args.unknown_token,
                    task=task,
                )

            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    summary = {
        "task": task,
        "condition": "closed_book_direct_inference",
        "model": args.lm_model,
        "backend": args.backend,
        "prompt_version": PROMPT_VERSION,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "counts": {
            "n_total": n_total,
            "n_answered_non_unknown": n_answered,
            "n_unknown_abstention": n_unknown,
            "n_generation_error": n_error,
        },
        "rates": {
            "answered_non_unknown_rate": n_answered / n_total if n_total else None,
            "unknown_abstention_rate": n_unknown / n_total if n_total else None,
            "generation_error_rate": n_error / n_total if n_total else None,
        },
        "parse_status_counts": parse_status_counts,
        "latency_stats_s": {
            "n": len(latencies),
            "mean": statistics.mean(latencies) if latencies else None,
            "median": statistics.median(latencies) if latencies else None,
            "min": min(latencies) if latencies else None,
            "max": max(latencies) if latencies else None,
            "stdev": statistics.stdev(latencies) if len(latencies) > 1 else None,
        },
        "important_note": (
            "For this baseline, contexts/context_ids are intentionally empty. "
            "Do not report retrieval-aware taxonomy categories from evaluate_taskA.py; "
            "report closed-book answer/UNKNOWN rates and judged correctness."
        ),
    }
    (out_dir / "closed_book_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (out_dir / "generation_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    logger.info(
        "Finished closed-book Task %s generation: out=%s total=%d answered=%d unknown=%d error=%d",
        task,
        out_path.as_posix(),
        n_total,
        n_answered,
        n_unknown,
        n_error,
    )


if __name__ == "__main__":
    main()