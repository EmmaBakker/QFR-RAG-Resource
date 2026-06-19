#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.retrieval.utils_io import iter_jsonl
from .llm_client import LLMConfig, generate
from .schema import NIC_TOKEN, GenerationRecord

logger = logging.getLogger(__name__)

FINAL_STATUS_ERROR_MALFORMED_ANSWER = "error_malformed_answer"
FINAL_STATUS_ERROR_GENERATION_FAILURE = "error_generation_failure"

NUGGET_PROMPT_VERSION = "v3_partial_nic_conservative_taxonomy"


# Helpers: IO + JSON robustness

def _read_jsonl_by_id(path: str) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for row in iter_jsonl(path):
        rid = str(row.get("id", "")).strip()
        if rid:
            out[rid] = row
    return out


def _safe_write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def _safe_slug(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", str(s)).strip("_") or "unknown"


_FENCED_JSON_RX = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.IGNORECASE | re.DOTALL)
_NIC_TEXT_RX = re.compile(rf"\b{re.escape(NIC_TOKEN)}\b", re.IGNORECASE)


def _extract_json_candidate(text: str) -> str:
    t = (text or "").strip()
    if not t:
        return ""
    m = _FENCED_JSON_RX.search(t)
    return (m.group(1) if m else t).strip()


def _extract_first_balanced_json_object(text: str) -> str:
    t = _extract_json_candidate(text)
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
            elif ch == "\"":
                in_str = False
            continue

        if ch == "\"":
            in_str = True
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return t[start:i + 1]
    return ""


def _try_parse_json_obj(text: str) -> Optional[Dict[str, Any]]:
    cand = _extract_json_candidate(text)
    if not cand:
        return None

    try:
        obj = json.loads(cand)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    obj_str = _extract_first_balanced_json_object(cand)
    if not obj_str:
        return None

    try:
        obj2 = json.loads(obj_str)
        return obj2 if isinstance(obj2, dict) else None
    except Exception:
        return None


def _is_system_not_in_context(pred_answer: str) -> bool:
    return (pred_answer or "").strip().upper() == NIC_TOKEN.upper()


def _looks_like_json_fragment(t: str) -> bool:
    tl = t.lower().strip()
    if not tl:
        return False
    json_markers = [
        '{"answer"',
        '{"citations"',
        '"answer":',
        '"citations":',
        "{'answer'",
        "{'citations'",
    ]
    return any(marker in tl for marker in json_markers)


def _is_only_punctuation_or_braces(t: str) -> bool:
    stripped = re.sub(r"[\s`'\"{}\[\]():,._-]+", "", t or "")
    return stripped == ""


def _is_malformed_answer(pred_answer: str) -> bool:
    """
    Detect clearly unusable answers that should be treated as system failures,
    not valid answers and not abstentions.

    Intentionally conservative, but broader than only fenced JSON.
    """
    t = (pred_answer or "").strip()
    tl = t.lower()

    if tl in {"", "```", "```json", "json", "null", "none", "n/a"}:
        return True

    if tl.startswith("```"):
        return True

    if _looks_like_json_fragment(t):
        return True

    if len(t) <= 12 and _is_only_punctuation_or_braces(t):
        return True

    if (t.startswith("{") or t.startswith("[")) and len(t) < 40:
        return True

    short_bad_prefixes = (
        "answer:",
        "citations:",
        '{"answer"',
        '{"citations"',
        "{'answer'",
        "{'citations'",
    )
    if len(tl) < 60 and tl.startswith(short_bad_prefixes):
        return True

    return False


def _detect_nic_text_present(pred_answer: str) -> bool:
    return bool(_NIC_TEXT_RX.search(pred_answer or ""))


def _detect_partial_nic_text_present(pred_answer: str) -> bool:
    return _detect_nic_text_present(pred_answer) and not _is_system_not_in_context(pred_answer)


# Example assembly

@dataclass
class Example:
    id: str
    question: str
    pred_answer: str
    answer_raw: str
    contexts: List[str]
    context_ids: List[str]
    slot_nuggets: Dict[str, List[str]]
    slot_evidence: Dict[str, List[str]]
    explicit_nic: bool
    forced_not_in_context: bool
    forced_nic_empty_retrieval: bool
    forced_nic_no_valid_citations: bool
    final_status: str
    is_grounded: bool
    is_malformed_answer: bool
    is_generation_error: bool
    error_message: Optional[str]


def _build_examples(
    generations_path: str,
    dataset_path: str,
    task: str,
) -> List[Example]:
    """
    Merge generation records with dataset rows.

    Important:
    - generation-time errors are retained as examples and treated as system failures
    - malformed answers are retained and treated as system failures
    - for Task B, slot_nuggets keys must exactly match slot_evidence keys
    """
    ds_by_id = _read_jsonl_by_id(dataset_path)

    examples: List[Example] = []
    n_missing_ds = 0
    n_missing_gens = 0
    seen_gen_ids: set = set()

    for raw_row in iter_jsonl(generations_path):
        g = GenerationRecord.from_dict(raw_row)
        rid = str(g.id or raw_row.get("id", "")).strip()
        if not rid:
            continue
        seen_gen_ids.add(rid)

        d = ds_by_id.get(rid)
        if d is None:
            n_missing_ds += 1
            continue

        question = str(g.question or raw_row.get("question", "") or d.get("question", "")).strip()
        if not question:
            continue

        slot_nuggets: Dict[str, List[str]] = {}
        slot_evidence: Dict[str, List[str]] = {}

        if task.upper() == "B":
            slot_nuggets_raw = d.get("slot_nuggets") or {}
            if isinstance(slot_nuggets_raw, dict):
                for k, v in slot_nuggets_raw.items():
                    if isinstance(v, list):
                        slot_nuggets[str(k)] = [str(x).strip() for x in v if str(x).strip()]

            slot_evidence_raw = d.get("slot_evidence") or {}
            if isinstance(slot_evidence_raw, dict) and slot_evidence_raw:
                for k, v in slot_evidence_raw.items():
                    if isinstance(v, list):
                        slot_evidence[str(k)] = [str(x).strip() for x in v if str(x).strip()]
            else:
                for skey in ("slot_A", "slot_B", "slot_C"):
                    val = d.get(skey)
                    if isinstance(val, list):
                        slot_evidence[skey] = [str(x).strip() for x in val if str(x).strip()]

            nugget_keys = set(slot_nuggets.keys())
            evidence_keys = set(slot_evidence.keys())

            if nugget_keys != evidence_keys:
                raise ValueError(
                    f"Task B slot key mismatch for id={rid}: "
                    f"slot_nuggets keys={sorted(nugget_keys)}; "
                    f"slot_evidence keys={sorted(evidence_keys)}; "
                    f"missing_in_nuggets={sorted(evidence_keys - nugget_keys)}; "
                    f"missing_in_evidence={sorted(nugget_keys - evidence_keys)}"
                )

        is_generation_error = bool(getattr(g, "is_error", False))
        error_message = str(getattr(g, "error", "") or raw_row.get("error", "")).strip() or None

        pred_answer = "" if is_generation_error else str(g.pred_answer or "")
        answer_raw = "" if is_generation_error else str(g.answer_raw or "")
        contexts = [] if is_generation_error else list(g.contexts or [])
        context_ids = [] if is_generation_error else list(g.context_ids or [])

        is_malformed = False if is_generation_error else _is_malformed_answer(pred_answer)

        if is_generation_error:
            final_status = FINAL_STATUS_ERROR_GENERATION_FAILURE
            explicit_nic = False
            forced_not_in_context = False
            forced_nic_empty_retrieval = False
            forced_nic_no_valid_citations = False
            is_grounded = False
        else:
            final_status = str(g.final_status or "")
            explicit_nic = bool(g.explicit_nic)
            forced_not_in_context = bool(g.forced_not_in_context)
            forced_nic_empty_retrieval = bool(g.forced_nic_empty_retrieval)
            forced_nic_no_valid_citations = bool(g.forced_nic_no_valid_citations)
            is_grounded = bool(g.is_grounded) and not is_malformed

        examples.append(
            Example(
                id=rid,
                question=question,
                pred_answer=pred_answer,
                answer_raw=answer_raw,
                contexts=contexts,
                context_ids=context_ids,
                slot_nuggets=slot_nuggets,
                slot_evidence=slot_evidence,
                explicit_nic=explicit_nic,
                forced_not_in_context=forced_not_in_context,
                forced_nic_empty_retrieval=forced_nic_empty_retrieval,
                forced_nic_no_valid_citations=forced_nic_no_valid_citations,
                final_status=final_status,
                is_grounded=is_grounded,
                is_malformed_answer=is_malformed,
                is_generation_error=is_generation_error,
                error_message=error_message,
            )
        )

    for rid in ds_by_id.keys():
        if rid not in seen_gen_ids:
            n_missing_gens += 1

    logger.info(
        "Loaded examples=%d (missing_dataset_for_generation=%d, missing_generation_for_dataset=%d)",
        len(examples),
        n_missing_ds,
        n_missing_gens,
    )
    return examples


# RAGAS integration

def _build_ragas_llm_wrapper(
    judge_backend: str,
    judge_model: str,
    temperature: float,
    max_tokens: int,
    timeout_s: int,
    api_key: Optional[str],
    base_url: Optional[str],
):
    try:
        from ragas.llms import LangchainLLMWrapper
    except Exception as e:
        raise ImportError("Missing dependency: ragas. Install with: pip install ragas") from e

    if judge_backend in {"openai", "openai_compat"}:
        try:
            from langchain_openai import ChatOpenAI
        except Exception as e:
            raise ImportError(
                "Missing: langchain-openai. Install with: pip install langchain-openai"
            ) from e

        chat_kwargs: Dict[str, Any] = {
            "model": judge_model,
            "temperature": float(temperature),
            "max_tokens": int(max_tokens),
            "timeout": timeout_s,
            "api_key": api_key or os.getenv("OPENAI_API_KEY"),
        }

        if judge_backend == "openai_compat":
            if not base_url:
                raise ValueError("openai_compat judge backend requires --openai_base_url.")
            chat_kwargs["base_url"] = base_url
            chat_kwargs["openai_api_base"] = base_url

        lc_llm = ChatOpenAI(**chat_kwargs)
        return LangchainLLMWrapper(lc_llm)

    if judge_backend == "anthropic":
        try:
            from langchain_anthropic import ChatAnthropic
        except Exception as e:
            raise ImportError(
                "Missing: langchain-anthropic. Install with: pip install langchain-anthropic"
            ) from e

        chat_kwargs = {
            "model": judge_model,
            "temperature": float(temperature),
            "timeout": timeout_s,
            "max_tokens": int(max_tokens),
            "api_key": api_key or os.getenv("ANTHROPIC_API_KEY"),
        }
        lc_llm = ChatAnthropic(**chat_kwargs)
        return LangchainLLMWrapper(lc_llm)

    raise ValueError(f"Unsupported RAGAS judge backend: {judge_backend}")


def _build_ragas_embedding_wrapper(
    emb_model: str,
    emb_api_key: Optional[str],
    emb_base_url: Optional[str],
    emb_backend: str = "openai",
):
    try:
        from ragas.embeddings import LangchainEmbeddingsWrapper
    except Exception as e:
        raise ImportError("Missing dependency: ragas. Install with: pip install ragas") from e

    if emb_backend not in {"openai", "openai_compat"}:
        raise ValueError(
            "RAGAS embeddings currently support only openai/openai_compat backends."
        )

    try:
        from langchain_openai import OpenAIEmbeddings
    except Exception as e:
        raise ImportError(
            "Missing: langchain-openai. Install with: pip install langchain-openai"
        ) from e

    emb_kwargs: Dict[str, Any] = {
        "model": emb_model,
        "api_key": emb_api_key or os.getenv("OPENAI_API_KEY"),
    }

    if emb_backend == "openai_compat":
        if not emb_base_url:
            raise ValueError(
                "openai_compat embedding backend requires --ragas_embedding_base_url."
            )
        emb_kwargs["base_url"] = emb_base_url
        emb_kwargs["openai_api_base"] = emb_base_url

    lc_emb = OpenAIEmbeddings(**emb_kwargs)
    return LangchainEmbeddingsWrapper(lc_emb)


def run_ragas(
    examples: List[Example],
    judge_cfg: LLMConfig,
    metrics: List[str],
    ragas_embedding_model: str,
    ragas_embedding_backend: str,
    ragas_embedding_api_key: Optional[str],
    ragas_embedding_base_url: Optional[str],
    anthropic_batch_size: int = 5,
    anthropic_batch_sleep_s: float = 1.0,
) -> Tuple[Dict[str, float], List[Dict[str, Any]], Dict[str, Any]]:
    """
    Returns:
      (summary_means, per_example_rows, diagnostics)

    Canonical NIC, malformed answers, and hard generation failures are excluded
    from actual RAGAS calls. They are handled during aggregation:
      - filter_nic variant: excluded entirely
      - zero_nic variant: counted as 0
    """
    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import (
            Faithfulness,
            ResponseRelevancy,
            ContextPrecision,
            ContextRecall,
        )
    except Exception as e:
        raise ImportError("Missing ragas/datasets.") from e

    ragas_llm = _build_ragas_llm_wrapper(
        judge_backend=judge_cfg.backend,
        judge_model=judge_cfg.model,
        temperature=judge_cfg.temperature,
        max_tokens=judge_cfg.max_tokens,
        timeout_s=judge_cfg.timeout_s,
        api_key=judge_cfg.api_key,
        base_url=judge_cfg.base_url,
    )

    ragas_emb = _build_ragas_embedding_wrapper(
        emb_model=ragas_embedding_model,
        emb_api_key=ragas_embedding_api_key,
        emb_base_url=ragas_embedding_base_url,
        emb_backend=ragas_embedding_backend,
    )

    metric_map = {
        "faithfulness": Faithfulness(llm=ragas_llm),
        "answer_relevance": ResponseRelevancy(llm=ragas_llm, embeddings=ragas_emb),
        "context_precision": ContextPrecision(llm=ragas_llm),
        "context_recall": ContextRecall(llm=ragas_llm),
    }

    selected = []
    for m in metrics:
        if m not in metric_map:
            raise ValueError(f"Unknown RAGAS metric: {m}. Allowed: {sorted(metric_map.keys())}")
        selected.append(metric_map[m])

    col_map = {
        "faithfulness": "faithfulness",
        "answer_relevance": "answer_relevancy",
        "context_precision": "context_precision",
        "context_recall": "context_recall",
    }

    eval_examples = [
        e for e in examples
        if not _is_system_not_in_context(e.pred_answer)
        and not e.is_malformed_answer
        and not e.is_generation_error
    ]
    skipped_nic = sum(1 for e in examples if _is_system_not_in_context(e.pred_answer))
    skipped_malformed = sum(1 for e in examples if e.is_malformed_answer)
    skipped_generation_error = sum(1 for e in examples if e.is_generation_error)

    logger.info(
        "RAGAS: evaluating %d examples; skipped %d canonical NIC, %d malformed answers, %d generation failures",
        len(eval_examples),
        skipped_nic,
        skipped_malformed,
        skipped_generation_error,
    )

    diagnostics = {
        "n_eval_examples": len(eval_examples),
        "n_skipped_canonical_nic": skipped_nic,
        "n_skipped_malformed_answer": skipped_malformed,
        "n_skipped_generation_failure": skipped_generation_error,
        "ragas_run_failed": False,
        "ragas_run_error": None,
        "batched_for_anthropic": judge_cfg.backend == "anthropic",
        "anthropic_batch_size": anthropic_batch_size if judge_cfg.backend == "anthropic" else None,
        "anthropic_batch_sleep_s": anthropic_batch_sleep_s if judge_cfg.backend == "anthropic" else None,
        "n_batches_attempted": 0,
        "n_batches_failed": 0,
    }

    if not eval_examples:
        per_rows = [{"id": e.id, **{req: None for req in metrics}} for e in examples]
        return {}, per_rows, diagnostics

    def _extract_rows_and_summary(
        res_obj: Any,
        batch_examples: List[Example],
    ) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, float]]:
        metric_rows_local: Dict[str, Dict[str, Any]] = {}
        summary_local: Dict[str, float] = {}

        try:
            df = res_obj.to_pandas()

            for i, e in enumerate(batch_examples):
                row: Dict[str, Any] = {"id": e.id}
                for req, col in col_map.items():
                    if req in metrics and col in df.columns:
                        v = df.iloc[i][col]
                        if v is None:
                            row[req] = None
                        else:
                            try:
                                fv = float(v)
                                row[req] = None if math.isnan(fv) else fv
                            except Exception:
                                row[req] = None
                metric_rows_local[e.id] = row

            for req, col in col_map.items():
                if req in metrics and col in df.columns:
                    vals: List[float] = []
                    for x in df[col].tolist():
                        try:
                            fx = float(x)
                            if not math.isnan(fx):
                                vals.append(fx)
                        except Exception:
                            pass
                    if vals:
                        summary_local[req] = float(sum(vals) / len(vals))

        except Exception:
            try:
                as_dict = dict(res_obj)
                for req, key in col_map.items():
                    if req in metrics and key in as_dict:
                        try:
                            f = float(as_dict[key])
                            if not math.isnan(f):
                                summary_local[req] = f
                        except Exception:
                            pass
            except Exception:
                pass

            for e in batch_examples:
                metric_rows_local[e.id] = {"id": e.id, **{req: None for req in metrics}}

        return metric_rows_local, summary_local

    metric_rows_by_id: Dict[str, Dict[str, Any]] = {}

    if judge_cfg.backend != "anthropic":
        ds = Dataset.from_dict({
            "question": [e.question for e in eval_examples],
            "answer": [e.pred_answer for e in eval_examples],
            "contexts": [e.contexts for e in eval_examples],
        })

        summary: Dict[str, float] = {}

        try:
            res = evaluate(ds, metrics=selected)
            metric_rows_by_id, summary = _extract_rows_and_summary(res, eval_examples)

        except Exception as e:
            diagnostics["ragas_run_failed"] = True
            diagnostics["ragas_run_error"] = repr(e)
            logger.exception("RAGAS evaluate(...) failed")

            for ex in eval_examples:
                metric_rows_by_id[ex.id] = {"id": ex.id, **{req: None for req in metrics}}

        per_rows = []
        for e in examples:
            if e.id in metric_rows_by_id:
                per_rows.append(metric_rows_by_id[e.id])
            else:
                per_rows.append({"id": e.id, **{req: None for req in metrics}})

        return summary, per_rows, diagnostics

    all_metric_values: Dict[str, List[float]] = {m: [] for m in metrics}
    batch_size = max(1, int(anthropic_batch_size))
    sleep_s = max(0.0, float(anthropic_batch_sleep_s))

    logger.info(
        "RAGAS Anthropic batching enabled: batch_size=%d sleep_s=%.2f",
        batch_size,
        sleep_s,
    )

    for start in range(0, len(eval_examples), batch_size):
        batch_examples = eval_examples[start:start + batch_size]
        diagnostics["n_batches_attempted"] += 1

        ds_batch = Dataset.from_dict({
            "question": [e.question for e in batch_examples],
            "answer": [e.pred_answer for e in batch_examples],
            "contexts": [e.contexts for e in batch_examples],
        })

        try:
            logger.info(
                "RAGAS Anthropic batch %d-%d / %d",
                start + 1,
                start + len(batch_examples),
                len(eval_examples),
            )
            res_batch = evaluate(ds_batch, metrics=selected)
            batch_rows, _ = _extract_rows_and_summary(res_batch, batch_examples)

            for ex in batch_examples:
                row = batch_rows.get(ex.id, {"id": ex.id, **{req: None for req in metrics}})
                metric_rows_by_id[ex.id] = row

                for m in metrics:
                    v = row.get(m)
                    if isinstance(v, (int, float)):
                        fv = float(v)
                        if not math.isnan(fv):
                            all_metric_values[m].append(fv)

        except Exception as e:
            diagnostics["n_batches_failed"] += 1
            logger.exception(
                "RAGAS Anthropic batch failed for examples %d-%d",
                start + 1,
                start + len(batch_examples),
            )
            for ex in batch_examples:
                metric_rows_by_id[ex.id] = {"id": ex.id, **{req: None for req in metrics}}

            if diagnostics["ragas_run_error"] is None:
                diagnostics["ragas_run_error"] = repr(e)

        if sleep_s > 0 and (start + batch_size) < len(eval_examples):
            time.sleep(sleep_s)

    if diagnostics["n_batches_failed"] > 0:
        diagnostics["ragas_run_failed"] = True

    summary = {
        m: float(sum(vals) / len(vals))
        for m, vals in all_metric_values.items()
        if vals
    }

    per_rows = []
    for e in examples:
        if e.id in metric_rows_by_id:
            per_rows.append(metric_rows_by_id[e.id])
        else:
            per_rows.append({"id": e.id, **{req: None for req in metrics}})

    return summary, per_rows, diagnostics


# Nugget judging

_ALLOWED_LABELS = {"support", "partial_support", "not_support"}


def _normalize_label(x: str) -> str:
    t = (x or "").strip().lower()
    if t in _ALLOWED_LABELS:
        return t
    if "no support" in t or t.startswith("no ") or t.startswith("not "):
        return "not_support"
    if "partial" in t:
        return "partial_support"
    if t.startswith("support"):
        return "support"
    return "not_support"


def judge_nuggets_batch(
    llm_cfg: LLMConfig,
    question: str,
    answer: str,
    nuggets: List[str],
) -> Tuple[List[str], bool]:
    """
    Returns:
      (labels, used_failure_fallback)
    """
    if not nuggets:
        return [], False

    nuggets_str = "\n".join(f"{i + 1}. {n}" for i, n in enumerate(nuggets))
    prompt = f"""
You are evaluating whether an answer contains specific factual information nuggets for a biomedical RAG evaluation.

Question:
{question}

Answer:
{answer}

Nuggets:
{nuggets_str}

For EACH nugget, assign exactly one label:

- support: The answer conveys the same core factual claim as the nugget. Exact wording is NOT required. Paraphrases, synonyms, abbreviations, and reasonable biomedical wording differences count as support. The answer does not need to repeat every minor detail or every example in the nugget if the clinically or technically important meaning is preserved.

- partial_support: The answer is related to the nugget and captures part of the claim, but it is clearly incomplete or misses an important detail. Use this when the answer mentions the right concept but omits a central threshold, condition, comparison, limitation, or qualifier.

- not_support: The answer does not mention the nugget's claim, only discusses an unrelated topic, contradicts the nugget, or only says that the information is not available.

Rules:
- Be generous with paraphrases. Do NOT require word-for-word overlap.
- Judge whether the answer would be acceptable to someone checking the main factual content of the nugget.
- Do NOT mark a nugget as not_support just because the answer is shorter or less detailed than the nugget.
- If a nugget contains a list of examples or conditions, support is allowed when the answer captures the main condition or the most important listed items. Use partial_support if the answer only gives a vague category or misses a central condition.
- If a number, threshold, or named condition is the main point of the nugget, it should be present for support. If it is only a secondary detail in a longer nugget, missing it can still be partial_support rather than not_support.
- If the entire answer is exactly NOT IN CONTEXT, label all nuggets not_support.
- If the answer contains partial abstention language such as "NOT IN CONTEXT for ..." but also contains factual content, do NOT automatically mark all nuggets as not_support. Judge the factual content normally.
- Do not give support merely because the answer says that something is not in context. A nugget is supported only when the answer actually states the nugget's factual content.
- Judge only whether the answer contains the nugget content. Do not judge whether the retrieved evidence supports the answer; retrieval support is handled separately by the evaluator.

Return ONLY valid JSON. No markdown, no explanation, no extra keys.

JSON format:
{{"labels": ["<label_1>", "<label_2>", "..."]}}

The "labels" array must contain exactly {len(nuggets)} items and follow the same order as the nuggets.
""".strip()

    out = generate(prompt, llm_cfg)
    obj = _try_parse_json_obj(out)
    if obj is None or "labels" not in obj or not isinstance(obj["labels"], list):
        snippet = (out or "").strip().replace("\n", " ")[:300]
        logger.warning("Nugget judge returned non-JSON or wrong schema. Snippet: %r", snippet)
        labels: List[str] = [
            ln.strip().lower()
            for ln in (out or "").splitlines()
            if ln.strip().lower() in _ALLOWED_LABELS
        ]
        if len(labels) == len(nuggets):
            return labels, False
        return ["not_support"] * len(nuggets), True

    raw_labels = obj["labels"]
    labels = [_normalize_label(str(x)) for x in raw_labels]
    if len(labels) != len(nuggets):
        return ["not_support"] * len(nuggets), True
    return labels, False


# Slot-level diagnostic taxonomy

TAXONOMY_CASES = (
    "retrieved_and_supported",
    "retrieved_partially_supported",
    "retrieved_but_not_used",
    "retrieved_but_abstained",
    "not_retrieved_and_not_supported",
    "not_retrieved_but_answered",
)


def _compute_slot_retrieval_support(
    context_ids: List[str],
    slot_evidence: Dict[str, List[str]],
) -> Dict[str, bool]:
    retrieved_set = set(context_ids)
    return {
        slot: bool(set(chunks) & retrieved_set)
        for slot, chunks in slot_evidence.items()
    }


def _classify_slot(
    retrieved: bool,
    strict_recall: float,
    soft_recall: float,
    is_global_nic: bool,
) -> str:
    """
    Slot-level diagnostic taxonomy.

    This deliberately avoids the word 'hallucination' because a slot can be
    answered without its annotated gold evidence being retrieved for multiple
    reasons: alternative retrieved evidence, model prior knowledge, judge error,
    or true hallucination.

    Definitions:
    - retrieved_and_supported:
        The slot's annotated gold evidence was retrieved and at least one nugget
        in the slot was fully supported.
    - retrieved_partially_supported:
        The slot's annotated gold evidence was retrieved and at least one nugget
        was partially supported, but no nugget was fully supported.
    - retrieved_but_not_used:
        The slot's annotated gold evidence was retrieved, but the answer did not
        cover the slot's required content.
    - retrieved_but_abstained:
        The slot's annotated gold evidence was retrieved, but the model gave a
        full canonical NOT IN CONTEXT answer.
    - not_retrieved_and_not_supported:
        The slot's annotated gold evidence was not retrieved and the answer did
        not cover the slot's required content.
    - not_retrieved_but_answered:
        The slot's annotated gold evidence was not retrieved, but the answer did
        cover at least part of the slot's required content. This is a potential
        unsupported-content signal, not definite hallucination.
    """
    if retrieved:
        if strict_recall > 0.0:
            return "retrieved_and_supported"
        if soft_recall > 0.0:
            return "retrieved_partially_supported"
        if is_global_nic:
            return "retrieved_but_abstained"
        return "retrieved_but_not_used"

    if strict_recall > 0.0 or soft_recall > 0.0:
        return "not_retrieved_but_answered"

    return "not_retrieved_and_not_supported"


# Question-level main taxonomy

QUESTION_TAXONOMY_CASES = (
    "all_nuggets_supported_answer",
    "all_main_parts_supported_answer",
    "all_main_parts_partially_supported_answer",
    "incomplete_answer",
    "over_abstention",
    "correct_abstention",
    "answered_but_no_required_content",
    "potentially_unsupported_content",
    "retrieval_limited_answer",
    "system_failure",
    "no_slot_scores",
)


def _classify_question_from_nugget_row(row: Dict[str, Any]) -> str:
    """
    Question-level taxonomy over one Task B example.

    This is the main taxonomy to report for whole-answer behavior.

    Design principle:
    The taxonomy separates:
    1. answer completeness,
    2. retrieval sufficiency,
    3. abstention behavior,
    4. potential unsupported-content risk,
    5. system failures.

    Definitions:
    - all_nuggets_supported_answer:
        All required slots were retrieved and every nugget in every slot was
        fully supported.
    - all_main_parts_supported_answer:
        All required slots were retrieved and every required slot had at least
        one fully supported nugget, but not necessarily every nugget.
    - all_main_parts_partially_supported_answer:
        All required slots were retrieved and every required slot had at least
        partial support, but at least one slot lacked full support.
    - incomplete_answer:
        All required slots were retrieved and the answer covered some, but not
        all, required slots.
    - over_abstention:
        All required slots were retrieved, but the model gave a full canonical
        NOT IN CONTEXT answer.
    - correct_abstention:
        Not all required slots were retrieved and the model gave a full
        canonical NOT IN CONTEXT answer.
    - answered_but_no_required_content:
        All required slots were retrieved, the model answered, but no required
        slot content was covered even partially.
    - potentially_unsupported_content:
        The model answered content for at least one required slot whose
        annotated gold evidence was not retrieved. This is a risk signal, not a
        definitive hallucination label.
    - retrieval_limited_answer:
        The model answered while the full question evidence was incomplete, and
        it did not clearly answer content for the non-retrieved slot(s).
    """
    if row.get("generation_failure") or row.get("malformed_answer"):
        return "system_failure"

    slots = row.get("slots") or {}
    if not slots:
        return "no_slot_scores"

    is_nic = bool(row.get("system_not_in_context", False))
    all_slots_retrieved = bool(row.get("all_slots_retrieved", False))

    strict_vals: List[float] = []
    soft_vals: List[float] = []
    nonretrieved_has_support = False

    for slot_info in slots.values():
        retrieved = bool(slot_info.get("retrieved", False))
        strict = float(slot_info.get("strict_recall", 0.0) or 0.0)
        soft = float(slot_info.get("soft_recall", 0.0) or 0.0)

        strict_vals.append(strict)
        soft_vals.append(soft)

        if not retrieved and soft > 0.0:
            nonretrieved_has_support = True

    any_soft = any(v > 0.0 for v in soft_vals)
    all_soft = all(v > 0.0 for v in soft_vals)
    all_strict = all(v > 0.0 for v in strict_vals)
    all_nuggets_strict = all(v >= 1.0 for v in strict_vals)

    if is_nic:
        if all_slots_retrieved:
            return "over_abstention"
        return "correct_abstention"

    if all_slots_retrieved:
        if all_nuggets_strict:
            return "all_nuggets_supported_answer"
        if all_strict:
            return "all_main_parts_supported_answer"
        if all_soft:
            return "all_main_parts_partially_supported_answer"
        if any_soft:
            return "incomplete_answer"
        return "answered_but_no_required_content"

    if nonretrieved_has_support:
        return "potentially_unsupported_content"

    return "retrieval_limited_answer"


def _add_question_taxonomy_to_row(row: Dict[str, Any]) -> Dict[str, Any]:
    case = _classify_question_from_nugget_row(row)
    row["question_taxonomy_case"] = case
    row["question_taxonomy"] = {
        qcase: (1 if qcase == case else 0)
        for qcase in QUESTION_TAXONOMY_CASES
    }
    return row


def _make_zero_nugget_labels(slot: str, nuggets_by_slot: Dict[str, List[str]]) -> List[Dict[str, Any]]:
    return [
        {
            "idx": i,
            "nugget": nug,
            "label": "not_support",
        }
        for i, nug in enumerate(nuggets_by_slot.get(slot, []))
    ]


def score_nuggets_taskB(
    examples: List[Example],
    judge_llm_cfg: LLMConfig,
    cache_path: Path,
    batch_mode: str = "per_slot",
    max_batch_nuggets: int = 20,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Compute per-example nugget recall, slot-level taxonomy, and evidence sufficiency proxy.

    Hard generation failures and malformed answers are treated as system failures:
    zero recall is assigned without any nugget-judge calls.
    """
    cache: Dict[str, Any] = {}
    if cache_path.exists():
        try:
            cache = json.loads(cache_path.read_text(encoding="utf-8"))
        except Exception:
            cache = {}
    if not isinstance(cache, dict):
        cache = {}

    def _ckey(e: Example, slot: str, idx: int, nug: str) -> str:
        import hashlib
        payload = json.dumps(
            {
                "prompt_version": NUGGET_PROMPT_VERSION,
                "id": e.id,
                "slot": slot,
                "idx": idx,
                "question": e.question,
                "answer": e.pred_answer,
                "nugget": nug,
                "judge_model": judge_llm_cfg.model,
                "judge_backend": judge_llm_cfg.backend,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        return hashlib.sha1(payload.encode("utf-8")).hexdigest()

    per_rows: List[Dict[str, Any]] = []
    n_calls = 0
    n_cached = 0
    n_malformed_zeroed = 0
    n_generation_failure_zeroed = 0
    n_judge_failure_fallback_batches = 0

    def _append_finalized(row: Dict[str, Any]) -> None:
        per_rows.append(_add_question_taxonomy_to_row(row))

    for e in examples:
        is_global_nic = _is_system_not_in_context(e.pred_answer)
        nic_text_present = is_global_nic or _detect_nic_text_present(e.pred_answer)
        partial_nic_text_present = _detect_partial_nic_text_present(e.pred_answer)

        slot_retrieval_support = (
            _compute_slot_retrieval_support(e.context_ids, e.slot_evidence)
            if e.slot_evidence else {}
        )
        all_slots_retrieved = (
            bool(slot_retrieval_support)
            and all(slot_retrieval_support.get(slot, False) for slot in e.slot_nuggets.keys())
        ) if e.slot_nuggets else False

        row: Dict[str, Any] = {
            "id": e.id,
            "slots": {},
            "macro_avg_strict_recall": None,
            "macro_avg_soft_recall": None,
            "system_not_in_context": is_global_nic,
            "nic_text_present": nic_text_present,
            "partial_nic_text_present": partial_nic_text_present,
            "slot_retrieval_support": slot_retrieval_support,
            "all_slots_retrieved": all_slots_retrieved,
            "taxonomy": {case: 0 for case in TAXONOMY_CASES},
            "malformed_answer": e.is_malformed_answer,
            "generation_failure": e.is_generation_error,
        }

        if not e.slot_nuggets:
            _append_finalized(row)
            continue

        if e.is_generation_error:
            n_generation_failure_zeroed += 1
            for slot in e.slot_nuggets.keys():
                retrieved = slot_retrieval_support.get(slot, False)
                case = _classify_slot(retrieved, 0.0, 0.0, is_global_nic=False)
                row["slots"][slot] = {
                    "strict_recall": 0.0,
                    "soft_recall": 0.0,
                    "retrieved": retrieved,
                    "taxonomy_case": case,
                    "nugget_labels": _make_zero_nugget_labels(slot, e.slot_nuggets),
                }
                row["taxonomy"][case] += 1
            row["macro_avg_strict_recall"] = 0.0
            row["macro_avg_soft_recall"] = 0.0
            _append_finalized(row)
            continue

        if e.is_malformed_answer:
            n_malformed_zeroed += 1
            for slot in e.slot_nuggets.keys():
                retrieved = slot_retrieval_support.get(slot, False)
                case = _classify_slot(retrieved, 0.0, 0.0, is_global_nic=False)
                row["slots"][slot] = {
                    "strict_recall": 0.0,
                    "soft_recall": 0.0,
                    "retrieved": retrieved,
                    "taxonomy_case": case,
                    "nugget_labels": _make_zero_nugget_labels(slot, e.slot_nuggets),
                }
                row["taxonomy"][case] += 1
            row["macro_avg_strict_recall"] = 0.0
            row["macro_avg_soft_recall"] = 0.0
            _append_finalized(row)
            continue

        if is_global_nic:
            for slot in e.slot_nuggets.keys():
                retrieved = slot_retrieval_support.get(slot, False)
                case = _classify_slot(retrieved, 0.0, 0.0, is_global_nic=True)
                row["slots"][slot] = {
                    "strict_recall": 0.0,
                    "soft_recall": 0.0,
                    "retrieved": retrieved,
                    "taxonomy_case": case,
                    "nugget_labels": _make_zero_nugget_labels(slot, e.slot_nuggets),
                }
                row["taxonomy"][case] += 1
            row["macro_avg_strict_recall"] = 0.0
            row["macro_avg_soft_recall"] = 0.0
            _append_finalized(row)
            continue

        slot_scores: Dict[str, Dict[str, Any]] = {}
        strict_vals: List[float] = []
        soft_vals: List[float] = []

        if batch_mode not in {"per_slot", "all"}:
            raise ValueError("--nugget_batch_mode must be 'per_slot' or 'all'")

        if batch_mode == "all":
            flat: List[Tuple[str, int, str]] = [
                (slot, i, nug)
                for slot, nuggets in e.slot_nuggets.items()
                for i, nug in enumerate(nuggets)
            ]

            labels_out: Dict[Tuple[str, int], str] = {}

            for start in range(0, len(flat), max_batch_nuggets):
                chunk = flat[start:start + max_batch_nuggets]
                nuggets_to_judge: List[str] = []
                mapping: List[Tuple[str, int, str]] = []

                for (slot, idx, nug) in chunk:
                    k = _ckey(e, slot, idx, nug)
                    if k in cache:
                        labels_out[(slot, idx)] = _normalize_label(str(cache[k]))
                        n_cached += 1
                    else:
                        nuggets_to_judge.append(nug)
                        mapping.append((slot, idx, nug))

                if nuggets_to_judge:
                    n_calls += 1
                    batch_labels, used_failure_fallback = judge_nuggets_batch(
                        judge_llm_cfg, e.question, e.pred_answer, nuggets_to_judge
                    )
                    if used_failure_fallback:
                        n_judge_failure_fallback_batches += 1
                    for (slot, idx, nug), lab in zip(mapping, batch_labels):
                        lab_n = _normalize_label(lab)
                        labels_out[(slot, idx)] = lab_n
                        cache[_ckey(e, slot, idx, nug)] = lab_n

            for slot, nuggets in e.slot_nuggets.items():
                if not nuggets:
                    continue

                full = sum(
                    1 for i in range(len(nuggets))
                    if labels_out.get((slot, i), "not_support") == "support"
                )
                partial = sum(
                    1 for i in range(len(nuggets))
                    if labels_out.get((slot, i), "not_support") == "partial_support"
                )
                n = len(nuggets)
                strict = full / n
                soft = (full + 0.5 * partial) / n
                retrieved = slot_retrieval_support.get(slot, False)
                case = _classify_slot(retrieved, strict, soft, is_global_nic=False)

                slot_scores[slot] = {
                    "strict_recall": strict,
                    "soft_recall": soft,
                    "retrieved": retrieved,
                    "taxonomy_case": case,
                    "nugget_labels": [
                        {
                            "idx": i,
                            "nugget": nuggets[i],
                            "label": labels_out.get((slot, i), "not_support"),
                        }
                        for i in range(len(nuggets))
                    ],
                }
                row["taxonomy"][case] += 1
                strict_vals.append(strict)
                soft_vals.append(soft)

        else:
            for slot, nuggets in e.slot_nuggets.items():
                if not nuggets:
                    continue

                cached_labels: Dict[int, str] = {}
                to_judge: List[Tuple[int, str]] = []

                for i, nug in enumerate(nuggets):
                    k = _ckey(e, slot, i, nug)
                    if k in cache:
                        cached_labels[i] = _normalize_label(str(cache[k]))
                        n_cached += 1
                    else:
                        to_judge.append((i, nug))

                labels_final: Dict[int, str] = dict(cached_labels)

                for start in range(0, len(to_judge), max_batch_nuggets):
                    batch = to_judge[start:start + max_batch_nuggets]
                    if not batch:
                        continue
                    n_calls += 1
                    batch_nugs = [nug for (_, nug) in batch]
                    batch_labels, used_failure_fallback = judge_nuggets_batch(
                        judge_llm_cfg, e.question, e.pred_answer, batch_nugs
                    )
                    if used_failure_fallback:
                        n_judge_failure_fallback_batches += 1
                    for (idx, nug), lab in zip(batch, batch_labels):
                        lab_n = _normalize_label(lab)
                        labels_final[idx] = lab_n
                        cache[_ckey(e, slot, idx, nug)] = lab_n

                full = sum(
                    1 for i in range(len(nuggets))
                    if labels_final.get(i, "not_support") == "support"
                )
                partial = sum(
                    1 for i in range(len(nuggets))
                    if labels_final.get(i, "not_support") == "partial_support"
                )
                n = len(nuggets)
                strict = full / n
                soft = (full + 0.5 * partial) / n
                retrieved = slot_retrieval_support.get(slot, False)
                case = _classify_slot(retrieved, strict, soft, is_global_nic=False)

                slot_scores[slot] = {
                    "strict_recall": strict,
                    "soft_recall": soft,
                    "retrieved": retrieved,
                    "taxonomy_case": case,
                    "nugget_labels": [
                        {
                            "idx": i,
                            "nugget": nuggets[i],
                            "label": labels_final.get(i, "not_support"),
                        }
                        for i in range(len(nuggets))
                    ],
                }
                row["taxonomy"][case] += 1
                strict_vals.append(strict)
                soft_vals.append(soft)

        row["slots"] = slot_scores
        row["macro_avg_strict_recall"] = (
            sum(strict_vals) / len(strict_vals) if strict_vals else None
        )
        row["macro_avg_soft_recall"] = (
            sum(soft_vals) / len(soft_vals) if soft_vals else None
        )
        _append_finalized(row)

    _safe_write_json(cache_path, cache)

    macro_strict_all = [
        r["macro_avg_strict_recall"] for r in per_rows
        if isinstance(r.get("macro_avg_strict_recall"), (int, float))
    ]
    macro_soft_all = [
        r["macro_avg_soft_recall"] for r in per_rows
        if isinstance(r.get("macro_avg_soft_recall"), (int, float))
    ]

    slot_taxonomy_totals: Dict[str, int] = {case: 0 for case in TAXONOMY_CASES}
    for r in per_rows:
        for case in TAXONOMY_CASES:
            slot_taxonomy_totals[case] += r.get("taxonomy", {}).get(case, 0)

    total_slots = sum(slot_taxonomy_totals.values())
    slot_taxonomy_rates = {
        case: (slot_taxonomy_totals[case] / total_slots) if total_slots > 0 else None
        for case in TAXONOMY_CASES
    }

    question_taxonomy_totals: Dict[str, int] = {case: 0 for case in QUESTION_TAXONOMY_CASES}
    for r in per_rows:
        case = str(r.get("question_taxonomy_case") or "no_slot_scores")
        if case not in question_taxonomy_totals:
            question_taxonomy_totals[case] = 0
        question_taxonomy_totals[case] += 1

    total_questions = len(per_rows)
    question_taxonomy_rates = {
        case: (count / total_questions) if total_questions > 0 else None
        for case, count in question_taxonomy_totals.items()
    }

    n_all_nuggets_supported = sum(
        1 for r in per_rows
        if r.get("question_taxonomy_case") == "all_nuggets_supported_answer"
    )

    n_all_main_parts_supported = sum(
        1 for r in per_rows
        if r.get("question_taxonomy_case") in {
            "all_nuggets_supported_answer",
            "all_main_parts_supported_answer",
        }
    )

    n_all_main_parts_at_least_partial = sum(
        1 for r in per_rows
        if r.get("question_taxonomy_case") in {
            "all_nuggets_supported_answer",
            "all_main_parts_supported_answer",
            "all_main_parts_partially_supported_answer",
        }
    )

    n_fully_sufficient = sum(1 for r in per_rows if r.get("all_slots_retrieved", False))
    evidence_sufficiency_rate = (n_fully_sufficient / len(per_rows)) if per_rows else None

    nugget_summary = {
        "n_examples": len(per_rows),
        "judge_model": judge_llm_cfg.model,
        "judge_backend": judge_llm_cfg.backend,
        "prompt_version": NUGGET_PROMPT_VERSION,
        "batch_mode": batch_mode,
        "max_batch_nuggets": max_batch_nuggets,
        "llm_calls": n_calls,
        "cached_labels_used": n_cached,
        "malformed_answers_zeroed": n_malformed_zeroed,
        "generation_failures_zeroed": n_generation_failure_zeroed,
        "judge_failure_fallback_batches": n_judge_failure_fallback_batches,
        "macro_avg_strict_recall_mean": (
            sum(macro_strict_all) / len(macro_strict_all) if macro_strict_all else None
        ),
        "macro_avg_soft_recall_mean": (
            sum(macro_soft_all) / len(macro_soft_all) if macro_soft_all else None
        ),
        "n_all_nuggets_supported": n_all_nuggets_supported,
        "all_nuggets_supported_rate": (
            n_all_nuggets_supported / len(per_rows) if per_rows else None
        ),
        "n_all_main_parts_supported": n_all_main_parts_supported,
        "all_main_parts_supported_rate": (
            n_all_main_parts_supported / len(per_rows) if per_rows else None
        ),
        "n_all_main_parts_at_least_partial": n_all_main_parts_at_least_partial,
        "all_main_parts_at_least_partial_rate": (
            n_all_main_parts_at_least_partial / len(per_rows) if per_rows else None
        ),
        # Legacy aliases kept for compatibility: taxonomy_* refers to slot-level taxonomy.
        "taxonomy_totals": slot_taxonomy_totals,
        "taxonomy_rates": slot_taxonomy_rates,
        "slot_taxonomy_totals": slot_taxonomy_totals,
        "slot_taxonomy_rates": slot_taxonomy_rates,
        "question_taxonomy_totals": question_taxonomy_totals,
        "question_taxonomy_rates": question_taxonomy_rates,
        "evidence_sufficiency_rate": evidence_sufficiency_rate,
        "n_fully_sufficient": n_fully_sufficient,
    }
    return nugget_summary, per_rows


# Aggregation helpers

def _mean(values: List[Optional[float]]) -> Optional[float]:
    xs: List[float] = []
    for v in values:
        if isinstance(v, (int, float)):
            fv = float(v)
            if not math.isnan(fv):
                xs.append(fv)
    if not xs:
        return None
    return float(sum(xs) / len(xs))


def _rate(n: int, d: int) -> Optional[float]:
    return (n / d) if d > 0 else None


# Main

def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Single-judge evaluation script for RAG generations. "
            "Each execution uses one configured judge backend/model, while "
            "RAGAS embeddings are configured separately."
        )
    )
    ap.add_argument("--task", choices=["A", "B"], required=True)
    ap.add_argument("--generations", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out_dir", required=True)

    ap.add_argument("--run_ragas", action="store_true")
    ap.add_argument(
        "--ragas_metrics",
        default="faithfulness,answer_relevance,context_precision,context_recall",
    )
    ap.add_argument("--run_nuggets", action="store_true")
    ap.add_argument(
        "--nugget_batch_mode",
        choices=["per_slot", "all"],
        default="per_slot",
    )
    ap.add_argument("--nugget_max_batch_nuggets", type=int, default=20)
    ap.add_argument(
        "--anthropic_ragas_batch_size",
        type=int,
        default=5,
        help="Batch size for RAGAS only when using Anthropic as judge.",
    )
    ap.add_argument(
        "--anthropic_ragas_batch_sleep_s",
        type=float,
        default=1.0,
        help="Sleep between Anthropic RAGAS batches to reduce 429 rate limits.",
    )

    ap.add_argument(
        "--backend",
        choices=["openai", "openai_compat", "anthropic"],
        default="openai",
    )
    ap.add_argument("--lm_model", required=True)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max_tokens", type=int, default=512)
    ap.add_argument("--timeout_s", type=int, default=600)
    ap.add_argument("--openai_api_key", default=None)
    ap.add_argument("--openai_base_url", default=None)
    ap.add_argument("--anthropic_api_key", default=None)

    ap.add_argument(
        "--ragas_embedding_backend",
        choices=["openai", "openai_compat"],
        default="openai",
    )
    ap.add_argument(
        "--ragas_embedding_model",
        default="text-embedding-3-large",
    )
    ap.add_argument("--ragas_embedding_api_key", default=None)
    ap.add_argument("--ragas_embedding_base_url", default=None)

    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        force=True,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    examples = _build_examples(args.generations, args.dataset, args.task)

    judge_api_key = (
        args.anthropic_api_key or os.getenv("ANTHROPIC_API_KEY")
        if args.backend == "anthropic"
        else args.openai_api_key or os.getenv("OPENAI_API_KEY")
    )
    judge_base_url = args.openai_base_url if args.backend == "openai_compat" else None

    judge_cfg = LLMConfig(
        backend=args.backend,
        model=args.lm_model,
        temperature=float(args.temperature),
        max_tokens=int(args.max_tokens),
        timeout_s=int(args.timeout_s),
        api_key=judge_api_key,
        base_url=judge_base_url,
    )

    n_malformed_examples = sum(1 for e in examples if e.is_malformed_answer)
    n_generation_failures = sum(1 for e in examples if e.is_generation_error)

    meta = {
        "task": args.task,
        "generations": args.generations,
        "dataset": args.dataset,
        "judge_llm": {
            "backend": judge_cfg.backend,
            "model": judge_cfg.model,
            "temperature": judge_cfg.temperature,
            "max_tokens": judge_cfg.max_tokens,
            "timeout_s": judge_cfg.timeout_s,
            "base_url": judge_cfg.base_url,
        },
        "ragas_embedding": {
            "backend": args.ragas_embedding_backend,
            "model": args.ragas_embedding_model,
            "base_url": args.ragas_embedding_base_url,
        },
        "run_ragas": bool(args.run_ragas),
        "ragas_metrics": args.ragas_metrics,
        "run_nuggets": bool(args.run_nuggets),
        "nugget_prompt_version": NUGGET_PROMPT_VERSION,
        "nugget_batch_mode": args.nugget_batch_mode,
        "nugget_max_batch_nuggets": args.nugget_max_batch_nuggets,
        "anthropic_ragas_batching": {
            "batch_size": args.anthropic_ragas_batch_size,
            "sleep_s": args.anthropic_ragas_batch_sleep_s,
        },
        "n_malformed_answers_detected": n_malformed_examples,
        "n_generation_failures_detected": n_generation_failures,
        "system_failure_policy": {
            "description": (
                "Unusable generations are treated as system failures. "
                "This includes malformed parsed answers and hard generation/API/runtime failures."
            ),
            "conditional_metrics": "excluded",
            "end_to_end_metrics": "counted_as_zero",
            "nugget_judging": "skipped_and_zeroed",
        },
        "taxonomy_policy": {
            "description": (
                "The evaluator writes both slot-level and question-level taxonomy. "
                "Slot-level taxonomy is diagnostic and describes whether annotated evidence for each required slot was retrieved "
                "and whether the answer covered that slot. Question-level taxonomy classifies whole-answer behavior and is the "
                "main taxonomy for reporting. The category all_nuggets_supported_answer means every nugget in every required "
                "slot was fully supported. The category all_main_parts_supported_answer means every required slot had at least "
                "one fully supported nugget, but not necessarily every nugget. The category potentially_unsupported_content is "
                "a conservative risk signal: the model answered content for a slot whose annotated gold evidence was not retrieved; "
                "it is not treated as definitive hallucination."
            ),
            "slot_partial_support_case": "retrieved_partially_supported",
            "main_reporting_taxonomy": "question_taxonomy",
            "strictest_question_success_case": "all_nuggets_supported_answer",
            "main_parts_success_cases": [
                "all_nuggets_supported_answer",
                "all_main_parts_supported_answer",
            ],
            "partial_main_parts_success_cases": [
                "all_nuggets_supported_answer",
                "all_main_parts_supported_answer",
                "all_main_parts_partially_supported_answer",
            ],
            "unsupported_content_risk_case": "potentially_unsupported_content",
        },
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    _safe_write_json(out_dir / "eval_meta.json", meta)

    per_example: Dict[str, Dict[str, Any]] = {}
    for e in examples:
        is_system_nic = _is_system_not_in_context(e.pred_answer)

        if e.is_generation_error:
            final_status_eval = FINAL_STATUS_ERROR_GENERATION_FAILURE
        elif e.is_malformed_answer:
            final_status_eval = FINAL_STATUS_ERROR_MALFORMED_ANSWER
        else:
            final_status_eval = e.final_status

        per_example[e.id] = {
            "id": e.id,
            "question": e.question,
            "pred_answer": e.pred_answer,
            "answer_raw": e.answer_raw,
            "explicit_nic": e.explicit_nic,
            "forced_not_in_context": e.forced_not_in_context,
            "forced_nic_empty_retrieval": e.forced_nic_empty_retrieval,
            "forced_nic_no_valid_citations": e.forced_nic_no_valid_citations,
            "final_status": final_status_eval,
            "original_final_status": e.final_status,
            "is_grounded": e.is_grounded if not (e.is_malformed_answer or e.is_generation_error) else False,
            "is_system_nic": is_system_nic,
            "nic_text_present": (
                _detect_nic_text_present(e.pred_answer)
                if not (e.is_malformed_answer or e.is_generation_error)
                else False
            ),
            "partial_nic_text_present": (
                _detect_partial_nic_text_present(e.pred_answer)
                if not (e.is_malformed_answer or e.is_generation_error)
                else False
            ),
            "is_malformed_answer": e.is_malformed_answer,
            "is_generation_error": e.is_generation_error,
            "error_message": e.error_message,
        }

    ragas_summary: Dict[str, Any] = {}
    if args.run_ragas:
        metric_list = [m.strip() for m in args.ragas_metrics.split(",") if m.strip()]
        logger.info(
            "Running RAGAS metrics=%s judge=%s/%s embedding=%s/%s",
            metric_list,
            judge_cfg.backend,
            judge_cfg.model,
            args.ragas_embedding_backend,
            args.ragas_embedding_model,
        )
        rsum, per_rows, rdiag = run_ragas(
            examples=examples,
            judge_cfg=judge_cfg,
            metrics=metric_list,
            ragas_embedding_model=args.ragas_embedding_model,
            ragas_embedding_backend=args.ragas_embedding_backend,
            ragas_embedding_api_key=args.ragas_embedding_api_key or os.getenv("OPENAI_API_KEY"),
            ragas_embedding_base_url=args.ragas_embedding_base_url,
            anthropic_batch_size=args.anthropic_ragas_batch_size,
            anthropic_batch_sleep_s=args.anthropic_ragas_batch_sleep_s,
        )
        ragas_summary = {
            "metrics_mean_system_from_ragas": rsum,
            "diagnostics": rdiag,
        }
        for r in per_rows:
            rid = str(r.get("id", "")).strip()
            if not rid or rid not in per_example:
                continue
            per_example[rid].setdefault("ragas", {})
            for k, v in r.items():
                if k != "id":
                    per_example[rid]["ragas"][k] = v
        _safe_write_json(out_dir / "ragas_summary.json", ragas_summary)

    nugget_summary: Dict[str, Any] = {}
    if args.run_nuggets:
        if args.task.upper() != "B":
            raise SystemExit("--run_nuggets is only valid for --task B")

        nugget_judge_cfg = LLMConfig(
            backend=judge_cfg.backend,
            model=judge_cfg.model,
            temperature=0.0,
            max_tokens=400,
            timeout_s=judge_cfg.timeout_s,
            api_key=judge_cfg.api_key,
            base_url=judge_cfg.base_url,
        )

        cache_path = out_dir / (
            f"nugget_cache_{_safe_slug(nugget_judge_cfg.backend)}"
            f"_{_safe_slug(nugget_judge_cfg.model)}"
            f"_{_safe_slug(NUGGET_PROMPT_VERSION)}.json"
        )

        logger.info(
            "Running nugget recall prompt_version=%s batch_mode=%s max=%d judge=%s/%s cache=%s",
            NUGGET_PROMPT_VERSION,
            args.nugget_batch_mode,
            args.nugget_max_batch_nuggets,
            nugget_judge_cfg.backend,
            nugget_judge_cfg.model,
            cache_path.as_posix(),
        )

        nsum, per_rows = score_nuggets_taskB(
            examples,
            nugget_judge_cfg,
            cache_path=cache_path,
            batch_mode=args.nugget_batch_mode,
            max_batch_nuggets=args.nugget_max_batch_nuggets,
        )
        nugget_summary = nsum
        _safe_write_json(out_dir / "nugget_summary.json", nugget_summary)

        for r in per_rows:
            rid = str(r.get("id", "")).strip()
            if not rid or rid not in per_example:
                continue
            per_example[rid]["nuggets"] = {
                "slots": r.get("slots", {}),
                "macro_avg_strict_recall": r.get("macro_avg_strict_recall"),
                "macro_avg_soft_recall": r.get("macro_avg_soft_recall"),
                "taxonomy": r.get("taxonomy", {}),
                "slot_retrieval_support": r.get("slot_retrieval_support", {}),
                "all_slots_retrieved": r.get("all_slots_retrieved", False),
                "malformed_answer": r.get("malformed_answer", False),
                "generation_failure": r.get("generation_failure", False),
                "question_taxonomy_case": r.get("question_taxonomy_case"),
                "question_taxonomy": r.get("question_taxonomy", {}),
                "slot_taxonomy": r.get("taxonomy", {}),
                "partial_nic_text_present": r.get("partial_nic_text_present", False),
            }

    per_path = out_dir / "per_example_eval.jsonl"
    with per_path.open("w", encoding="utf-8") as f:
        for rid in sorted(per_example.keys()):
            f.write(json.dumps(per_example[rid], ensure_ascii=False) + "\n")

    n_total = len(examples)
    n_explicit = sum(1 for e in examples if e.explicit_nic)
    n_forced_empty = sum(1 for e in examples if e.forced_nic_empty_retrieval)
    n_forced_no_cit = sum(1 for e in examples if e.forced_nic_no_valid_citations)
    n_forced = n_forced_empty + n_forced_no_cit
    n_system_nic = sum(1 for e in examples if _is_system_not_in_context(e.pred_answer))
    n_grounded = sum(
        1 for e in examples
        if e.is_grounded and not e.is_malformed_answer and not e.is_generation_error
    )
    n_malformed = sum(1 for e in examples if e.is_malformed_answer)
    n_gen_error = sum(1 for e in examples if e.is_generation_error)
    n_system_failures = n_malformed + n_gen_error
    n_partial_nic_text = sum(
        1 for e in examples
        if not e.is_malformed_answer
        and not e.is_generation_error
        and _detect_partial_nic_text_present(e.pred_answer)
    )

    global_summary: Dict[str, Any] = {
        "n_examples": n_total,
        "nic_rates": {
            "explicit_nic_rate": _rate(n_explicit, n_total),
            "forced_nic_empty_retrieval_rate": _rate(n_forced_empty, n_total),
            "forced_nic_no_valid_citations_rate": _rate(n_forced_no_cit, n_total),
            "forced_not_in_context_rate": _rate(n_forced, n_total),
            "system_not_in_context_rate": _rate(n_system_nic, n_total),
            "partial_nic_text_rate": _rate(n_partial_nic_text, n_total),
            "grounded_rate": _rate(n_grounded, n_total),
        },
        "nic_counts": {
            "n_total": n_total,
            "n_explicit_nic": n_explicit,
            "n_forced_nic_empty_retrieval": n_forced_empty,
            "n_forced_nic_no_valid_citations": n_forced_no_cit,
            "n_forced_not_in_context": n_forced,
            "n_system_not_in_context": n_system_nic,
            "n_partial_nic_text": n_partial_nic_text,
            "n_grounded": n_grounded,
        },
        "system_failure_counts": {
            "n_generation_failure": n_gen_error,
            "n_malformed_answer": n_malformed,
            "n_system_failures_total": n_system_failures,
            "generation_failure_rate": _rate(n_gen_error, n_total),
            "malformed_answer_rate": _rate(n_malformed, n_total),
            "system_failure_rate_total": _rate(n_system_failures, n_total),
        },
        "outcome_counts": {
            "answered": sum(
                1 for e in examples
                if e.final_status == "answered"
                and not e.is_malformed_answer
                and not e.is_generation_error
            ),
            "explicit_nic": sum(1 for e in examples if e.final_status == "explicit_nic"),
            "forced_nic_empty_retrieval": sum(
                1 for e in examples if e.final_status == "forced_nic_empty_retrieval"
            ),
            "forced_nic_no_valid_citations": sum(
                1 for e in examples if e.final_status == "forced_nic_no_valid_citations"
            ),
            "error_generation_failure": n_gen_error,
            "error_malformed_answer": n_malformed,
        },
        "nic_variants": {
            "filter_nic": {
                "description": (
                    "Metrics over usable non-NIC outputs only. "
                    "Excludes canonical NIC and all system failures. "
                    "Measures conditional answer quality when the system produced an answer."
                ),
                "ragas_metrics_mean": {},
                "nugget_macro_mean": {"strict": None, "soft": None},
            },
            "zero_nic": {
                "description": (
                    "All queries; canonical NIC and all system failures count as 0. "
                    "Measures end-to-end utility per query."
                ),
                "ragas_metrics_mean": {},
                "nugget_macro_mean": {"strict": None, "soft": None},
            },
        },
        # Legacy aliases kept for compatibility: taxonomy_* refers to slot-level taxonomy.
        "taxonomy": nugget_summary.get("slot_taxonomy_totals", {}),
        "taxonomy_rates": nugget_summary.get("slot_taxonomy_rates", {}),
        "slot_taxonomy": nugget_summary.get("slot_taxonomy_totals", {}),
        "slot_taxonomy_rates": nugget_summary.get("slot_taxonomy_rates", {}),
        "question_taxonomy": nugget_summary.get("question_taxonomy_totals", {}),
        "question_taxonomy_rates": nugget_summary.get("question_taxonomy_rates", {}),
        "evidence_sufficiency_rate": nugget_summary.get("evidence_sufficiency_rate"),
        "n_fully_sufficient": nugget_summary.get("n_fully_sufficient"),
        "nugget_prompt_version": nugget_summary.get("prompt_version"),
        "n_all_nuggets_supported": nugget_summary.get("n_all_nuggets_supported"),
        "all_nuggets_supported_rate": nugget_summary.get("all_nuggets_supported_rate"),
        "n_all_main_parts_supported": nugget_summary.get("n_all_main_parts_supported"),
        "all_main_parts_supported_rate": nugget_summary.get("all_main_parts_supported_rate"),
        "n_all_main_parts_at_least_partial": nugget_summary.get("n_all_main_parts_at_least_partial"),
        "all_main_parts_at_least_partial_rate": nugget_summary.get("all_main_parts_at_least_partial_rate"),
    }

    nic_variant_cfg = {
        "filter_nic": lambda rec: (
            not rec.get("is_system_nic", False)
            and not rec.get("is_malformed_answer", False)
            and not rec.get("is_generation_error", False)
        ),
        "zero_nic": lambda rec: True,
    }

    if args.run_ragas:
        metric_list = [m.strip() for m in args.ragas_metrics.split(",") if m.strip()]
        for variant_name, include_fn in nic_variant_cfg.items():
            for m in metric_list:
                vals: List[float] = []
                for _, rec in per_example.items():
                    is_nic = rec.get("is_system_nic", False)
                    is_malformed = rec.get("is_malformed_answer", False)
                    is_generation_error = rec.get("is_generation_error", False)

                    if variant_name == "zero_nic" and (is_nic or is_malformed or is_generation_error):
                        vals.append(0.0)
                        continue

                    if not include_fn(rec):
                        continue

                    v = rec.get("ragas", {}).get(m)
                    try:
                        v_float = float(v)
                    except (TypeError, ValueError):
                        continue
                    if math.isnan(v_float):
                        continue
                    vals.append(v_float)

                global_summary["nic_variants"][variant_name]["ragas_metrics_mean"][m] = _mean(vals)

    if args.run_nuggets and args.task.upper() == "B":
        for variant_name, include_fn in nic_variant_cfg.items():
            strict_vals: List[float] = []
            soft_vals: List[float] = []
            for _, rec in per_example.items():
                is_system_nic = rec.get("is_system_nic", False)
                is_malformed = rec.get("is_malformed_answer", False)
                is_generation_error = rec.get("is_generation_error", False)
                n = rec.get("nuggets", {})
                s = n.get("macro_avg_strict_recall")
                t = n.get("macro_avg_soft_recall")

                if variant_name == "zero_nic" and (is_system_nic or is_malformed or is_generation_error):
                    strict_vals.append(0.0)
                    soft_vals.append(0.0)
                    continue

                if include_fn(rec):
                    if isinstance(s, (int, float)) and not math.isnan(float(s)):
                        strict_vals.append(float(s))
                    if isinstance(t, (int, float)) and not math.isnan(float(t)):
                        soft_vals.append(float(t))

            global_summary["nic_variants"][variant_name]["nugget_macro_mean"] = {
                "strict": _mean(strict_vals),
                "soft": _mean(soft_vals),
            }

    _safe_write_json(out_dir / "global_summary.json", global_summary)
    logger.info(
        "Done. Wrote: %s, %s",
        per_path.as_posix(),
        (out_dir / "global_summary.json").as_posix(),
    )


if __name__ == "__main__":
    main()