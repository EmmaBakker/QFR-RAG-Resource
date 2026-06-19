#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from src.retrieval.utils_io import iter_jsonl
from .llm_client import LLMConfig, generate
from .schema import NIC_TOKEN, GenerationRecord

logger = logging.getLogger(__name__)

FINAL_STATUS_ERROR_MALFORMED_ANSWER = "error_malformed_answer"
FINAL_STATUS_ERROR_GENERATION_FAILURE = "error_generation_failure"

TASKA_JUDGE_PROMPT_VERSION = "taskA_judge_v3_conservative_taxonomy"

TAXONOMY_CASES = (
    "retrieved_and_correct",
    "retrieved_but_incorrect",
    "over_abstention",
    "correct_abstention",
    "answered_without_retrieved_gold_evidence",
    "system_failure",
    "judge_failure",
)

_EXPECTED_CONTEXT_ID_PATTERN = re.compile(r"^doc[a-zA-Z0-9]+_chunk\d+$")
_FENCED_JSON_RX = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.IGNORECASE | re.DOTALL)
_NIC_TEXT_RX = re.compile(rf"\b{re.escape(NIC_TOKEN)}\b", re.IGNORECASE)

JUDGE_STATUS_OK = "ok"
JUDGE_STATUS_PARSE_ERROR = "parse_error"
JUDGE_STATUS_API_ERROR = "api_error"
JUDGE_STATUS_SKIPPED_SYSTEM_FAILURE = "skipped_system_failure"
JUDGE_STATUS_SKIPPED_CANONICAL_NIC = "skipped_canonical_nic"


# IO HELPERS

def _read_jsonl_by_id(path: str) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for row in iter_jsonl(path):
        rid = str(row.get("id", "")).strip()
        if rid and rid not in out:
            out[rid] = row
    return out


def _safe_write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def _safe_slug(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", str(s)).strip("_") or "unknown"


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


# TEXT / OUTPUT ROBUSTNESS

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
    """
    Detect model outputs that are likely leaked JSON fragments rather than
    usable parsed answers.

    This includes both generation-style JSON fragments and judge-style JSON
    fragments for robustness.
    """
    tl = (t or "").lower().strip()
    if not tl:
        return False

    json_markers = [
        # Generation-style fragments
        '{"answer"',
        '{"citations"',
        '"answer":',
        '"citations":',
        "{'answer'",
        "{'citations'",

        # Judge-style fragments, kept for robustness
        '{"reasoning"',
        '{"verdict"',
        '"reasoning":',
        '"verdict":',
        "{'reasoning'",
        "{'verdict'",
    ]
    return any(marker in tl for marker in json_markers)


def _is_only_punctuation_or_braces(t: str) -> bool:
    stripped = re.sub(r"[\s`'\"{}\[\]():,._-]+", "", t or "")
    return stripped == ""


def _is_malformed_answer(pred_answer: str) -> bool:
    """
    Conservative malformed-answer detection:
    unusable outputs are treated as system failures, not valid answers.
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


# DATA MODEL

@dataclass
class Example:
    id: str
    question: str
    gold_answer: str
    gold_evidence: List[Dict[str, str]]

    pred_answer: str
    answer_raw: str
    contexts: List[str]
    context_ids: List[str]

    explicit_nic: bool
    forced_not_in_context: bool
    forced_nic_empty_retrieval: bool
    forced_nic_no_valid_citations: bool
    final_status: str
    is_grounded: bool

    is_malformed_answer: bool
    is_generation_error: bool
    error_message: Optional[str]


# EXAMPLE ASSEMBLY

def _build_examples(
    generations_path: str,
    dataset_path: str,
    task: str = "A",
) -> List[Example]:
    ds_by_id = _read_jsonl_by_id(dataset_path)

    examples: List[Example] = []
    n_missing_ds = 0
    n_missing_gens = 0
    n_empty_gold = 0
    seen_gen_ids: Set[str] = set()

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

        if str(d.get("task", "")).upper() != task.upper():
            continue

        question = str(
            g.question or raw_row.get("question", "") or d.get("question", "")
        ).strip()
        gold_answer = str(d.get("answer", "") or "").strip()
        gold_evidence = d.get("gold_evidence") or []

        if not question or not gold_answer:
            n_empty_gold += 1
            continue

        if not isinstance(gold_evidence, list):
            gold_evidence = []

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
                gold_answer=gold_answer,
                gold_evidence=gold_evidence,

                pred_answer=pred_answer,
                answer_raw=answer_raw,
                contexts=contexts,
                context_ids=context_ids,

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

    for rid, d in ds_by_id.items():
        if str(d.get("task", "")).upper() != task.upper():
            continue
        if rid not in seen_gen_ids:
            n_missing_gens += 1

    logger.info(
        "Loaded examples=%d (missing_dataset_for_generation=%d, missing_generation_for_dataset=%d, skipped_empty_gold=%d)",
        len(examples),
        n_missing_ds,
        n_missing_gens,
        n_empty_gold,
    )
    return examples


# RETRIEVAL / EVIDENCE

def _validate_context_ids(
    examples: List[Example],
    allow_unvalidated_context_ids: bool,
) -> Dict[str, Any]:
    n_examples_with_context_ids = 0
    n_examples_with_bad_context_ids = 0
    bad_examples: List[str] = []

    for e in examples:
        if not e.context_ids:
            continue

        n_examples_with_context_ids += 1

        bad_ids = [
            str(cid) for cid in e.context_ids
            if not _EXPECTED_CONTEXT_ID_PATTERN.match(str(cid))
        ]

        if bad_ids:
            n_examples_with_bad_context_ids += 1

            if len(bad_examples) < 10:
                bad_examples.append(e.id)

            if not allow_unvalidated_context_ids:
                raise ValueError(
                    f"id={e.id}: context_ids do not match expected pattern "
                    f"{_EXPECTED_CONTEXT_ID_PATTERN.pattern}. "
                    f"Example bad ids: {bad_ids[:3]}"
                )

    return {
        "expected_pattern": _EXPECTED_CONTEXT_ID_PATTERN.pattern,
        "allow_unvalidated_context_ids": allow_unvalidated_context_ids,
        "n_examples_with_context_ids": n_examples_with_context_ids,
        "n_examples_with_bad_context_ids": n_examples_with_bad_context_ids,
        "bad_example_ids_sample": bad_examples,
    }


def _gold_evidence_candidate_ids(ev: Dict[str, str]) -> Set[str]:
    """
    Build possible full chunk IDs from a gold_evidence row.

    Expected common format:
      {"doc_id": "docabc", "chunk_id": "chunk000001"}

    But this also supports rows where chunk_id already contains a full
    doc..._chunk... identifier.
    """
    out: Set[str] = set()

    doc_id = str(ev.get("doc_id", "")).strip()
    chunk_id = str(ev.get("chunk_id", "")).strip()

    if chunk_id:
        out.add(chunk_id)

    if doc_id and chunk_id:
        if chunk_id.startswith(doc_id + "_"):
            out.add(chunk_id)
        else:
            out.add(f"{doc_id}_{chunk_id}")

    return {x for x in out if x}


def _has_retrieved_gold_evidence(
    context_ids: List[str],
    gold_evidence: List[Dict[str, str]],
) -> Tuple[bool, List[str]]:
    if not context_ids or not gold_evidence:
        return False, []

    ctx_set: Set[str] = {str(cid).strip() for cid in context_ids if str(cid).strip()}
    matched: List[str] = []

    for ev in gold_evidence:
        if not isinstance(ev, dict):
            continue

        for candidate in _gold_evidence_candidate_ids(ev):
            if candidate in ctx_set:
                matched.append(candidate)

    # Preserve deterministic order and remove duplicates.
    matched_unique = sorted(set(matched))
    return bool(matched_unique), matched_unique


# JUDGE

_ALLOWED_VERDICTS = {"YES", "NO"}


def _judge_cache_key(
    question: str,
    gold_answer: str,
    pred_answer: str,
    judge_cfg: LLMConfig,
    prompt_version: str,
) -> str:
    payload = json.dumps(
        {
            "question": question,
            "gold_answer": gold_answer,
            "pred_answer": pred_answer,
            "judge_backend": judge_cfg.backend,
            "judge_model": judge_cfg.model,
            "temperature": judge_cfg.temperature,
            "prompt_version": prompt_version,
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def _judge_prompt(question: str, gold_answer: str, pred_answer: str) -> str:
    return f"""
You are evaluating answer correctness for a retrieval-augmented QA system.

Question:
{question}

Reference answer:
{gold_answer}

Candidate answer:
{pred_answer if pred_answer.strip() else "[NO ANSWER PROVIDED]"}

Task:
Decide whether the candidate answer is semantically correct relative to the reference answer.

Instructions:
- Focus on semantic equivalence to the reference answer.
- Accept paraphrases and minor wording differences.
- Reject factually incorrect, contradictory, or materially incomplete answers.
- Do NOT judge style, tone, formatting, or verbosity.
- Do NOT infer missing facts unless they are clearly entailed.
- If the answer abstains or says the information is not in context while the reference contains an answer, this is incorrect.

Return ONLY valid JSON, with no markdown and no extra text.

JSON schema:
{{
  "reasoning": "1-3 concise sentences",
  "verdict": "YES" or "NO"
}}
""".strip()


def _parse_judge_response(raw: str) -> Tuple[Optional[bool], str, str]:
    """
    Returns:
      (judge_correct_or_none, judge_status, reasoning)
    """
    obj = _try_parse_json_obj(raw)
    if obj is not None:
        verdict = str(obj.get("verdict", "")).strip().upper()
        reasoning = str(obj.get("reasoning", "")).strip()
        if verdict in _ALLOWED_VERDICTS:
            return verdict == "YES", JUDGE_STATUS_OK, reasoning

    text = (raw or "").strip()

    verdict = None
    for line in reversed(text.splitlines()):
        u = line.strip().upper()
        if u in {"YES", "VERDICT: YES"}:
            verdict = True
            break
        if u in {"NO", "VERDICT: NO"}:
            verdict = False
            break

    if verdict is not None:
        reasoning_lines = []
        for line in text.splitlines():
            if line.strip().upper() in {"YES", "NO", "VERDICT: YES", "VERDICT: NO"}:
                break
            reasoning_lines.append(line)
        return verdict, JUDGE_STATUS_OK, " ".join(reasoning_lines).strip()

    snippet = text.replace("\n", " ")[:300]
    return None, JUDGE_STATUS_PARSE_ERROR, f"Could not parse judge output: {snippet}"


def _call_judge_once(
    judge_cfg: LLMConfig,
    question: str,
    gold_answer: str,
    pred_answer: str,
) -> str:
    prompt = _judge_prompt(
        question=question,
        gold_answer=gold_answer,
        pred_answer=pred_answer,
    )
    return generate(prompt, judge_cfg)


def judge_correctness_taskA(
    examples: List[Example],
    judge_llm_cfg: LLMConfig,
    cache_path: Path,
    max_retries: int = 3,
    retry_sleep_s: float = 3.0,
    prompt_version: str = TASKA_JUDGE_PROMPT_VERSION,
) -> Tuple[Dict[str, Any], Dict[str, Dict[str, Any]]]:
    """
    Judge answer correctness for Task A against gold reference answers.

    Returns:
      (judge_summary, per_example_judge_by_id)
    """
    cache: Dict[str, Any] = {}
    if cache_path.exists():
        try:
            cache = json.loads(cache_path.read_text(encoding="utf-8"))
        except Exception:
            cache = {}

    if not isinstance(cache, dict):
        cache = {}

    per_example: Dict[str, Dict[str, Any]] = {}

    n_calls = 0
    n_cached = 0
    n_retries = 0

    status_counts: Dict[str, int] = {
        JUDGE_STATUS_OK: 0,
        JUDGE_STATUS_PARSE_ERROR: 0,
        JUDGE_STATUS_API_ERROR: 0,
        JUDGE_STATUS_SKIPPED_SYSTEM_FAILURE: 0,
        JUDGE_STATUS_SKIPPED_CANONICAL_NIC: 0,
    }

    correct_flags_conditional: List[float] = []
    correct_flags_end_to_end: List[float] = []

    for e in examples:
        is_system_nic = _is_system_not_in_context(e.pred_answer)

        if e.is_generation_error or e.is_malformed_answer:
            status = JUDGE_STATUS_SKIPPED_SYSTEM_FAILURE
            judge_correct = None
            reasoning = "Skipped judge call due to system failure."
            raw = ""
            status_counts[status] += 1
            correct_flags_end_to_end.append(0.0)

        elif is_system_nic:
            status = JUDGE_STATUS_SKIPPED_CANONICAL_NIC
            judge_correct = None
            reasoning = "Skipped judge call for canonical NOT IN CONTEXT answer."
            raw = ""
            status_counts[status] += 1
            correct_flags_end_to_end.append(0.0)

        else:
            ckey = _judge_cache_key(
                question=e.question,
                gold_answer=e.gold_answer,
                pred_answer=e.pred_answer,
                judge_cfg=judge_llm_cfg,
                prompt_version=prompt_version,
            )

            if ckey in cache:
                cached = cache[ckey]
                judge_correct = cached.get("judge_correct")
                status = str(cached.get("judge_status", JUDGE_STATUS_PARSE_ERROR))
                reasoning = str(cached.get("judge_reasoning", "")).strip()
                raw = str(cached.get("judge_raw", "")).strip()
                n_cached += 1

            else:
                last_raw = ""
                last_reasoning = ""
                judge_correct = None
                status = JUDGE_STATUS_API_ERROR

                for attempt in range(1, max_retries + 1):
                    try:
                        if attempt > 1:
                            n_retries += 1

                        out = _call_judge_once(
                            judge_cfg=judge_llm_cfg,
                            question=e.question,
                            gold_answer=e.gold_answer,
                            pred_answer=e.pred_answer,
                        )
                        last_raw = out
                        judge_correct, parsed_status, last_reasoning = _parse_judge_response(out)
                        status = parsed_status

                        if parsed_status == JUDGE_STATUS_OK:
                            break

                    except Exception as ex:
                        last_raw = ""
                        last_reasoning = f"Judge API error: {repr(ex)}"
                        status = JUDGE_STATUS_API_ERROR

                    if attempt < max_retries:
                        time.sleep(retry_sleep_s)

                raw = last_raw
                reasoning = last_reasoning
                n_calls += 1

                cache[ckey] = {
                    "judge_correct": judge_correct,
                    "judge_status": status,
                    "judge_reasoning": reasoning,
                    "judge_raw": raw,
                }

            status_counts[status] = status_counts.get(status, 0) + 1

            if status == JUDGE_STATUS_OK and isinstance(judge_correct, bool):
                correct_flags_conditional.append(1.0 if judge_correct else 0.0)
                correct_flags_end_to_end.append(1.0 if judge_correct else 0.0)
            else:
                correct_flags_end_to_end.append(0.0)

        per_example[e.id] = {
            "judge_correct": judge_correct,
            "judge_status": status,
            "judge_reasoning": reasoning,
            "judge_raw": raw,
        }

    _safe_write_json(cache_path, cache)

    judge_summary = {
        "judge_model": judge_llm_cfg.model,
        "judge_backend": judge_llm_cfg.backend,
        "prompt_version": prompt_version,
        "llm_calls": n_calls,
        "cached_labels_used": n_cached,
        "n_retries": n_retries,
        "judge_status_counts": status_counts,
        "accuracy_conditional_judged_only": _mean(correct_flags_conditional),
        "accuracy_end_to_end_all_queries": _mean(correct_flags_end_to_end),
    }

    return judge_summary, per_example


# TAXONOMY

def _classify_taskA_case(
    retrieved: bool,
    judge_correct: Optional[bool],
    judge_status: str,
    is_system_nic: bool,
    nic_text_present: bool,
    is_generation_error: bool,
    is_malformed_answer: bool,
) -> str:
    """
    Task A retrieval-aware taxonomy.

    Definitions:
    - retrieved_and_correct:
        Annotated gold evidence was retrieved and the answer was judged correct.
    - retrieved_but_incorrect:
        Annotated gold evidence was retrieved, the model answered, but the answer
        was judged incorrect.
    - over_abstention:
        Annotated gold evidence was retrieved, but the model gave a canonical
        NOT IN CONTEXT answer.
    - correct_abstention:
        Annotated gold evidence was not retrieved and the model gave a canonical
        NOT IN CONTEXT answer.
    - answered_without_retrieved_gold_evidence:
        The model answered even though annotated gold evidence was not retrieved.
        This is an evidence-support risk signal, not a definitive hallucination label.
    - system_failure:
        Generation failed or the parsed answer was malformed.
    - judge_failure:
        The model answered, but the judge did not return a valid correctness label.

    Note:
    Only canonical NOT IN CONTEXT counts as full abstention. Partial NOT IN
    CONTEXT text is recorded separately as a diagnostic signal.
    """
    if is_generation_error or is_malformed_answer:
        return "system_failure"

    abstained = is_system_nic

    if retrieved:
        if abstained:
            return "over_abstention"

        if judge_status != JUDGE_STATUS_OK:
            return "judge_failure"

        if judge_correct is True:
            return "retrieved_and_correct"

        return "retrieved_but_incorrect"

    # Gold evidence not retrieved
    if abstained:
        return "correct_abstention"

    if judge_status != JUDGE_STATUS_OK:
        return "judge_failure"

    return "answered_without_retrieved_gold_evidence"


def build_taskA_taxonomy(
    examples: List[Example],
    judge_by_id: Dict[str, Dict[str, Any]],
) -> Tuple[Dict[str, Any], Dict[str, Dict[str, Any]]]:
    per_taxonomy: Dict[str, Dict[str, Any]] = {}
    totals: Dict[str, int] = {case: 0 for case in TAXONOMY_CASES}

    n_retrieved_gold = 0
    n_not_retrieved_gold = 0

    for e in examples:
        judge_rec = judge_by_id.get(e.id, {})
        judge_correct = judge_rec.get("judge_correct")
        judge_status = str(judge_rec.get("judge_status", JUDGE_STATUS_PARSE_ERROR))

        is_system_nic = _is_system_not_in_context(e.pred_answer)
        nic_text_present = (
            False if (e.is_generation_error or e.is_malformed_answer)
            else _detect_nic_text_present(e.pred_answer)
        )

        retrieved, matched_gold_context_ids = _has_retrieved_gold_evidence(
            e.context_ids,
            e.gold_evidence,
        )

        if retrieved:
            n_retrieved_gold += 1
        else:
            n_not_retrieved_gold += 1

        case = _classify_taskA_case(
            retrieved=retrieved,
            judge_correct=judge_correct if isinstance(judge_correct, bool) else None,
            judge_status=judge_status,
            is_system_nic=is_system_nic,
            nic_text_present=nic_text_present,
            is_generation_error=e.is_generation_error,
            is_malformed_answer=e.is_malformed_answer,
        )
        totals[case] += 1

        per_taxonomy[e.id] = {
            "retrieved_gold_evidence": retrieved,
            "matched_gold_context_ids": matched_gold_context_ids,
            "taxonomy_case": case,
        }

    n_total = len(examples)
    rates = {k: _rate(v, n_total) for k, v in totals.items()}

    summary = {
        "n_examples": n_total,
        "n_retrieved_gold_evidence": n_retrieved_gold,
        "n_not_retrieved_gold_evidence": n_not_retrieved_gold,
        "taxonomy_totals": totals,
        "taxonomy_rates": rates,
    }
    return summary, per_taxonomy


# MAIN

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Task A evaluation: reference-based correctness + retrieval-aware taxonomy."
    )
    ap.add_argument("--task", choices=["A"], default="A")
    ap.add_argument("--generations", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out_dir", required=True)

    ap.add_argument(
        "--backend",
        choices=["openai", "openai_compat", "anthropic"],
        default="openai",
    )
    ap.add_argument("--lm_model", required=True)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max_tokens", type=int, default=300)
    ap.add_argument("--timeout_s", type=int, default=600)
    ap.add_argument("--openai_api_key", default=None)
    ap.add_argument("--openai_base_url", default=None)
    ap.add_argument("--anthropic_api_key", default=None)

    ap.add_argument("--judge_max_retries", type=int, default=3)
    ap.add_argument("--judge_retry_sleep_s", type=float, default=3.0)
    ap.add_argument("--allow_unvalidated_context_ids", action="store_true")

    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        force=True,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    examples = _build_examples(
        generations_path=args.generations,
        dataset_path=args.dataset,
        task="A",
    )

    context_validation = _validate_context_ids(
        examples=examples,
        allow_unvalidated_context_ids=args.allow_unvalidated_context_ids,
    )

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
        "task": "A",
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
        "judge_prompt_version": TASKA_JUDGE_PROMPT_VERSION,
        "context_id_validation": context_validation,
        "judge_retry_policy": {
            "max_retries": args.judge_max_retries,
            "retry_sleep_s": args.judge_retry_sleep_s,
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
            "judge_calls": "skipped_for_canonical_nic_and_system_failures",
        },
        "taxonomy_policy": {
            "description": (
                "Task A uses a reference-answer correctness judge and a retrieval-aware taxonomy. "
                "The taxonomy separates correctness, retrieval sufficiency, abstention behavior, "
                "answered-without-annotated-evidence risk, system failures, and judge failures. "
                "Only canonical NOT IN CONTEXT is treated as full abstention; partial NOT IN CONTEXT "
                "language is recorded diagnostically."
            ),
            "cases": list(TAXONOMY_CASES),
            "unsupported_content_risk_case": "answered_without_retrieved_gold_evidence",
        },
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    _safe_write_json(out_dir / "eval_meta.json", meta)

    cache_path = out_dir / (
        f"taskA_judge_cache_{_safe_slug(judge_cfg.backend)}"
        f"_{_safe_slug(judge_cfg.model)}"
        f"_{_safe_slug(TASKA_JUDGE_PROMPT_VERSION)}.json"
    )

    logger.info(
        "Running Task A judge with %s/%s cache=%s",
        judge_cfg.backend,
        judge_cfg.model,
        cache_path.as_posix(),
    )

    judge_summary, judge_by_id = judge_correctness_taskA(
        examples=examples,
        judge_llm_cfg=judge_cfg,
        cache_path=cache_path,
        max_retries=args.judge_max_retries,
        retry_sleep_s=args.judge_retry_sleep_s,
        prompt_version=TASKA_JUDGE_PROMPT_VERSION,
    )
    _safe_write_json(out_dir / "judge_summary.json", judge_summary)

    taxonomy_summary, taxonomy_by_id = build_taskA_taxonomy(
        examples=examples,
        judge_by_id=judge_by_id,
    )
    _safe_write_json(out_dir / "taxonomy_summary.json", taxonomy_summary)

    per_example: Dict[str, Dict[str, Any]] = {}

    for e in examples:
        is_system_nic = _is_system_not_in_context(e.pred_answer)

        if e.is_generation_error:
            final_status_eval = FINAL_STATUS_ERROR_GENERATION_FAILURE
        elif e.is_malformed_answer:
            final_status_eval = FINAL_STATUS_ERROR_MALFORMED_ANSWER
        else:
            final_status_eval = e.final_status

        judge_rec = judge_by_id.get(e.id, {})
        tax_rec = taxonomy_by_id.get(e.id, {})

        per_example[e.id] = {
            "id": e.id,
            "question": e.question,
            "gold_answer": e.gold_answer,
            "gold_evidence": e.gold_evidence,
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

            "taskA_judge": {
                "judge_correct": judge_rec.get("judge_correct"),
                "judge_status": judge_rec.get("judge_status"),
                "judge_reasoning": judge_rec.get("judge_reasoning"),
            },
            "taskA_taxonomy": {
                "retrieved_gold_evidence": tax_rec.get("retrieved_gold_evidence"),
                "matched_gold_context_ids": tax_rec.get("matched_gold_context_ids", []),
                "taxonomy_case": tax_rec.get("taxonomy_case"),
            },
        }

    per_path = out_dir / "per_example_eval.jsonl"
    with per_path.open("w", encoding="utf-8") as f:
        for rid in sorted(per_example.keys()):
            f.write(json.dumps(per_example[rid], ensure_ascii=False) + "\n")

    n_total = len(examples)

    n_explicit = sum(1 for e in examples if e.explicit_nic)
    n_forced_empty = sum(1 for e in examples if e.forced_nic_empty_retrieval)
    n_forced_no_cit = sum(1 for e in examples if e.forced_nic_no_valid_citations)
    n_forced_flag = sum(1 for e in examples if e.forced_not_in_context)
    n_system_nic = sum(1 for e in examples if _is_system_not_in_context(e.pred_answer))

    n_partial_nic_text = sum(
        1 for e in examples
        if not e.is_malformed_answer
        and not e.is_generation_error
        and _detect_partial_nic_text_present(e.pred_answer)
    )

    n_grounded = sum(
        1 for e in examples
        if e.is_grounded and not e.is_malformed_answer and not e.is_generation_error
    )

    n_malformed = sum(1 for e in examples if e.is_malformed_answer)
    n_gen_error = sum(1 for e in examples if e.is_generation_error)
    n_system_failures = n_malformed + n_gen_error

    n_judge_ok = sum(
        1 for rec in judge_by_id.values()
        if rec.get("judge_status") == JUDGE_STATUS_OK
    )
    n_judge_correct = sum(
        1 for rec in judge_by_id.values()
        if rec.get("judge_status") == JUDGE_STATUS_OK and rec.get("judge_correct") is True
    )

    global_summary: Dict[str, Any] = {
        "n_examples": n_total,
        "nic_rates": {
            "explicit_nic_rate": _rate(n_explicit, n_total),
            "forced_nic_empty_retrieval_rate": _rate(n_forced_empty, n_total),
            "forced_nic_no_valid_citations_rate": _rate(n_forced_no_cit, n_total),
            "forced_not_in_context_flag_rate": _rate(n_forced_flag, n_total),
            "system_not_in_context_rate": _rate(n_system_nic, n_total),
            "partial_nic_text_rate": _rate(n_partial_nic_text, n_total),
            "grounded_rate": _rate(n_grounded, n_total),
        },
        "nic_counts": {
            "n_total": n_total,
            "n_explicit_nic": n_explicit,
            "n_forced_nic_empty_retrieval": n_forced_empty,
            "n_forced_nic_no_valid_citations": n_forced_no_cit,
            "n_forced_not_in_context_flag": n_forced_flag,
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
        "judge": {
            "n_judge_ok": n_judge_ok,
            "n_judge_correct": n_judge_correct,
            "accuracy_conditional_judged_only": judge_summary.get("accuracy_conditional_judged_only"),
            "accuracy_end_to_end_all_queries": judge_summary.get("accuracy_end_to_end_all_queries"),
            "judge_status_counts": judge_summary.get("judge_status_counts", {}),
            "prompt_version": judge_summary.get("prompt_version"),
            "llm_calls": judge_summary.get("llm_calls"),
            "cached_labels_used": judge_summary.get("cached_labels_used"),
            "n_retries": judge_summary.get("n_retries"),
        },
        "nic_variants": {
            "filter_nic": {
                "description": (
                    "Metrics over usable non-NIC outputs only. "
                    "Excludes canonical NIC and all system failures."
                ),
                "taskA_correctness_mean": None,
            },
            "zero_nic": {
                "description": (
                    "All queries; canonical NIC and all system failures count as 0. "
                    "Measures end-to-end utility per query."
                ),
                "taskA_correctness_mean": None,
            },
        },
        "taxonomy": taxonomy_summary.get("taxonomy_totals", {}),
        "taxonomy_rates": taxonomy_summary.get("taxonomy_rates", {}),
        "retrieval": {
            "n_retrieved_gold_evidence": taxonomy_summary.get("n_retrieved_gold_evidence"),
            "n_not_retrieved_gold_evidence": taxonomy_summary.get("n_not_retrieved_gold_evidence"),
            "retrieved_gold_evidence_rate": _rate(
                taxonomy_summary.get("n_retrieved_gold_evidence", 0),
                n_total,
            ),
        },
    }

    nic_variant_cfg = {
        "filter_nic": lambda rec: (
            not rec.get("is_system_nic", False)
            and not rec.get("is_malformed_answer", False)
            and not rec.get("is_generation_error", False)
            and rec.get("taskA_judge", {}).get("judge_status") == JUDGE_STATUS_OK
        ),
        "zero_nic": lambda rec: True,
    }

    for variant_name, include_fn in nic_variant_cfg.items():
        vals: List[float] = []

        for _, rec in per_example.items():
            is_system_nic = rec.get("is_system_nic", False)
            is_malformed = rec.get("is_malformed_answer", False)
            is_generation_error = rec.get("is_generation_error", False)
            judge_correct = rec.get("taskA_judge", {}).get("judge_correct")
            judge_status = rec.get("taskA_judge", {}).get("judge_status")

            if variant_name == "zero_nic" and (is_system_nic or is_malformed or is_generation_error):
                vals.append(0.0)
                continue

            if not include_fn(rec):
                continue

            if judge_status == JUDGE_STATUS_OK and isinstance(judge_correct, bool):
                vals.append(1.0 if judge_correct else 0.0)

        global_summary["nic_variants"][variant_name]["taskA_correctness_mean"] = _mean(vals)

    _safe_write_json(out_dir / "global_summary.json", global_summary)

    logger.info(
        "Done. Wrote: %s, %s, %s",
        per_path.as_posix(),
        (out_dir / "judge_summary.json").as_posix(),
        (out_dir / "global_summary.json").as_posix(),
    )


if __name__ == "__main__":
    main()