#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import math
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple


try:
    from src.retrieval.utils_io import iter_jsonl
except Exception as e:
    raise ImportError(
        "Could not import iter_jsonl. Run from repo root with: export PYTHONPATH=$PWD"
    ) from e

try:
    from .llm_client import LLMConfig, generate
    from .schema import NIC_TOKEN, GenerationRecord
except Exception:
    from src.rag_pipeline.llm_client import LLMConfig, generate
    from src.rag_pipeline.schema import NIC_TOKEN, GenerationRecord


logger = logging.getLogger(__name__)


# ============================================================
# CONSTANTS
# ============================================================

FINAL_STATUS_ERROR_MALFORMED_ANSWER = "error_malformed_answer"
FINAL_STATUS_ERROR_GENERATION_FAILURE = "error_generation_failure"

CORRECTNESS_PROMPT_VERSION = "bioasq_llm_correctness_v3"

JUDGE_STATUS_OK = "ok"
JUDGE_STATUS_PARSE_ERROR = "parse_error"
JUDGE_STATUS_API_ERROR = "api_error"
JUDGE_STATUS_SKIPPED_SYSTEM_FAILURE = "skipped_system_failure"
JUDGE_STATUS_SKIPPED_CANONICAL_NIC = "skipped_canonical_nic"

_ALLOWED_VERDICTS = {"YES", "NO"}

_FENCED_JSON_RX = re.compile(
    r"```(?:json)?\s*(.*?)\s*```",
    re.IGNORECASE | re.DOTALL,
)
_NIC_TEXT_RX = re.compile(
    rf"\b{re.escape(NIC_TOKEN)}\b",
    re.IGNORECASE,
)

BIOASQ_KNOWN_TYPES = {"yesno", "factoid", "list", "summary"}


# ============================================================
# GENERIC IO / NUMERIC HELPERS
# ============================================================

def _safe_write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(obj, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        path.write_text("", encoding="utf-8")
        return

    cols: List[str] = []
    for row in rows:
        for k in row.keys():
            if k not in cols:
                cols.append(k)

    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def _safe_slug(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", str(s)).strip("_") or "unknown"


def _is_number(x: Any) -> bool:
    if not isinstance(x, (int, float)):
        return False
    try:
        return not math.isnan(float(x))
    except Exception:
        return False


def _mean(values: List[Optional[float]]) -> Optional[float]:
    xs: List[float] = []
    for v in values:
        if _is_number(v):
            xs.append(float(v))
    if not xs:
        return None
    return float(sum(xs) / len(xs))


def _rate(n: int, d: int) -> Optional[float]:
    return (n / d) if d > 0 else None


def _safe_float(x: Any) -> Optional[float]:
    try:
        v = float(x)
        if math.isnan(v):
            return None
        return v
    except Exception:
        return None


def _get_nested(d: Dict[str, Any], path: List[str], default: Any = None) -> Any:
    cur: Any = d
    for p in path:
        if not isinstance(cur, dict) or p not in cur:
            return default
        cur = cur[p]
    return cur


# ============================================================
# JSON / TEXT ROBUSTNESS
# ============================================================

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


def _detect_nic_text_present(pred_answer: str) -> bool:
    return bool(_NIC_TEXT_RX.search(pred_answer or ""))


def _looks_like_json_fragment(t: str) -> bool:
    tl = (t or "").lower().strip()
    if not tl:
        return False

    markers = [
        '{"answer"',
        '{"citations"',
        '"answer":',
        '"citations":',
        "{'answer'",
        "{'citations'",
        '{"reasoning"',
        '{"verdict"',
        '"reasoning":',
        '"verdict":',
    ]
    return any(m in tl for m in markers)


def _is_only_punctuation_or_braces(t: str) -> bool:
    stripped = re.sub(r"[\s`'\"{}\[\]():,._-]+", "", t or "")
    return stripped == ""


def _is_malformed_answer(pred_answer: str) -> bool:
    """
    Detect unusable parsed answers.

    This intentionally treats broken JSON/code-fence leakage as system failure,
    not as an ordinary incorrect answer.
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


# ============================================================
# BIOASQ TEXT NORMALIZATION / EXACT-STYLE HELPERS
# ============================================================

def _normalize_text(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"^[\W_]+|[\W_]+$", "", s)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _dedupe_preserve_order(items: List[str]) -> List[str]:
    out: List[str] = []
    seen: Set[str] = set()
    for x in items:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def _normalize_yesno_text(s: str) -> Optional[str]:
    t = _normalize_text(s)
    if t == "yes":
        return "yes"
    if t == "no":
        return "no"
    return None


def _extract_yesno_prediction(answer: str) -> Optional[str]:
    t = (answer or "").strip().lower()
    if not t:
        return None

    patterns = [
        r"^(yes|no)\b",
        r"^the answer is\s+(yes|no)\b",
        r"^it is\s+(yes|no)\b",
        r"^answer[:\s]+(yes|no)\b",
    ]
    for pat in patterns:
        m = re.search(pat, t)
        if m:
            return m.group(1)
    return None


def _gold_exact_groups(qtype: str, exact_answer: Any) -> List[Set[str]]:
    """
    BioASQ exact answer normalization.

    Output:
      - yesno: one group containing {"yes"} or {"no"}
      - factoid: one synonym group
      - list: one group per required answer item
      - summary/no exact: empty list
    """
    qtype = (qtype or "").strip().lower()

    if exact_answer in (None, ""):
        return []

    if qtype == "yesno":
        yn = _normalize_yesno_text(str(exact_answer))
        return [{yn}] if yn else []

    if isinstance(exact_answer, str):
        norm = _normalize_text(exact_answer)
        return [{norm}] if norm else []

    if isinstance(exact_answer, list):
        if qtype == "factoid":
            # BioASQ factoid exact_answer is commonly a list of synonyms,
            # sometimes nested. Treat all alternatives as one synonym group.
            group: Set[str] = set()
            for item in exact_answer:
                if isinstance(item, list):
                    for x in item:
                        n = _normalize_text(str(x))
                        if n:
                            group.add(n)
                else:
                    n = _normalize_text(str(item))
                    if n:
                        group.add(n)
            return [group] if group else []

        groups: List[Set[str]] = []
        for item in exact_answer:
            if isinstance(item, list):
                g = {_normalize_text(str(x)) for x in item if _normalize_text(str(x))}
                if g:
                    groups.append(g)
            else:
                n = _normalize_text(str(item))
                if n:
                    groups.append({n})
        return groups

    norm = _normalize_text(str(exact_answer))
    return [{norm}] if norm else []


def _gold_ideal_text(ideal_answer: Any) -> str:
    if ideal_answer is None:
        return ""
    if isinstance(ideal_answer, str):
        return ideal_answer.strip()
    if isinstance(ideal_answer, list):
        parts = [str(x).strip() for x in ideal_answer if str(x).strip()]
        return "\n".join(parts).strip()
    return str(ideal_answer).strip()


def _strip_list_prefix(line: str) -> str:
    line = re.sub(r"^\s*[-*]\s+", "", line)
    line = re.sub(r"^\s*\d+[\.)]\s+", "", line)
    line = re.sub(r"^\s*[A-Za-z][\.)]\s+", "", line)
    return line.strip()


def _extract_candidate_strings(answer: str, max_candidates: int = 10) -> List[str]:
    """
    Heuristic extraction for exact-style metrics.

    This is intentionally auxiliary: it is not a replacement for official BioASQ
    submission parsing when the system produces natural-language answers.
    """
    t = (answer or "").strip()
    if not t:
        return []

    # Try JSON answer/list first if the model returned structured content.
    obj = _try_parse_json_obj(t)
    if isinstance(obj, dict):
        for key in ["answer", "answers", "exact_answer", "items"]:
            val = obj.get(key)
            if isinstance(val, list):
                cleaned = [_normalize_text(str(x)) for x in val if _normalize_text(str(x))]
                return _dedupe_preserve_order(cleaned)[:max_candidates]
            if isinstance(val, str) and val.strip():
                t = val.strip()
                break

    t = re.sub(r"^the answer is\s+", "", t, flags=re.I)
    t = re.sub(r"^answers?\s*[:\-]\s*", "", t, flags=re.I)

    lines_raw = [ln.strip() for ln in t.splitlines() if ln.strip()]
    lines = [_strip_list_prefix(ln) for ln in lines_raw if _strip_list_prefix(ln)]

    candidates: List[str] = []

    if len(lines) >= 2:
        candidates = lines
    elif ";" in t:
        candidates = [x.strip() for x in t.split(";") if x.strip()]
    elif len(re.findall(r",", t)) >= 1 and len(t) < 300:
        comma_parts = [x.strip() for x in t.split(",") if x.strip()]
        if 1 < len(comma_parts) <= max_candidates:
            candidates = comma_parts
    elif re.search(r"\bor\b", t, flags=re.I) and len(t) < 200:
        parts = [x.strip() for x in re.split(r"\bor\b", t, flags=re.I) if x.strip()]
        if 1 < len(parts) <= max_candidates:
            candidates = parts

    if not candidates:
        candidates = [t]

    cleaned = [_normalize_text(c) for c in candidates if _normalize_text(c)]
    return _dedupe_preserve_order(cleaned)[:max_candidates]


def _macro_f1_binary(gold: List[str], pred: List[str], labels: List[str]) -> Optional[float]:
    if not gold:
        return None

    f1s: List[float] = []
    for lab in labels:
        tp = sum(1 for g, p in zip(gold, pred) if g == lab and p == lab)
        fp = sum(1 for g, p in zip(gold, pred) if g != lab and p == lab)
        fn = sum(1 for g, p in zip(gold, pred) if g == lab and p != lab)

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        f1s.append(f1)

    return float(sum(f1s) / len(f1s))


def _match_candidate_to_group(candidate: str, group: Set[str]) -> bool:
    cand = _normalize_text(candidate)
    return bool(cand and cand in group)


def _mention_match_against_group(answer: str, group: Set[str]) -> bool:
    """
    More permissive auxiliary check: does any gold synonym occur as a normalized
    phrase inside the generated answer? This is not official BioASQ exact scoring,
    but helps diagnose prose answers.
    """
    ans = f" {_normalize_text(answer)} "
    if not ans.strip():
        return False
    for g in group:
        gg = f" {_normalize_text(g)} "
        if gg.strip() and gg in ans:
            return True
    return False


# ============================================================
# DATA MODEL / ASSEMBLY
# ============================================================

@dataclass
class Example:
    id: str
    question: str
    qtype: str

    gold_documents: List[str]
    gold_snippets: List[Dict[str, Any]]
    gold_exact_answer: Any
    gold_ideal_answer: Any
    gold_exact_groups: List[Set[str]]
    gold_ideal_text: str

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


def _read_gold_jsonl_by_id(path: str) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}

    for row in iter_jsonl(path):
        rid = str(row.get("query_id") or row.get("id") or "").strip()
        if rid and rid not in out:
            out[rid] = row

    return out


def _build_examples(generations_path: str, gold_answers_path: str) -> Tuple[List[Example], Dict[str, Any]]:
    gold_by_id = _read_gold_jsonl_by_id(gold_answers_path)

    examples: List[Example] = []
    n_missing_gold = 0
    n_missing_id = 0
    n_missing_question_or_type = 0
    seen_ids: Set[str] = set()

    for raw_row in iter_jsonl(generations_path):
        g = GenerationRecord.from_dict(raw_row)
        rid = str(raw_row.get("id") or g.id or "").strip()

        if not rid:
            n_missing_id += 1
            continue

        seen_ids.add(rid)

        gold = gold_by_id.get(rid)
        if gold is None:
            n_missing_gold += 1
            continue

        question = str(raw_row.get("question") or gold.get("question") or g.question or "").strip()
        qtype = str(gold.get("type") or "").strip().lower()

        if not question or not qtype:
            n_missing_question_or_type += 1
            continue

        if qtype not in BIOASQ_KNOWN_TYPES:
            logger.warning("Unknown BioASQ question type for id=%s: %r", rid, qtype)

        gold_documents = list(gold.get("documents") or [])
        gold_snippets = list(gold.get("snippets") or [])
        gold_exact_answer = gold.get("exact_answer")
        gold_ideal_answer = gold.get("ideal_answer")
        error_message = str(raw_row.get("error", "") or "").strip() or None
        final_status_raw = str(raw_row.get("final_status", "") or "").strip().lower()

        is_generation_error = final_status_raw in {
            "error",
            "error_generation_failure",
        } or error_message is not None

        pred_answer = "" if is_generation_error else str(raw_row.get("pred_answer", "") or "")
        answer_raw = "" if is_generation_error else str(raw_row.get("answer_raw", "") or "")
        contexts = [] if is_generation_error else list(raw_row.get("contexts") or [])
        context_ids = [] if is_generation_error else list(raw_row.get("context_ids") or [])

        is_malformed = False if is_generation_error else _is_malformed_answer(pred_answer)

        if is_generation_error:
            final_status = FINAL_STATUS_ERROR_GENERATION_FAILURE
            explicit_nic = False
            forced_not_in_context = False
            forced_nic_empty_retrieval = False
            forced_nic_no_valid_citations = False
            is_grounded = False
        else:
            final_status = str(raw_row.get("final_status", "") or "")
            explicit_nic = bool(raw_row.get("explicit_nic", False))
            forced_not_in_context = bool(raw_row.get("forced_not_in_context", False))
            forced_nic_empty_retrieval = bool(raw_row.get("forced_nic_empty_retrieval", False))
            forced_nic_no_valid_citations = bool(raw_row.get("forced_nic_no_valid_citations", False))
            is_grounded = bool(raw_row.get("is_grounded", False)) and not is_malformed
        examples.append(
            Example(
                id=rid,
                question=question,
                qtype=qtype,
                gold_documents=gold_documents,
                gold_snippets=gold_snippets,
                gold_exact_answer=gold_exact_answer,
                gold_ideal_answer=gold_ideal_answer,
                gold_exact_groups=_gold_exact_groups(qtype, gold_exact_answer),
                gold_ideal_text=_gold_ideal_text(gold_ideal_answer),
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

    diagnostics = {
        "n_gold_rows": len(gold_by_id),
        "n_generation_ids_seen": len(seen_ids),
        "n_examples_loaded": len(examples),
        "n_generation_rows_missing_id": n_missing_id,
        "n_missing_gold_for_generation": n_missing_gold,
        "n_missing_question_or_type": n_missing_question_or_type,
        "n_gold_without_generation": len(set(gold_by_id) - seen_ids),
    }

    logger.info("Loaded BioASQ examples=%d diagnostics=%s", len(examples), diagnostics)
    return examples, diagnostics


# ============================================================
# LLM CORRECTNESS JUDGE
# ============================================================

def _correctness_prompt(e: Example) -> str:
    gold_exact_repr = json.dumps(e.gold_exact_answer, ensure_ascii=False)
    gold_ideal_repr = e.gold_ideal_text or "[NO IDEAL ANSWER PROVIDED]"

    return f"""
You are evaluating a biomedical QA answer for a BioASQ-style benchmark.

Question type:
{e.qtype}

Question:
{e.question}

Gold exact answer:
{gold_exact_repr}

Gold ideal answer:
{gold_ideal_repr}

Candidate answer:
{e.pred_answer if e.pred_answer.strip() else "[NO ANSWER PROVIDED]"}

Task:
Decide whether the candidate answer is semantically correct relative to the gold answer(s).

Instructions:
- Focus on factual correctness, not style, length, or wording.
- Use the gold exact answer for yes/no, factoid, and list questions.
- Use the gold ideal answer for summary-style questions and for semantic context.
- Accept synonyms, abbreviations, and biomedical paraphrases when they preserve the same factual meaning.
- For yes/no questions, the candidate must make the correct yes/no decision.
- For factoid questions, the candidate must identify the correct entity/concept.
- For list questions, the candidate should be materially correct overall; minor omissions can still be incorrect if they change the substance of the answer.
- For summary questions, the candidate should be broadly correct, clinically/biomedically coherent, and not materially misleading.
- If the answer says NOT IN CONTEXT or otherwise abstains while the gold answer is answerable, mark it incorrect.
- Reject answers that are materially incomplete, contradictory, hallucinated, or unsupported by the gold answer.

Return ONLY valid JSON. No markdown, no extra keys.

JSON schema:
{{
  "reasoning": "1-3 concise sentences",
  "verdict": "YES" or "NO"
}}
""".strip()


def _judge_cache_key(e: Example, judge_cfg: LLMConfig, prompt_version: str) -> str:
    payload = json.dumps(
        {
            "id": e.id,
            "question": e.question,
            "qtype": e.qtype,
            "gold_exact_answer": e.gold_exact_answer,
            "gold_ideal_answer": e.gold_ideal_answer,
            "pred_answer": e.pred_answer,
            "judge_backend": judge_cfg.backend,
            "judge_model": judge_cfg.model,
            "temperature": judge_cfg.temperature,
            "prompt_version": prompt_version,
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def _parse_judge_response(raw: str) -> Tuple[Optional[bool], str, str]:
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


def run_correctness_judge(
    examples: List[Example],
    judge_cfg: LLMConfig,
    cache_path: Path,
    max_retries: int = 3,
    retry_sleep_s: float = 3.0,
    prompt_version: str = CORRECTNESS_PROMPT_VERSION,
) -> Tuple[Dict[str, Any], Dict[str, Dict[str, Any]]]:
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

    acc_conditional: List[float] = []
    acc_end_to_end: List[float] = []
    type_conditional: Dict[str, List[float]] = {}
    type_end_to_end: Dict[str, List[float]] = {}

    for e in examples:
        is_system_nic = _is_system_not_in_context(e.pred_answer)

        if e.is_generation_error or e.is_malformed_answer:
            status = JUDGE_STATUS_SKIPPED_SYSTEM_FAILURE
            judge_correct = None
            reasoning = "Skipped judge call due to system failure."
            raw = ""

            status_counts[status] += 1
            acc_end_to_end.append(0.0)
            type_end_to_end.setdefault(e.qtype, []).append(0.0)

        elif is_system_nic:
            status = JUDGE_STATUS_SKIPPED_CANONICAL_NIC
            judge_correct = None
            reasoning = "Skipped judge call for canonical NOT IN CONTEXT answer."
            raw = ""

            status_counts[status] += 1
            acc_end_to_end.append(0.0)
            type_end_to_end.setdefault(e.qtype, []).append(0.0)

        else:
            ckey = _judge_cache_key(e, judge_cfg, prompt_version)

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

                prompt = _correctness_prompt(e)

                for attempt in range(1, max_retries + 1):
                    try:
                        if attempt > 1:
                            n_retries += 1

                        out = generate(prompt, judge_cfg)
                        last_raw = out
                        judge_correct, status, last_reasoning = _parse_judge_response(out)

                        if status == JUDGE_STATUS_OK:
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
                val = 1.0 if judge_correct else 0.0
                acc_conditional.append(val)
                acc_end_to_end.append(val)
                type_conditional.setdefault(e.qtype, []).append(val)
                type_end_to_end.setdefault(e.qtype, []).append(val)
            else:
                acc_end_to_end.append(0.0)
                type_end_to_end.setdefault(e.qtype, []).append(0.0)

        per_example[e.id] = {
            "judge_correct": judge_correct,
            "judge_status": status,
            "judge_reasoning": reasoning,
            "judge_raw": raw,
        }

    _safe_write_json(cache_path, cache)

    summary = {
        "judge_model": judge_cfg.model,
        "judge_backend": judge_cfg.backend,
        "prompt_version": prompt_version,
        "llm_calls": n_calls,
        "cached_labels_used": n_cached,
        "n_retries": n_retries,
        "judge_status_counts": status_counts,
        "accuracy_conditional_judged_only": _mean(acc_conditional),
        "accuracy_end_to_end_all_queries": _mean(acc_end_to_end),
        "accuracy_conditional_by_type": {
            k: _mean(v) for k, v in sorted(type_conditional.items())
        },
        "accuracy_end_to_end_by_type": {
            k: _mean(v) for k, v in sorted(type_end_to_end.items())
        },
    }

    return summary, per_example


# ============================================================
# HEURISTIC BIOASQ-STYLE METRICS
# ============================================================

def _compute_yesno_example(e: Example) -> Dict[str, Any]:
    gold = None
    if e.gold_exact_groups and e.gold_exact_groups[0]:
        gold = next(iter(e.gold_exact_groups[0]))

    pred = None
    if not (
        e.is_generation_error
        or e.is_malformed_answer
        or _is_system_not_in_context(e.pred_answer)
    ):
        pred = _extract_yesno_prediction(e.pred_answer)

    correct = (pred == gold) if (pred is not None and gold is not None) else False

    return {
        "gold_yesno": gold,
        "pred_yesno": pred,
        "parseable": pred is not None,
        "correct": correct,
    }


def _compute_factoid_example(e: Example) -> Dict[str, Any]:
    groups = e.gold_exact_groups

    preds: List[str] = []
    if not (
        e.is_generation_error
        or e.is_malformed_answer
        or _is_system_not_in_context(e.pred_answer)
    ):
        preds = _extract_candidate_strings(e.pred_answer, max_candidates=10)

    strict_acc = 0.0
    lenient_acc_at_5 = 0.0
    mrr = 0.0
    mention_match = 0.0

    if groups and preds:
        first = preds[0]
        if any(_match_candidate_to_group(first, g) for g in groups):
            strict_acc = 1.0

        for rank, cand in enumerate(preds[:5], start=1):
            if any(_match_candidate_to_group(cand, g) for g in groups):
                lenient_acc_at_5 = 1.0
                mrr = 1.0 / float(rank)
                break

    if groups and e.pred_answer:
        if any(_mention_match_against_group(e.pred_answer, g) for g in groups):
            mention_match = 1.0

    return {
        "gold_groups": [sorted(g) for g in groups],
        "pred_candidates": preds,
        "parseable": bool(preds),
        "strict_accuracy": strict_acc,
        "lenient_accuracy_at_5": lenient_acc_at_5,
        "mrr": mrr,
        "mention_match_aux": mention_match,
    }


def _compute_list_example(e: Example) -> Dict[str, Any]:
    groups = e.gold_exact_groups

    preds: List[str] = []
    if not (
        e.is_generation_error
        or e.is_malformed_answer
        or _is_system_not_in_context(e.pred_answer)
    ):
        preds = _extract_candidate_strings(e.pred_answer, max_candidates=20)

    preds = _dedupe_preserve_order(preds)

    tp = 0
    fp = 0
    matched_gold: Set[int] = set()

    for cand in preds:
        found_idx = None
        for i, group in enumerate(groups):
            if i in matched_gold:
                continue
            if _match_candidate_to_group(cand, group):
                found_idx = i
                break

        if found_idx is not None:
            tp += 1
            matched_gold.add(found_idx)
        else:
            fp += 1

    fn = max(0, len(groups) - len(matched_gold))

    precision = tp / len(preds) if preds else 0.0
    recall = tp / len(groups) if groups else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0 else 0.0
    )

    mention_hits = 0
    if groups and e.pred_answer:
        for group in groups:
            if _mention_match_against_group(e.pred_answer, group):
                mention_hits += 1

    mention_recall_aux = mention_hits / len(groups) if groups else 0.0

    return {
        "gold_groups": [sorted(g) for g in groups],
        "pred_candidates": preds,
        "parseable": bool(preds),
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mention_recall_aux": mention_recall_aux,
    }


def _compute_summary_example(e: Example, rouge_scorer: Any) -> Dict[str, Any]:
    pred = ""
    if not (
        e.is_generation_error
        or e.is_malformed_answer
        or _is_system_not_in_context(e.pred_answer)
    ):
        pred = (e.pred_answer or "").strip()

    gold = e.gold_ideal_text

    out = {
        "gold_ideal_text": gold,
        "pred_ideal_text": pred,
        "parseable": bool(pred),
        "rouge2_fmeasure": None,
        "rougeL_fmeasure": None,
    }

    if rouge_scorer is None or not pred or not gold:
        return out

    try:
        scores = rouge_scorer.score(gold, pred)
        out["rouge2_fmeasure"] = float(scores["rouge2"].fmeasure)
        out["rougeL_fmeasure"] = float(scores["rougeL"].fmeasure)
    except Exception as ex:
        out["rouge_error"] = repr(ex)

    return out


def compute_heuristic_bioasq_style_metrics(
    examples: List[Example],
) -> Tuple[Dict[str, Any], Dict[str, Dict[str, Any]], Dict[str, Any]]:
    rouge_scorer = None
    rouge_diag = {"available": False, "error": None}

    try:
        from rouge_score import rouge_scorer as rouge_scorer_mod
        rouge_scorer = rouge_scorer_mod.RougeScorer(
            ["rouge2", "rougeL"],
            use_stemmer=True,
        )
        rouge_diag["available"] = True
    except Exception as ex:
        rouge_diag["error"] = repr(ex)

    per_example: Dict[str, Dict[str, Any]] = {}

    yesno_gold: List[str] = []
    yesno_pred: List[str] = []
    yesno_acc: List[float] = []

    factoid_strict: List[float] = []
    factoid_lenient: List[float] = []
    factoid_mrr: List[float] = []
    factoid_mention: List[float] = []

    list_p: List[float] = []
    list_r: List[float] = []
    list_f1: List[float] = []
    list_mention_r: List[float] = []

    summary_r2: List[float] = []
    summary_rl: List[float] = []

    type_counts: Dict[str, int] = {}

    for e in examples:
        qtype = e.qtype
        type_counts[qtype] = type_counts.get(qtype, 0) + 1

        if qtype == "yesno":
            row = _compute_yesno_example(e)
            per_example[e.id] = {"type": qtype, **row}

            if row["gold_yesno"] is not None:
                yesno_gold.append(row["gold_yesno"])
                yesno_pred.append(row["pred_yesno"] or "unparseable")
                yesno_acc.append(1.0 if row["correct"] else 0.0)

        elif qtype == "factoid":
            row = _compute_factoid_example(e)
            per_example[e.id] = {"type": qtype, **row}

            factoid_strict.append(float(row["strict_accuracy"]))
            factoid_lenient.append(float(row["lenient_accuracy_at_5"]))
            factoid_mrr.append(float(row["mrr"]))
            factoid_mention.append(float(row["mention_match_aux"]))

        elif qtype == "list":
            row = _compute_list_example(e)
            per_example[e.id] = {"type": qtype, **row}

            list_p.append(float(row["precision"]))
            list_r.append(float(row["recall"]))
            list_f1.append(float(row["f1"]))
            list_mention_r.append(float(row["mention_recall_aux"]))

        elif qtype == "summary":
            row = _compute_summary_example(e, rouge_scorer)
            per_example[e.id] = {"type": qtype, **row}

            if _is_number(row.get("rouge2_fmeasure")):
                summary_r2.append(float(row["rouge2_fmeasure"]))
            if _is_number(row.get("rougeL_fmeasure")):
                summary_rl.append(float(row["rougeL_fmeasure"]))

        else:
            per_example[e.id] = {"type": qtype, "unsupported_type_for_exact_style": True}

    summary = {
        "metric_note": (
            "These are heuristic BioASQ-style exact/list/ROUGE metrics. "
            "They should be interpreted as auxiliary diagnostics unless the system output "
            "was constrained to official BioASQ exact-answer format."
        ),
        "type_counts": dict(sorted(type_counts.items())),
        "yesno": {
            "n": len(yesno_acc),
            "accuracy": _mean(yesno_acc),
            "macro_f1": _macro_f1_binary(yesno_gold, yesno_pred, ["yes", "no"]),
        },
        "factoid": {
            "n": len(factoid_strict),
            "strict_accuracy": _mean(factoid_strict),
            "lenient_accuracy_at_5": _mean(factoid_lenient),
            "mrr": _mean(factoid_mrr),
            "mention_match_aux": _mean(factoid_mention),
        },
        "list": {
            "n": len(list_p),
            "mean_precision": _mean(list_p),
            "mean_recall": _mean(list_r),
            "mean_f1": _mean(list_f1),
            "mention_recall_aux": _mean(list_mention_r),
        },
        "summary": {
            "n": type_counts.get("summary", 0),
            "rouge2_fmeasure_mean": _mean(summary_r2),
            "rougeL_fmeasure_mean": _mean(summary_rl),
        },
    }

    diagnostics = {"rouge": rouge_diag}
    return summary, per_example, diagnostics


# ============================================================
# RAGAS INTEGRATION
# ============================================================

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
                "Missing dependency: langchain-openai. Install with: pip install langchain-openai"
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

        return LangchainLLMWrapper(ChatOpenAI(**chat_kwargs))

    if judge_backend == "anthropic":
        try:
            from langchain_anthropic import ChatAnthropic
        except Exception as e:
            raise ImportError(
                "Missing dependency: langchain-anthropic. Install with: pip install langchain-anthropic"
            ) from e

        chat_kwargs = {
            "model": judge_model,
            "temperature": float(temperature),
            "timeout": timeout_s,
            "max_tokens": int(max_tokens),
            "api_key": api_key or os.getenv("ANTHROPIC_API_KEY"),
        }

        return LangchainLLMWrapper(ChatAnthropic(**chat_kwargs))

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
        raise ValueError("RAGAS embeddings currently support only openai/openai_compat backends.")

    try:
        from langchain_openai import OpenAIEmbeddings
    except Exception as e:
        raise ImportError(
            "Missing dependency: langchain-openai. Install with: pip install langchain-openai"
        ) from e

    emb_kwargs: Dict[str, Any] = {
        "model": emb_model,
        "api_key": emb_api_key or os.getenv("OPENAI_API_KEY"),
    }

    if emb_backend == "openai_compat":
        if not emb_base_url:
            raise ValueError("openai_compat embedding backend requires --ragas_embedding_base_url.")
        emb_kwargs["base_url"] = emb_base_url
        emb_kwargs["openai_api_base"] = emb_base_url

    return LangchainEmbeddingsWrapper(OpenAIEmbeddings(**emb_kwargs))


def run_ragas(
    examples: List[Example],
    judge_cfg: LLMConfig,
    metrics: List[str],
    ragas_embedding_model: str,
    ragas_embedding_backend: str,
    ragas_embedding_api_key: Optional[str],
    ragas_embedding_base_url: Optional[str],
    batch_size: int = 25,
    batch_sleep_s: float = 0.0,
) -> Tuple[Dict[str, float], List[Dict[str, Any]], Dict[str, Any]]:
    """
    RAGAS evaluation.

    Canonical NIC, malformed answers, and generation failures are excluded from
    RAGAS calls. They are handled later through filter-NIC and zero-NIC aggregation.
    """
    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import Faithfulness, ResponseRelevancy
    except Exception as e:
        raise ImportError("Missing ragas/datasets dependencies.") from e

    allowed = {"faithfulness", "answer_relevance"}
    for m in metrics:
        if m not in allowed:
            raise ValueError(
                f"Unsupported RAGAS metric in this script: {m}. "
                f"Allowed: {sorted(allowed)}"
            )

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

    selected_metrics = []
    if "faithfulness" in metrics:
        selected_metrics.append(Faithfulness(llm=ragas_llm))
    if "answer_relevance" in metrics:
        selected_metrics.append(ResponseRelevancy(llm=ragas_llm, embeddings=ragas_emb))

    eval_examples = [
        e for e in examples
        if not _is_system_not_in_context(e.pred_answer)
        and not e.is_malformed_answer
        and not e.is_generation_error
    ]

    diagnostics = {
        "n_eval_examples": len(eval_examples),
        "n_skipped_canonical_nic": sum(1 for e in examples if _is_system_not_in_context(e.pred_answer)),
        "n_skipped_malformed_answer": sum(1 for e in examples if e.is_malformed_answer),
        "n_skipped_generation_failure": sum(1 for e in examples if e.is_generation_error),
        "ragas_run_failed": False,
        "ragas_run_error": None,
        "batch_size": batch_size,
        "batch_sleep_s": batch_sleep_s,
        "n_batches_attempted": 0,
        "n_batches_failed": 0,
    }

    if not eval_examples:
        return {}, [{"id": e.id, **{m: None for m in metrics}} for e in examples], diagnostics

    metric_rows_by_id: Dict[str, Dict[str, Any]] = {}
    metric_values: Dict[str, List[float]] = {m: [] for m in metrics}

    batch_size = max(1, int(batch_size))
    batch_sleep_s = max(0.0, float(batch_sleep_s))

    for start in range(0, len(eval_examples), batch_size):
        batch = eval_examples[start:start + batch_size]
        diagnostics["n_batches_attempted"] += 1

        ds = Dataset.from_dict(
            {
                "question": [e.question for e in batch],
                "answer": [e.pred_answer for e in batch],
                "contexts": [e.contexts for e in batch],
            }
        )

        try:
            logger.info(
                "Running RAGAS batch %d-%d / %d",
                start + 1,
                start + len(batch),
                len(eval_examples),
            )

            res = evaluate(ds, metrics=selected_metrics)
            df = res.to_pandas()

            for i, e in enumerate(batch):
                row: Dict[str, Any] = {"id": e.id}

                if "faithfulness" in metrics and "faithfulness" in df.columns:
                    v = _safe_float(df.iloc[i]["faithfulness"])
                    row["faithfulness"] = v
                    if v is not None:
                        metric_values["faithfulness"].append(v)

                if "answer_relevance" in metrics:
                    col = "answer_relevancy" if "answer_relevancy" in df.columns else "answer_relevance"
                    if col in df.columns:
                        v = _safe_float(df.iloc[i][col])
                        row["answer_relevance"] = v
                        if v is not None:
                            metric_values["answer_relevance"].append(v)

                metric_rows_by_id[e.id] = row

        except Exception as ex:
            diagnostics["n_batches_failed"] += 1
            diagnostics["ragas_run_failed"] = True
            if diagnostics["ragas_run_error"] is None:
                diagnostics["ragas_run_error"] = repr(ex)

            logger.exception(
                "RAGAS batch failed for examples %d-%d",
                start + 1,
                start + len(batch),
            )

            for e in batch:
                metric_rows_by_id[e.id] = {"id": e.id, **{m: None for m in metrics}}

        if batch_sleep_s > 0 and (start + batch_size) < len(eval_examples):
            time.sleep(batch_sleep_s)

    summary = {
        m: float(sum(vals) / len(vals))
        for m, vals in metric_values.items()
        if vals
    }

    per_rows = []
    for e in examples:
        per_rows.append(metric_rows_by_id.get(e.id, {"id": e.id, **{m: None for m in metrics}}))

    return summary, per_rows, diagnostics


# ============================================================
# SUMMARY / TABLE BUILDING
# ============================================================

def _make_flat_summary_rows(global_summary: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    def add(metric_group: str, metric: str, value: Any) -> None:
        rows.append({
            "metric_group": metric_group,
            "metric": metric,
            "value": value,
        })

    add("basic", "n_examples", global_summary.get("n_examples"))

    for k, v in (global_summary.get("type_counts") or {}).items():
        add("type_counts", k, v)

    for k, v in (global_summary.get("nic_rates") or {}).items():
        add("nic_rates", k, v)

    for k, v in (global_summary.get("system_failure_counts") or {}).items():
        add("system_failure_counts", k, v)

    for variant, block in (global_summary.get("correctness_variants") or {}).items():
        add(f"correctness_{variant}", "correctness_mean", block.get("correctness_mean"))

    for variant, block in (global_summary.get("ragas_variants") or {}).items():
        for m, v in (block.get("metrics_mean") or {}).items():
            add(f"ragas_{variant}", m, v)

    h = _get_nested(global_summary, ["heuristic_bioasq_style", "summary"], {}) or {}
    for qtype, block in h.items():
        if isinstance(block, dict):
            for m, v in block.items():
                if m == "n":
                    add(f"heuristic_bioasq_{qtype}", "n", v)
                elif _is_number(v) or v is None:
                    add(f"heuristic_bioasq_{qtype}", m, v)

    return rows


def _make_type_correctness_rows(global_summary: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    correctness = global_summary.get("correctness") or {}
    conditional = correctness.get("accuracy_conditional_by_type") or {}
    e2e = correctness.get("accuracy_end_to_end_by_type") or {}

    qtypes = sorted(set(conditional) | set(e2e))

    for qtype in qtypes:
        rows.append({
            "type": qtype,
            "conditional_correctness": conditional.get(qtype),
            "zero_nic_correctness": e2e.get(qtype),
        })

    return rows


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "BioASQ generation evaluation. Computes LLM-judged semantic correctness, "
            "RAGAS faithfulness/relevance, and auxiliary heuristic BioASQ-style metrics."
        )
    )

    ap.add_argument("--generations", required=True)
    ap.add_argument("--gold_answers", required=True)
    ap.add_argument("--out_dir", required=True)

    ap.add_argument("--skip_correctness", action="store_true")
    ap.add_argument("--skip_ragas", action="store_true")
    ap.add_argument("--skip_heuristic_bioasq_style", action="store_true")

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

    ap.add_argument("--judge_max_retries", type=int, default=3)
    ap.add_argument("--judge_retry_sleep_s", type=float, default=3.0)

    ap.add_argument(
        "--ragas_metrics",
        default="faithfulness,answer_relevance",
        help="Supported here: faithfulness,answer_relevance",
    )
    ap.add_argument(
        "--ragas_embedding_backend",
        choices=["openai", "openai_compat"],
        default="openai",
    )
    ap.add_argument("--ragas_embedding_model", default="text-embedding-3-large")
    ap.add_argument("--ragas_embedding_api_key", default=None)
    ap.add_argument("--ragas_embedding_base_url", default=None)
    ap.add_argument("--ragas_batch_size", type=int, default=25)
    ap.add_argument("--ragas_batch_sleep_s", type=float, default=0.0)

    ap.add_argument(
        "--write_contexts",
        action="store_true",
        help="Write full retrieved contexts into per_example_eval.jsonl. Can make files large.",
    )

    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        force=True,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    examples, load_diagnostics = _build_examples(args.generations, args.gold_answers)

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

    n_total = len(examples)

    n_explicit = sum(1 for e in examples if e.explicit_nic)
    n_forced_empty = sum(1 for e in examples if e.forced_nic_empty_retrieval)
    n_forced_no_cit = sum(1 for e in examples if e.forced_nic_no_valid_citations)
    n_forced_total = sum(1 for e in examples if e.forced_not_in_context)
    n_system_nic = sum(1 for e in examples if _is_system_not_in_context(e.pred_answer))
    n_partial_nic_text = sum(
        1 for e in examples
        if not _is_system_not_in_context(e.pred_answer)
        and not e.is_generation_error
        and not e.is_malformed_answer
        and _detect_nic_text_present(e.pred_answer)
    )
    n_grounded = sum(
        1 for e in examples
        if e.is_grounded and not e.is_malformed_answer and not e.is_generation_error
    )
    n_malformed = sum(1 for e in examples if e.is_malformed_answer)
    n_gen_error = sum(1 for e in examples if e.is_generation_error)

    meta = {
        "generations": args.generations,
        "gold_answers": args.gold_answers,
        "load_diagnostics": load_diagnostics,
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
        "run_correctness": not args.skip_correctness,
        "run_ragas": not args.skip_ragas,
        "run_heuristic_bioasq_style": not args.skip_heuristic_bioasq_style,
        "ragas_metrics": args.ragas_metrics,
        "heuristic_bioasq_style_metric_note": (
            "Exact-answer/list/ROUGE metrics are heuristic diagnostics unless outputs "
            "were constrained to official BioASQ exact-answer submission format. "
            "Use LLM-judged semantic correctness as the main generation correctness metric."
        ),
        "system_failure_policy": {
            "description": (
                "Generation/API/runtime failures and malformed parsed answers are treated "
                "as system failures. They are excluded from conditional metrics and counted "
                "as zero in end-to-end metrics."
            ),
            "conditional_metrics": "excluded",
            "end_to_end_metrics": "counted_as_zero",
        },
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    _safe_write_json(out_dir / "eval_meta.json", meta)

    # Base per-example rows
    per_example: Dict[str, Dict[str, Any]] = {}

    for e in examples:
        if e.is_generation_error:
            final_status_eval = FINAL_STATUS_ERROR_GENERATION_FAILURE
        elif e.is_malformed_answer:
            final_status_eval = FINAL_STATUS_ERROR_MALFORMED_ANSWER
        else:
            final_status_eval = e.final_status

        rec = {
            "id": e.id,
            "question": e.question,
            "type": e.qtype,
            "gold_exact_answer": e.gold_exact_answer,
            "gold_ideal_answer": e.gold_ideal_answer,
            "pred_answer": e.pred_answer,
            "answer_raw": e.answer_raw,
            "context_ids": e.context_ids,
            "n_contexts": len(e.contexts),
            "explicit_nic": e.explicit_nic,
            "forced_not_in_context": e.forced_not_in_context,
            "forced_nic_empty_retrieval": e.forced_nic_empty_retrieval,
            "forced_nic_no_valid_citations": e.forced_nic_no_valid_citations,
            "final_status": final_status_eval,
            "original_final_status": e.final_status,
            "is_grounded": (
                e.is_grounded
                if not (e.is_malformed_answer or e.is_generation_error)
                else False
            ),
            "is_system_nic": _is_system_not_in_context(e.pred_answer),
            "nic_text_present": (
                _detect_nic_text_present(e.pred_answer)
                if not (e.is_malformed_answer or e.is_generation_error)
                else False
            ),
            "is_malformed_answer": e.is_malformed_answer,
            "is_generation_error": e.is_generation_error,
            "error_message": e.error_message,
        }

        if args.write_contexts:
            rec["contexts"] = e.contexts

        per_example[e.id] = rec

    # Correctness judge
    correctness_summary: Dict[str, Any] = {}
    judge_by_id: Dict[str, Dict[str, Any]] = {}

    if not args.skip_correctness:
        cache_path = out_dir / (
            f"bioasq_correctness_cache_{_safe_slug(judge_cfg.backend)}_"
            f"{_safe_slug(judge_cfg.model)}_{_safe_slug(CORRECTNESS_PROMPT_VERSION)}.json"
        )

        logger.info(
            "Running BioASQ correctness judge with %s/%s cache=%s",
            judge_cfg.backend,
            judge_cfg.model,
            cache_path.as_posix(),
        )

        correctness_summary, judge_by_id = run_correctness_judge(
            examples=examples,
            judge_cfg=judge_cfg,
            cache_path=cache_path,
            max_retries=args.judge_max_retries,
            retry_sleep_s=args.judge_retry_sleep_s,
            prompt_version=CORRECTNESS_PROMPT_VERSION,
        )
        _safe_write_json(out_dir / "correctness_summary.json", correctness_summary)

        for rid, rec in judge_by_id.items():
            if rid in per_example:
                per_example[rid]["correctness"] = {
                    "judge_correct": rec.get("judge_correct"),
                    "judge_status": rec.get("judge_status"),
                    "judge_reasoning": rec.get("judge_reasoning"),
                }

    # RAGAS
    ragas_summary: Dict[str, Any] = {}

    if not args.skip_ragas:
        metric_list = [m.strip() for m in args.ragas_metrics.split(",") if m.strip()]

        logger.info(
            "Running BioASQ RAGAS metrics=%s judge=%s/%s embedding=%s/%s",
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
            batch_size=args.ragas_batch_size,
            batch_sleep_s=args.ragas_batch_sleep_s,
        )

        ragas_summary = {
            "metrics_mean_system_from_ragas": rsum,
            "diagnostics": rdiag,
        }
        _safe_write_json(out_dir / "ragas_summary.json", ragas_summary)

        for r in per_rows:
            rid = str(r.get("id", "")).strip()
            if rid in per_example:
                per_example[rid]["ragas"] = {
                    k: v for k, v in r.items() if k != "id"
                }

    # Heuristic BioASQ-style metrics
    heuristic_summary: Dict[str, Any] = {}
    heuristic_per_example: Dict[str, Dict[str, Any]] = {}
    heuristic_diag: Dict[str, Any] = {}

    if not args.skip_heuristic_bioasq_style:
        (
            heuristic_summary,
            heuristic_per_example,
            heuristic_diag,
        ) = compute_heuristic_bioasq_style_metrics(examples)

        _safe_write_json(
            out_dir / "heuristic_bioasq_style_summary.json",
            {
                "summary": heuristic_summary,
                "diagnostics": heuristic_diag,
            },
        )

        for rid, rec in heuristic_per_example.items():
            if rid in per_example:
                per_example[rid]["heuristic_bioasq_style"] = rec

    # Global summary
    global_summary: Dict[str, Any] = {
        "n_examples": n_total,
        "type_counts": {
            qtype: sum(1 for e in examples if e.qtype == qtype)
            for qtype in sorted(set(e.qtype for e in examples))
        },
        "nic_rates": {
            "explicit_nic_rate": _rate(n_explicit, n_total),
            "forced_nic_empty_retrieval_rate": _rate(n_forced_empty, n_total),
            "forced_nic_no_valid_citations_rate": _rate(n_forced_no_cit, n_total),
            "forced_not_in_context_rate": _rate(n_forced_total, n_total),
            "system_not_in_context_rate": _rate(n_system_nic, n_total),
            "partial_nic_text_rate": _rate(n_partial_nic_text, n_total),
            "grounded_rate": _rate(n_grounded, n_total),
        },
        "nic_counts": {
            "n_total": n_total,
            "n_explicit_nic": n_explicit,
            "n_forced_nic_empty_retrieval": n_forced_empty,
            "n_forced_nic_no_valid_citations": n_forced_no_cit,
            "n_forced_not_in_context": n_forced_total,
            "n_system_not_in_context": n_system_nic,
            "n_partial_nic_text": n_partial_nic_text,
            "n_grounded": n_grounded,
        },
        "system_failure_counts": {
            "n_generation_failure": n_gen_error,
            "n_malformed_answer": n_malformed,
            "n_system_failures_total": n_gen_error + n_malformed,
            "generation_failure_rate": _rate(n_gen_error, n_total),
            "malformed_answer_rate": _rate(n_malformed, n_total),
            "system_failure_rate_total": _rate(n_gen_error + n_malformed, n_total),
        },
        "correctness": correctness_summary,
        "ragas": ragas_summary,
        "heuristic_bioasq_style": {
            "summary": heuristic_summary,
            "diagnostics": heuristic_diag,
        },
    }

    # Correctness variants: filter-NIC and zero-NIC
    if not args.skip_correctness:
        global_summary["correctness_variants"] = {
            "filter_nic": {
                "description": (
                    "Metrics over usable non-NIC outputs only. Excludes canonical NIC "
                    "and system failures."
                ),
                "correctness_mean": None,
            },
            "zero_nic": {
                "description": (
                    "All queries; canonical NIC and system failures count as 0."
                ),
                "correctness_mean": None,
            },
        }

        for variant_name in ["filter_nic", "zero_nic"]:
            vals: List[float] = []

            for _, rec in per_example.items():
                is_system_nic = bool(rec.get("is_system_nic", False))
                is_malformed = bool(rec.get("is_malformed_answer", False))
                is_generation_error = bool(rec.get("is_generation_error", False))

                if variant_name == "zero_nic" and (
                    is_system_nic or is_malformed or is_generation_error
                ):
                    vals.append(0.0)
                    continue

                if variant_name == "filter_nic" and (
                    is_system_nic or is_malformed or is_generation_error
                ):
                    continue

                judge_correct = rec.get("correctness", {}).get("judge_correct")
                judge_status = rec.get("correctness", {}).get("judge_status")

                if judge_status == JUDGE_STATUS_OK and isinstance(judge_correct, bool):
                    vals.append(1.0 if judge_correct else 0.0)
                elif variant_name == "zero_nic":
                    vals.append(0.0)

            global_summary["correctness_variants"][variant_name]["correctness_mean"] = _mean(vals)

    # RAGAS variants: filter-NIC and zero-NIC
    if not args.skip_ragas:
        metric_list = [m.strip() for m in args.ragas_metrics.split(",") if m.strip()]

        global_summary["ragas_variants"] = {
            "filter_nic": {
                "description": (
                    "Metrics over usable non-NIC outputs only. Excludes canonical NIC "
                    "and system failures."
                ),
                "metrics_mean": {},
            },
            "zero_nic": {
                "description": (
                    "All queries; canonical NIC and system failures count as 0."
                ),
                "metrics_mean": {},
            },
        }

        for variant_name in ["filter_nic", "zero_nic"]:
            for m in metric_list:
                vals: List[float] = []

                for _, rec in per_example.items():
                    is_system_nic = bool(rec.get("is_system_nic", False))
                    is_malformed = bool(rec.get("is_malformed_answer", False))
                    is_generation_error = bool(rec.get("is_generation_error", False))

                    if variant_name == "zero_nic" and (
                        is_system_nic or is_malformed or is_generation_error
                    ):
                        vals.append(0.0)
                        continue

                    if variant_name == "filter_nic" and (
                        is_system_nic or is_malformed or is_generation_error
                    ):
                        continue

                    v = rec.get("ragas", {}).get(m)
                    fv = _safe_float(v)
                    if fv is not None:
                        vals.append(fv)
                    elif variant_name == "zero_nic":
                        vals.append(0.0)

                global_summary["ragas_variants"][variant_name]["metrics_mean"][m] = _mean(vals)

    # Write outputs
    per_rows = [per_example[rid] for rid in sorted(per_example.keys())]
    _write_jsonl(out_dir / "per_example_eval.jsonl", per_rows)

    _safe_write_json(out_dir / "global_summary.json", global_summary)

    summary_rows = _make_flat_summary_rows(global_summary)
    _write_csv(out_dir / "summary_metrics.csv", summary_rows)

    type_rows = _make_type_correctness_rows(global_summary)
    _write_csv(out_dir / "correctness_by_type.csv", type_rows)

    logger.info(
        "Done. Wrote: %s and %s",
        (out_dir / "per_example_eval.jsonl").as_posix(),
        (out_dir / "global_summary.json").as_posix(),
    )


if __name__ == "__main__":
    main()