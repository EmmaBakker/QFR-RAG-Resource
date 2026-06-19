from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

_ANSWER_RX = re.compile(r"(?is)\bANSWER\s*:\s*(.*?)(?:\n\s*\bCITATIONS\s*:|$)")
_CIT_BLOCK_RX = re.compile(r"(?is)\bCITATIONS\s*:\s*(.*)$")

_CHUNK_ID_RE = re.compile(r"(doc[a-z0-9]{8,}_(?:chunk)?\d{6,})", re.IGNORECASE)

# Fenced JSON block extraction: allow ```json ...``` or ```JSON ...```
_FENCED_JSON_RX = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.IGNORECASE | re.DOTALL)


def _normalize_text(text: str) -> str:
    return (text or "").replace("\\_", "_").strip()


def _extract_fenced_or_raw(text: str) -> str:
    t = _normalize_text(text)
    if not t:
        return ""
    m = _FENCED_JSON_RX.search(t)
    return (m.group(1) if m else t).strip()


def _extract_first_json_object(text: str) -> str:
    """
    Extract the first balanced {...} JSON object from text.
    Returns "" if not found.
    """
    t = _normalize_text(text)
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

        # not in string
        if ch == "\"":
            in_str = True
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return t[start : i + 1]

    return ""


def _try_parse_json(text: str) -> Optional[Dict[str, Any]]:
    candidate = _extract_fenced_or_raw(text)
    if not candidate:
        return None

    # 1) Try direct parse
    try:
        obj = json.loads(candidate)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    # 2) Try extracting a balanced JSON object from candidate
    obj_str = _extract_first_json_object(candidate)
    if not obj_str:
        return None

    try:
        obj2 = json.loads(obj_str)
    except Exception:
        return None

    return obj2 if isinstance(obj2, dict) else None



# ── Unified parse (parses JSON exactly once) ──────────────────────────────────

@dataclass
class ParsedOutput:
    """Fields extracted from a single raw model response."""
    answer: str
    citations: List[str]


def parse_all(raw: str) -> ParsedOutput:
    """
    Parse raw model output in a single pass.

    Prefer this over calling parse_answer / parse_citations separately: those
    each call _try_parse_json independently, meaning the same JSON object is
    parsed twice per response.

    Fallback (non-JSON) paths are also unified here.
    """
    t = _normalize_text(raw)
    if not t:
        return ParsedOutput(answer="", citations=[])

    # ── JSON path ─────────────────────────────────────────────────────────────
    obj = _try_parse_json(t)
    if obj is not None:
        answer = str(obj.get("answer", "") or "").strip()

        citations: List[str] = []
        if isinstance(obj.get("citations"), list):
            seen: set = set()
            for c in obj["citations"]:
                c_str = str(c).strip()
                if not c_str:
                    continue
                m = _CHUNK_ID_RE.search(c_str)
                cid = m.group(1) if m else c_str
                if cid and cid not in seen:
                    citations.append(cid)
                    seen.add(cid)

        return ParsedOutput(answer=answer, citations=citations)

    # ── Regex / plaintext fallback ────────────────────────────────────────────
    answer = ""
    m_ans = _ANSWER_RX.search(t)
    if m_ans:
        ans = (m_ans.group(1) or "").strip()
        answer = re.sub(r"^\s*-\s*", "", ans).strip()
    else:
        for ln in t.splitlines():
            ln = ln.strip()
            if ln:
                answer = ln
                break

    m_cit = _CIT_BLOCK_RX.search(t)
    scope = m_cit.group(1) if m_cit else t
    cids = _CHUNK_ID_RE.findall(scope)
    seen_set: set = set()
    citations_fb: List[str] = []
    for c in cids:
        cid = c.strip()
        if cid and cid not in seen_set:
            citations_fb.append(cid)
            seen_set.add(cid)

    return ParsedOutput(answer=answer, citations=citations_fb)

