"""
Shared schema for the generation → evaluation pipeline.

GenerationRecord is the canonical unit written by gen_from_trace.py and read
by evaluation.py.  Keeping it in one place makes the JSONL contract explicit
and type-checked rather than scattered across inline dicts and string .get()
calls.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

# ── Canonical abstention token ────────────────────────────────────────────────
# Single source of truth.  Import this instead of hard-coding the string.
NIC_TOKEN: str = "NOT IN CONTEXT"

# ── final_status values ───────────────────────────────────────────────────────
FINAL_STATUS_ANSWERED = "answered"
FINAL_STATUS_EXPLICIT_NIC = "explicit_nic"
FINAL_STATUS_FORCED_NIC_EMPTY = "forced_nic_empty_retrieval"
FINAL_STATUS_FORCED_NIC_NO_CIT = "forced_nic_no_valid_citations"
FINAL_STATUS_ERROR = "error"

VALID_FINAL_STATUSES = frozenset(
    {
        FINAL_STATUS_ANSWERED,
        FINAL_STATUS_EXPLICIT_NIC,
        FINAL_STATUS_FORCED_NIC_EMPTY,
        FINAL_STATUS_FORCED_NIC_NO_CIT,
        FINAL_STATUS_ERROR,
    }
)


# ── Generation record ─────────────────────────────────────────────────────────

@dataclass
class GenerationRecord:
    """
    One row in generations.jsonl.

    Separation of concerns
    ──────────────────────
    raw_output / *_raw   — verbatim model output before any enforcement.
    pred_answer / citations — post-enforcement values used by evaluation.
    final_status          — single authoritative outcome tag.
    error                 — non-None only when the LLM call itself failed
                            (final_status == "error"); never conflated with NIC.
    llm_info              — backend timing/token metadata (Ollama eval_count etc.).
    """

    # ── Identity ──────────────────────────────────────────────────────────────
    id: str
    question: str

    # ── LLM input / raw output ────────────────────────────────────────────────
    prompt: Optional[str]
    raw_output: str

    # ── Parsed before enforcement ─────────────────────────────────────────────
    answer_raw: str
    citations_raw: List[str]

    # ── After enforcement ─────────────────────────────────────────────────────
    pred_answer: str            # NIC_TOKEN or model answer text
    citations: List[str]        # valid (within retrieved set)
    invalid_citations: List[str]
    num_citations: int

    # ── Outcome classification ────────────────────────────────────────────────
    explicit_nic: bool                  # model itself returned NIC_TOKEN
    forced_nic_empty_retrieval: bool    # no chunks retrieved
    forced_nic_no_valid_citations: bool # answered but zero valid citations
    # derived: True when either forced flag is True (NOT when explicit_nic is True)
    forced_not_in_context: bool
    is_grounded: bool
    # One of VALID_FINAL_STATUSES
    final_status: str

    # ── Retrieval context ─────────────────────────────────────────────────────
    retrieved: List[Dict[str, Any]]  # [{chunk_id, score}]
    contexts: List[str]
    context_ids: List[str]
    context_scores: List[float]

    # ── Prompt statistics ─────────────────────────────────────────────────────
    prompt_chars: int
    prompt_words: int
    prompt_token_est: int

    # ── Timing / backend metadata ─────────────────────────────────────────────
    llm_latency_s: Optional[float]
    llm_info: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None

    # ── Serialisation ─────────────────────────────────────────────────────────

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "GenerationRecord":
        """
        * llm_info: missing in pre-schema records → default {}
        * final_status: reconstructed from boolean flags if absent
        * FINAL_STATUS_ERROR: was not emitted by older gen_from_trace → keep
          as-is; evaluation.py handles it via the is_error property.
        """
        # Reconstruct final_status for records written before schema.py existed.
        final_status = str(d.get("final_status") or "").strip()
        if not final_status:
            explicit_nic = bool(d.get("explicit_nic", False))
            forced_empty = bool(d.get("forced_nic_empty_retrieval", False))
            forced_no_cit = bool(d.get("forced_nic_no_valid_citations", False))
            if explicit_nic:
                final_status = FINAL_STATUS_EXPLICIT_NIC
            elif forced_empty:
                final_status = FINAL_STATUS_FORCED_NIC_EMPTY
            elif forced_no_cit:
                final_status = FINAL_STATUS_FORCED_NIC_NO_CIT
            else:
                final_status = FINAL_STATUS_ANSWERED

        explicit_nic = bool(d.get("explicit_nic", False))
        forced_empty = bool(d.get("forced_nic_empty_retrieval", False))
        forced_no_cit = bool(d.get("forced_nic_no_valid_citations", False))

        return cls(
            id=str(d.get("id", "")),
            question=str(d.get("question", "")),
            prompt=d.get("prompt"),
            raw_output=str(d.get("raw_output", "")),
            answer_raw=str(d.get("answer_raw", "")),
            citations_raw=list(d.get("citations_raw") or []),
            pred_answer=str(d.get("pred_answer", "")),
            citations=list(d.get("citations") or []),
            invalid_citations=list(d.get("invalid_citations") or []),
            num_citations=int(d.get("num_citations", 0)),
            explicit_nic=explicit_nic,
            forced_nic_empty_retrieval=forced_empty,
            forced_nic_no_valid_citations=forced_no_cit,
            forced_not_in_context=bool(d.get("forced_not_in_context", False)),
            is_grounded=bool(d.get("is_grounded", False)),
            final_status=final_status,
            retrieved=list(d.get("retrieved") or []),
            contexts=list(d.get("contexts") or []),
            context_ids=list(d.get("context_ids") or []),
            context_scores=[float(x) for x in (d.get("context_scores") or [])],
            prompt_chars=int(d.get("prompt_chars", 0)),
            prompt_words=int(d.get("prompt_words", 0)),
            prompt_token_est=int(d.get("prompt_token_est", 0)),
            llm_latency_s=d.get("llm_latency_s"),
            llm_info=dict(d.get("llm_info") or {}),
            error=d.get("error"),
        )

    @property
    def is_error(self) -> bool:
        return self.final_status == FINAL_STATUS_ERROR

