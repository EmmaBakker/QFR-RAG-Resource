#!/usr/bin/env python3
"""
BioASQ chunking (PubMed title+abstract) with QFR-aligned constraints.

Reads:
  data/bioasq/processed/docs.jsonl

Writes:
  data/bioasq/processed/corpus.chunks.jsonl
  data/bioasq/processed/corpus.chunks.report.json

"""

from __future__ import annotations

import json
import hashlib
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from transformers import AutoTokenizer


# Config (match QFR where possible)
BIOASQ_DIR = Path("data/bioasq")
IN_DOCS = BIOASQ_DIR / "processed" / "docs.jsonl"

OUT_CHUNKS = BIOASQ_DIR / "processed" / "corpus.chunks.jsonl"
OUT_REPORT = BIOASQ_DIR / "processed" / "corpus.chunks.report.json"

CANONICAL_TOKENIZER_NAME = "ncbi/MedCPT-Query-Encoder"
CANONICAL_TOKENIZER_TRUST_REMOTE_CODE = False

MODEL_MAX_LEN_INCLUDING_SPECIAL = 512
MAX_PAYLOAD_TOKENS = 510
TARGET_PAYLOAD_TOKENS = 450

OVERLAP_TOKENS = 64

# Minimum chunk length (chosen for PubMed abstracts)
MIN_CHUNK_TOKENS = 80

# Prefix line
ADD_TITLE_PREFIX_LINE = True
TITLE_PREFIX_STYLE = "Title: {title}"

# Sentence splitting + overlap snapping
SENT_SPLIT_RX = re.compile(r"(?<=[.!?])\s+")
NUM_PREFIX_RX = re.compile(r"^\s*(?:\d+(\.\d+)*[\)\.]|[IVXLCDM]+[\)\.]|\(?[a-zA-Z]\)|[a-zA-Z]\.)\s*")


# Data model
@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    chunk_idx: int

    # doc provenance
    pmid: Optional[str]
    year: Optional[str]
    journal: Optional[str]

    # overlap audit
    overlap_tokens_requested: int
    overlap_tokens_used: int
    overlap_from_chunk_id: Optional[str]

    # content
    prefix: str
    body: str
    text: str
    token_len_payload: int
    text_hash: str


# Helpers
def sha1_text(s: str) -> str:
    return hashlib.sha1((s or "").encode("utf-8")).hexdigest()


def token_ids_payload(tokenizer, text: str) -> List[int]:
    return tokenizer.encode(text, add_special_tokens=False)


def decode_payload(tokenizer, ids: List[int]) -> str:
    return tokenizer.decode(ids, skip_special_tokens=True)


def payload_len(tokenizer, text: str) -> int:
    return len(token_ids_payload(tokenizer, text))


def clean_title_for_prefix(title: str) -> str:
    t = (title or "").strip()
    t = re.sub(r"\s+", " ", t).strip()
    t = NUM_PREFIX_RX.sub("", t).strip()
    if len(t) > 160:
        t = t[:157].rstrip() + "..."
    return t


def split_paragraph_by_sentences(text: str) -> List[str]:
    t = (text or "").strip()
    if not t:
        return []
    parts: List[str] = []
    for para in re.split(r"\n{2,}", t):
        para = para.strip()
        if not para:
            continue
        sents = SENT_SPLIT_RX.split(para)
        sents = [s.strip() for s in sents if s.strip()]
        parts.extend(sents)
    return parts


def snap_to_sentence_start(s: str) -> str:
    """
    Remove leading fragment so overlap prefix starts near a sentence boundary.
    Same heuristic as your QFR chunker.
    """
    if not s:
        return s
    t = s.strip()
    if not t:
        return t
    window = t[:240]
    m = re.search(r"([.!?])\s+|\n+", window)
    if not m:
        return t
    cut = m.end()
    if cut >= len(t) - 20:
        return t
    return t[cut:].lstrip()


def pick_prefix(title: str) -> str:
    if not ADD_TITLE_PREFIX_LINE:
        return ""
    tt = clean_title_for_prefix(title)
    if not tt:
        return ""
    return TITLE_PREFIX_STYLE.format(title=tt).strip()


def pack_text_into_raw_chunks(tokenizer, body_text: str) -> List[str]:
    """
    Create raw BODY-only chunks under token budgets.
    We pack sentences until TARGET_PAYLOAD_TOKENS; never exceed MAX_PAYLOAD_TOKENS.
    """
    body_text = (body_text or "").strip()
    if not body_text:
        return []

    if payload_len(tokenizer, body_text) <= MAX_PAYLOAD_TOKENS:
        return [body_text]

    sents = split_paragraph_by_sentences(body_text)
    if not sents:
        # fallback hard token split
        ids = token_ids_payload(tokenizer, body_text)
        out = []
        for i in range(0, len(ids), MAX_PAYLOAD_TOKENS):
            part = decode_payload(tokenizer, ids[i:i + MAX_PAYLOAD_TOKENS]).strip()
            if part:
                out.append(part)
        return out

    raw: List[str] = []
    cur: List[str] = []

    def cur_text() -> str:
        return " ".join(cur).strip()

    def flush():
        nonlocal cur
        txt = cur_text()
        if txt:
            raw.append(txt)
        cur = []

    for s in sents:
        cand = (" ".join(cur + [s])).strip()
        cand_len = payload_len(tokenizer, cand)

        if cand_len <= TARGET_PAYLOAD_TOKENS:
            cur.append(s)
            continue

        if cand_len <= MAX_PAYLOAD_TOKENS:
            cur.append(s)
            flush()
            continue

        # Would exceed MAX: flush current and handle long sentence
        flush()
        if payload_len(tokenizer, s) <= MAX_PAYLOAD_TOKENS:
            raw.append(s)
        else:
            ids = token_ids_payload(tokenizer, s)
            for i in range(0, len(ids), MAX_PAYLOAD_TOKENS):
                part = decode_payload(tokenizer, ids[i:i + MAX_PAYLOAD_TOKENS]).strip()
                if part:
                    raw.append(part)

    flush()
    return raw


def apply_overlap_within_doc(tokenizer, raw_bodies: List[str]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    prev_body_ids: Optional[List[int]] = None

    for body in raw_bodies:
        body = body.strip()
        body_ids = token_ids_payload(tokenizer, body)

        overlap_used = 0
        overlap_text = ""

        if prev_body_ids is not None and OVERLAP_TOKENS > 0:
            ov_ids = prev_body_ids[-OVERLAP_TOKENS:] if OVERLAP_TOKENS <= len(prev_body_ids) else prev_body_ids[:]

            # ensure overlap + current body <= MAX_PAYLOAD_TOKENS (body-only cap)
            max_ov = max(0, MAX_PAYLOAD_TOKENS - len(body_ids))
            if len(ov_ids) > max_ov:
                ov_ids = ov_ids[-max_ov:] if max_ov > 0 else []

            if ov_ids:
                ov_text = decode_payload(tokenizer, ov_ids).strip()
                ov_text = snap_to_sentence_start(ov_text)
                if ov_text:
                    ov_ids2 = token_ids_payload(tokenizer, ov_text)

                    # re-cap after snapping
                    max_ov2 = max(0, MAX_PAYLOAD_TOKENS - len(body_ids))
                    if len(ov_ids2) > max_ov2:
                        ov_ids2 = ov_ids2[-max_ov2:] if max_ov2 > 0 else []
                        ov_text = decode_payload(tokenizer, ov_ids2).strip()
                        ov_text = snap_to_sentence_start(ov_text)
                        ov_ids2 = token_ids_payload(tokenizer, ov_text) if ov_text else []

                    if ov_text and ov_ids2:
                        overlap_text = ov_text
                        overlap_used = len(ov_ids2)

        final_body = (overlap_text + "\n\n" + body).strip() if overlap_text else body

        out.append({"body": final_body, "overlap_used": overlap_used})

        prev_body_ids = body_ids

    return out


def main() -> None:
    if not IN_DOCS.exists():
        raise SystemExit(f"Missing input: {IN_DOCS}")

    tokenizer = AutoTokenizer.from_pretrained(
        CANONICAL_TOKENIZER_NAME,
        trust_remote_code=CANONICAL_TOKENIZER_TRUST_REMOTE_CODE,
        use_fast=True,
    )

    OUT_CHUNKS.parent.mkdir(parents=True, exist_ok=True)

    num_docs_in = 0
    num_docs_out = 0
    total_chunks = 0
    dropped_too_short = 0
    global_token_lens: List[int] = []

    with IN_DOCS.open("r", encoding="utf-8") as f_in, OUT_CHUNKS.open("w", encoding="utf-8", newline="\n") as f_out:
        for ln in f_in:
            ln = ln.strip()
            if not ln:
                continue
            num_docs_in += 1
            doc = json.loads(ln)

            doc_id = str(doc.get("doc_id", "")).strip()
            if not doc_id:
                continue

            title = str(doc.get("title", "") or "").strip()
            abstract = str(doc.get("abstract", "") or "").strip()

            # BODY POLICY
            body_text = abstract.strip() if abstract.strip() else title.strip()
            if not body_text:
                continue

            prefix = pick_prefix(title)

            raw_bodies = pack_text_into_raw_chunks(tokenizer, body_text)
            overlapped = apply_overlap_within_doc(tokenizer, raw_bodies)

            prev_chunk_id: Optional[str] = None
            kept_any = False

            for i, item in enumerate(overlapped):
                final_body = (item["body"] or "").strip()
                overlap_used = int(item["overlap_used"])
                overlap_from = prev_chunk_id if overlap_used > 0 else None

                # Compose final text with prefix (prefix not part of overlap)
                final_text = (prefix + "\n\n" + final_body).strip() if prefix else final_body

                # Enforce MAX_PAYLOAD_TOKENS INCLUDING prefix
                ids = token_ids_payload(tokenizer, final_text)
                prefix_out = prefix
                if len(ids) > MAX_PAYLOAD_TOKENS:
                    # If prefix pushes over, drop it first (same as QFR)
                    if prefix:
                        final_text = final_body
                        prefix_out = ""
                        ids = token_ids_payload(tokenizer, final_text)

                    # Still too long: hard trim
                    if len(ids) > MAX_PAYLOAD_TOKENS:
                        final_text = decode_payload(tokenizer, ids[:MAX_PAYLOAD_TOKENS]).strip()
                        ids = token_ids_payload(tokenizer, final_text)

                # Minimum-length filter
                if len(ids) < MIN_CHUNK_TOKENS:
                    dropped_too_short += 1
                    continue

                chunk_id = f"{doc_id}_chunk{i:06d}"
                prev_chunk_id = chunk_id

                ch = Chunk(
                    chunk_id=chunk_id,
                    doc_id=doc_id,
                    chunk_idx=i,
                    pmid=str(doc.get("pmid") or ""),
                    year=str(doc.get("year") or ""),
                    journal=str(doc.get("journal") or ""),
                    overlap_tokens_requested=OVERLAP_TOKENS,
                    overlap_tokens_used=overlap_used,
                    overlap_from_chunk_id=overlap_from,
                    prefix=prefix_out,
                    body=final_body,
                    text=final_text,
                    token_len_payload=len(ids),
                    text_hash=sha1_text(final_text),
                )

                f_out.write(json.dumps(asdict(ch), ensure_ascii=False) + "\n")
                global_token_lens.append(len(ids))
                total_chunks += 1
                kept_any = True

            if kept_any:
                num_docs_out += 1

    token_lens_sorted = sorted(global_token_lens)
    report: Dict[str, Any] = {
        "config": {
            "tokenizer": CANONICAL_TOKENIZER_NAME,
            "model_max_len_including_special": MODEL_MAX_LEN_INCLUDING_SPECIAL,
            "max_payload_tokens": MAX_PAYLOAD_TOKENS,
            "target_payload_tokens": TARGET_PAYLOAD_TOKENS,
            "overlap_tokens": OVERLAP_TOKENS,
            "overlap_is_body_only": True,
            "overlap_sentence_snap": True,
            "add_title_prefix_line": ADD_TITLE_PREFIX_LINE,
            "title_prefix_style": TITLE_PREFIX_STYLE if ADD_TITLE_PREFIX_LINE else None,
            "min_chunk_tokens": MIN_CHUNK_TOKENS,
            "body_policy": "abstract_only_else_title",
        },
        "global": {
            "num_docs_in": num_docs_in,
            "num_docs_with_kept_chunks": num_docs_out,
            "num_chunks": total_chunks,
            "num_chunks_dropped_too_short": dropped_too_short,
            "chunk_token_len_min": min(global_token_lens) if global_token_lens else None,
            "chunk_token_len_p50": token_lens_sorted[len(token_lens_sorted) // 2] if token_lens_sorted else None,
            "chunk_token_len_p90": token_lens_sorted[int(0.9 * (len(token_lens_sorted) - 1))] if token_lens_sorted else None,
            "chunk_token_len_max": max(global_token_lens) if global_token_lens else None,
        },
    }

    OUT_REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Wrote chunks: {OUT_CHUNKS} ({total_chunks} chunks)")
    print(f"Wrote report: {OUT_REPORT}")
    print(f"Dropped too-short chunks (<{MIN_CHUNK_TOKENS} tokens): {dropped_too_short}")


if __name__ == "__main__":
    main()