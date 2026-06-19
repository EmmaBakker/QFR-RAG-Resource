#!/usr/bin/env python3
"""
Reads:   data/processed/*.blocks.jsonl
Writes:  data/processed/corpus.chunks.jsonl
         data/processed/corpus.chunks.report.json

1) Candidate section segments:
   - Slides: segment key prefers section_path (often slide title).
   - Non-slides: segment key prefers section_id (normalized stable key), fallback section_path.
   - Heading blocks are NOT included in the chunk body (avoids breadcrumb spam),
     BUT we DO add a single compact prefix line per chunk like: "Section: Results".

2) Coalesce tiny segments forward:
   - Title-only (empty-text) segments are carried forward as metadata.
   - Tiny segments are merged forward until MIN_FINAL_TOKENS (unless last).
   - Optional merge-back of last tiny unit.

3) Chunk packing:
   - Token budget: MAX_PAYLOAD_TOKENS=510 (MedCPT model_max_len=512 incl specials).
   - Target payload ~450 to leave slack.
   - Tables/lists are atomic; split only if a single block exceeds the cap:
       * tables split by rows via table_struct when available
       * lists split by item boundaries
   - Paragraphs split by sentence boundaries if a single paragraph exceeds cap.

4) Overlap:
   - Token-level overlap, ONLY within the same merged Unit (never across units).
   - Papers/manuals: overlap 64; slides: overlap 40.
   - Overlap is computed from BODY TEXT ONLY (not from the "Section:" prefix line).
   - Overlap prefix is sentence-snapped to avoid starting mid-sentence.


"""

from __future__ import annotations

import json
import hashlib
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from collections import Counter

from transformers import AutoTokenizer


# Config
PROCESSED_DIR = Path("data/processed")
IN_GLOB = "*.blocks.jsonl"

OUT_CHUNKS = PROCESSED_DIR / "corpus.chunks.jsonl"
OUT_REPORT = PROCESSED_DIR / "corpus.chunks.report.json"

CANONICAL_TOKENIZER_NAME = "ncbi/MedCPT-Query-Encoder"
CANONICAL_TOKENIZER_TRUST_REMOTE_CODE = False

MODEL_MAX_LEN_INCLUDING_SPECIAL = 512
MAX_PAYLOAD_TOKENS = 510              # safe payload budget (512 - ~2 specials)
TARGET_PAYLOAD_TOKENS = 450           # target payload (slack for prefix)

DEFAULT_OVERLAP_TOKENS = 64           # papers/manuals
SLIDES_OVERLAP_TOKENS = 40            # slides

# Coalescing thresholds
MIN_SECTION_TOKENS = 90               # reported; coalescing driven primarily by MIN_FINAL_TOKENS
MIN_FINAL_TOKENS = 140                # do not emit unit smaller than this (unless last)

MERGE_BACK_IF_LAST_TOO_SMALL = True   # merge last tiny unit backward

# Prefix line (for retrieval usefulness)
ADD_SECTION_PREFIX_LINE = True
SECTION_PREFIX_STYLE = "Section: {title}"  # short, clean

# Block behavior
ATOMIC_BLOCK_TYPES = {"table", "list", "figure_caption"}
HEADING_TYPES = {"heading"}

# Sentence splitting (for splitting long paragraphs and snapping overlap)
SENT_SPLIT_RX = re.compile(r"(?<=[.!?])\s+")
NUM_PREFIX_RX = re.compile(r"^\s*(?:\d+(\.\d+)*[\)\.]|[IVXLCDM]+[\)\.]|\(?[a-zA-Z]\)|[a-zA-Z]\.)\s*")

# --- Drop policies (requested) ---
DROP_TABLE_OF_CONTENTS = True
DROP_TINY_CAPTION_CHUNKS = True
MIN_TINY_DROP_TOKENS = 80  # drop tiny caption-only chunks below this

TOC_RX = re.compile(r"^\s*(table\s+of\s+contents|contents|toc)\s*$", re.IGNORECASE)
TINY_CAPTION_RX = re.compile(r"^\s*(online\s+)?(table|figure)\s*\d+\b", re.IGNORECASE)

#-----------


@dataclass
class Block:
    doc_id: str
    source_file: str
    block_id: str
    order_idx: int
    block_type: str
    page_start: Optional[int]
    page_end: Optional[int]
    section_path: Optional[str]
    text: str
    char_len: int
    text_hash: str
    section_title: Optional[str] = None
    section_id: Optional[str] = None
    table_struct: Optional[Dict[str, Any]] = None


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    source_file: str
    chunk_idx: int

    page_start: Optional[int]
    page_end: Optional[int]

    # provenance / audit
    section_paths: List[str]
    section_ids: List[str]
    block_ids: List[str]
    block_types: List[str]

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


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            rows.append(json.loads(ln))
    return rows


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def safe_int(x: Any) -> Optional[int]:
    try:
        return int(x)
    except Exception:
        return None


def looks_like_slides_filename(name: str) -> bool:
    n = (name or "").lower()
    return ("slide" in n) or ("slides" in n) or ("ppt" in n) or ("presentation" in n)


def split_section_path(section_path: Optional[str]) -> List[str]:
    if not section_path:
        return []
    return [p.strip() for p in section_path.split(">") if p.strip()]


def section_leaf(section_path: Optional[str]) -> str:
    parts = split_section_path(section_path)
    return parts[-1] if parts else (section_path or "").strip()


def clean_title_for_prefix(title: str) -> str:
    t = (title or "").strip()
    t = re.sub(r"\s+", " ", t).strip()
    # drop leading numbering/bullets like "1.", "1.2", "A)", "IV."
    t = NUM_PREFIX_RX.sub("", t).strip()
    # avoid overlong prefix lines
    if len(t) > 140:
        t = t[:137].rstrip() + "..."
    return t


def token_ids_payload(tokenizer, text: str) -> List[int]:
    return tokenizer.encode(text, add_special_tokens=False)


def decode_payload(tokenizer, ids: List[int]) -> str:
    return tokenizer.decode(ids, skip_special_tokens=True)


def payload_len(tokenizer, text: str) -> int:
    return len(token_ids_payload(tokenizer, text))


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
    Heuristic: find earliest boundary like '. ' '? ' '! ' or newline within first ~240 chars.
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


def stable_uniq(seq: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for x in seq:
        x = (x or "").strip()
        if not x or x in seen:
            continue
        seen.add(x)
        out.append(x)
    return out


def is_toc_section(section_path: Optional[str], section_id: Optional[str]) -> bool:
    if not DROP_TABLE_OF_CONTENTS:
        return False
    leaf = section_leaf(section_path).strip() if section_path else ""
    sid = (section_id or "").strip()
    return bool(TOC_RX.match(leaf) or TOC_RX.match(sid))


def should_drop_chunk(chunk: Chunk) -> bool:
    # 1) Drop Table of Contents chunks
    if DROP_TABLE_OF_CONTENTS:
        if any(TOC_RX.match((p or "").strip()) for p in chunk.section_paths):
            return True
        if any(TOC_RX.match((s or "").strip()) for s in chunk.section_ids):
            return True
        prefix_title = (chunk.prefix or "").replace("Section:", "").strip()
        if prefix_title and TOC_RX.match(prefix_title):
            return True

    # 2) Drop tiny caption-like chunks (e.g., "Online Table 1 ...") that have no content
    if DROP_TINY_CAPTION_CHUNKS and chunk.token_len_payload < MIN_TINY_DROP_TOKENS:
        body = (chunk.body or "").strip()
        if TINY_CAPTION_RX.match(body):
            return True

    return False


# Candidate section segments

@dataclass
class SectionSegment:
    doc_id: str
    source_file: str
    section_path: str
    section_id: str
    page_start: Optional[int]
    page_end: Optional[int]
    blocks: List[Block]
    text: str
    token_len: int


def segment_blocks(tokenizer, blocks: List[Block], is_slides: bool) -> List[SectionSegment]:
    blocks = sorted(blocks, key=lambda b: (b.order_idx, b.block_id))

    def seg_key(b: Block) -> str:
        if is_slides:
            return (b.section_path or b.section_title or b.section_id or "slide_unknown").strip() or "slide_unknown"
        return (b.section_id or b.section_path or b.section_title or "no_section").strip() or "no_section"

    segments: List[SectionSegment] = []

    cur_key: Optional[str] = None
    cur_blocks: List[Block] = []
    cur_section_path: str = ""
    cur_section_id: str = ""
    cur_page_start: Optional[int] = None
    cur_page_end: Optional[int] = None

    def flush() -> None:
        nonlocal cur_key, cur_blocks, cur_section_path, cur_section_id, cur_page_start, cur_page_end
        if not cur_blocks:
            cur_key = None
            return

        # OPTIONAL: drop Table of Contents sections early (before units/chunks/overlap)
        if is_toc_section(cur_section_path, cur_section_id):
            cur_key = None
            cur_blocks = []
            cur_section_path = ""
            cur_section_id = ""
            cur_page_start = None
            cur_page_end = None
            return

        # Build segment text from non-heading blocks only
        parts: List[str] = []
        for b in cur_blocks:
            if b.block_type in HEADING_TYPES:
                continue
            txt = (b.text or "").strip()
            if txt:
                parts.append(txt)

        seg_text = "\n\n".join(parts).strip()
        seg_tokens = payload_len(tokenizer, seg_text) if seg_text else 0

        segments.append(
            SectionSegment(
                doc_id=cur_blocks[0].doc_id,
                source_file=cur_blocks[0].source_file,
                section_path=cur_section_path,
                section_id=cur_section_id,
                page_start=cur_page_start,
                page_end=cur_page_end,
                blocks=list(cur_blocks),
                text=seg_text,
                token_len=seg_tokens,
            )
        )

        cur_key = None
        cur_blocks = []
        cur_section_path = ""
        cur_section_id = ""
        cur_page_start = None
        cur_page_end = None

    for b in blocks:
        k = seg_key(b)
        if cur_key is None:
            cur_key = k
            cur_blocks = [b]
            cur_section_path = (b.section_path or b.section_title or "").strip() or k
            cur_section_id = (b.section_id or "").strip() or k.lower()
            cur_page_start = b.page_start
            cur_page_end = b.page_end
            continue

        if k != cur_key:
            flush()
            cur_key = k
            cur_blocks = [b]
            cur_section_path = (b.section_path or b.section_title or "").strip() or k
            cur_section_id = (b.section_id or "").strip() or k.lower()
            cur_page_start = b.page_start
            cur_page_end = b.page_end
        else:
            cur_blocks.append(b)
            if b.page_start is not None:
                cur_page_start = b.page_start if cur_page_start is None else min(cur_page_start, b.page_start)
            if b.page_end is not None:
                cur_page_end = b.page_end if cur_page_end is None else max(cur_page_end, b.page_end)

    flush()
    return segments


# Coalesce tiny segments forward into Units

@dataclass
class Unit:
    doc_id: str
    source_file: str
    unit_id: str
    page_start: Optional[int]
    page_end: Optional[int]
    section_paths: List[str]
    section_ids: List[str]
    blocks: List[Block]
    text: str
    token_len: int


def coalesce_segments(tokenizer, segs: List[SectionSegment]) -> List[Unit]:
    units: List[Unit] = []

    buf_paths: List[str] = []
    buf_ids: List[str] = []
    buf_blocks: List[Block] = []
    buf_text_parts: List[str] = []
    buf_page_start: Optional[int] = None
    buf_page_end: Optional[int] = None

    def buf_text() -> str:
        return "\n\n".join([t for t in buf_text_parts if t.strip()]).strip()

    def buf_tok() -> int:
        t = buf_text()
        return payload_len(tokenizer, t) if t else 0

    def flush_unit(unit_idx: int) -> None:
        nonlocal buf_paths, buf_ids, buf_blocks, buf_text_parts, buf_page_start, buf_page_end
        text = buf_text()
        tok = payload_len(tokenizer, text) if text else 0
        units.append(
            Unit(
                doc_id=segs[0].doc_id if segs else "",
                source_file=segs[0].source_file if segs else "",
                unit_id=f"u{unit_idx:04d}",
                page_start=buf_page_start,
                page_end=buf_page_end,
                section_paths=stable_uniq(buf_paths),
                section_ids=stable_uniq(buf_ids),
                blocks=list(buf_blocks),
                text=text,
                token_len=tok,
            )
        )
        buf_paths = []
        buf_ids = []
        buf_blocks = []
        buf_text_parts = []
        buf_page_start = None
        buf_page_end = None

    unit_idx = 0

    for i, seg in enumerate(segs):
        is_last = (i == len(segs) - 1)

        sp = (seg.section_path or "").strip()
        sid = (seg.section_id or "").strip()
        if sp:
            buf_paths.append(sp)
        if sid:
            buf_ids.append(sid)

        buf_blocks.extend(seg.blocks)

        # Only add non-empty segment text (title-only segments become provenance only)
        if seg.text.strip():
            buf_text_parts.append(seg.text.strip())

        if seg.page_start is not None:
            buf_page_start = seg.page_start if buf_page_start is None else min(buf_page_start, seg.page_start)
        if seg.page_end is not None:
            buf_page_end = seg.page_end if buf_page_end is None else max(buf_page_end, seg.page_end)

        tok = buf_tok()

        if is_last:
            flush_unit(unit_idx)
            unit_idx += 1
            break

        # If we still have no text, keep accumulating (title-only slides etc.)
        if tok == 0:
            continue

        # Merge forward until buffer is at least MIN_FINAL_TOKENS
        if tok < MIN_FINAL_TOKENS:
            continue

        flush_unit(unit_idx)
        unit_idx += 1

    # Optional merge-back of last tiny unit
    if MERGE_BACK_IF_LAST_TOO_SMALL and len(units) >= 2:
        last = units[-1]
        prev = units[-2]
        if 0 < last.token_len < MIN_FINAL_TOKENS:
            merged_text = (prev.text + "\n\n" + last.text).strip() if prev.text and last.text else (prev.text or last.text).strip()
            prev.page_start = prev.page_start if last.page_start is None else (last.page_start if prev.page_start is None else min(prev.page_start, last.page_start))
            prev.page_end = prev.page_end if last.page_end is None else (last.page_end if prev.page_end is None else max(prev.page_end, last.page_end))
            prev.section_paths = stable_uniq(prev.section_paths + last.section_paths)
            prev.section_ids = stable_uniq(prev.section_ids + last.section_ids)
            prev.blocks.extend(last.blocks)
            prev.text = merged_text
            prev.token_len = payload_len(tokenizer, merged_text) if merged_text else 0
            units.pop()

    return units


# Splitting oversized blocks safely

def split_atomic_block_if_needed(tokenizer, block: Block) -> List[str]:
    txt = (block.text or "").strip()
    if not txt:
        return []

    ids = token_ids_payload(tokenizer, txt)
    if len(ids) <= MAX_PAYLOAD_TOKENS:
        return [txt]

    # Oversized block: split safely if possible
    if block.block_type == "table":
        ts = block.table_struct or {}
        rows = ts.get("rows") if isinstance(ts, dict) else None
        header_rows = ts.get("header_rows") if isinstance(ts, dict) else None

        if isinstance(rows, list) and rows:
            hdr_idx = sorted([safe_int(x) for x in (header_rows or []) if safe_int(x) is not None])
            hdr_idx = [i for i in hdr_idx if 0 <= i < len(rows)]
            header = [rows[i] for i in hdr_idx] if hdr_idx else ([rows[0]] if rows else [])
            body = [rows[i] for i in range(len(rows)) if i not in set(hdr_idx)]

            def md_row(r: List[str]) -> str:
                r = [re.sub(r"\s+", " ", str(x or "").strip()) for x in r]
                return "| " + " | ".join(r) + " |"

            n_cols = max(len(r) for r in rows)
            header_line = md_row(header[-1] + [""] * (n_cols - len(header[-1]))) if header else ""
            sep_line = "| " + " | ".join(["---"] * n_cols) + " |"
            header_md = (header_line + "\n" + sep_line).strip() if header_line else ""

            pieces: List[str] = []
            cur_lines: List[str] = []

            def flush_piece():
                nonlocal cur_lines
                if cur_lines:
                    pieces.append("\n".join(cur_lines).strip())
                    cur_lines = []

            for r in body:
                row_md = md_row(r + [""] * (n_cols - len(r)))
                candidate_lines = ([header_md] if header_md else []) + cur_lines + [row_md]
                candidate_text = "\n".join([ln for ln in candidate_lines if ln.strip()]).strip()
                if payload_len(tokenizer, candidate_text) <= MAX_PAYLOAD_TOKENS:
                    if not cur_lines and header_md:
                        cur_lines.append(header_md)
                    cur_lines.append(row_md)
                else:
                    flush_piece()
                    start_lines = ([header_md] if header_md else []) + [row_md]
                    start_text = "\n".join([ln for ln in start_lines if ln.strip()]).strip()
                    if payload_len(tokenizer, start_text) <= MAX_PAYLOAD_TOKENS:
                        cur_lines = start_lines
                    else:
                        # Hard split row as last resort
                        hard = token_ids_payload(tokenizer, row_md)
                        for j in range(0, len(hard), MAX_PAYLOAD_TOKENS):
                            pieces.append(decode_payload(tokenizer, hard[j:j + MAX_PAYLOAD_TOKENS]).strip())

            flush_piece()
            return [p for p in pieces if p.strip()]

        # No structure -> hard token split
        pieces = []
        for i in range(0, len(ids), MAX_PAYLOAD_TOKENS):
            pieces.append(decode_payload(tokenizer, ids[i:i + MAX_PAYLOAD_TOKENS]).strip())
        return [p for p in pieces if p.strip()]

    if block.block_type == "list":
        lines = [ln.rstrip() for ln in txt.splitlines() if ln.strip()]
        items: List[str] = []
        cur_item: List[str] = []
        for ln in lines:
            if re.match(r"^\s*([•▪\-\u2022]|\d+\.)\s+\S+", ln):
                if cur_item:
                    items.append("\n".join(cur_item).strip())
                cur_item = [ln]
            else:
                cur_item.append(ln)
        if cur_item:
            items.append("\n".join(cur_item).strip())

        pieces: List[str] = []
        cur: List[str] = []

        def flush():
            nonlocal cur
            if cur:
                pieces.append("\n".join(cur).strip())
                cur = []

        for it in items:
            candidate = ("\n".join(cur + [it])).strip()
            if payload_len(tokenizer, candidate) <= MAX_PAYLOAD_TOKENS:
                cur.append(it)
            else:
                flush()
                if payload_len(tokenizer, it) <= MAX_PAYLOAD_TOKENS:
                    cur = [it]
                else:
                    # fallback: sentence split then token split
                    sents = split_paragraph_by_sentences(it)
                    if sents:
                        tmp: List[str] = []
                        for s in sents:
                            cand = (" ".join(tmp + [s])).strip()
                            if payload_len(tokenizer, cand) <= MAX_PAYLOAD_TOKENS:
                                tmp.append(s)
                            else:
                                if tmp:
                                    pieces.append(" ".join(tmp).strip())
                                    tmp = []
                                if payload_len(tokenizer, s) <= MAX_PAYLOAD_TOKENS:
                                    tmp = [s]
                                else:
                                    hard = token_ids_payload(tokenizer, s)
                                    for j in range(0, len(hard), MAX_PAYLOAD_TOKENS):
                                        pieces.append(decode_payload(tokenizer, hard[j:j + MAX_PAYLOAD_TOKENS]).strip())
                        if tmp:
                            pieces.append(" ".join(tmp).strip())
                    else:
                        hard = token_ids_payload(tokenizer, it)
                        for j in range(0, len(hard), MAX_PAYLOAD_TOKENS):
                            pieces.append(decode_payload(tokenizer, hard[j:j + MAX_PAYLOAD_TOKENS]).strip())

        flush()
        return [p for p in pieces if p.strip()]

    # Paragraph/other: split by sentences
    sents = split_paragraph_by_sentences(txt)
    if not sents:
        return [
            decode_payload(tokenizer, ids[i:i + MAX_PAYLOAD_TOKENS]).strip()
            for i in range(0, len(ids), MAX_PAYLOAD_TOKENS)
            if decode_payload(tokenizer, ids[i:i + MAX_PAYLOAD_TOKENS]).strip()
        ]

    pieces: List[str] = []
    cur: List[str] = []
    for s in sents:
        candidate = (" ".join(cur + [s])).strip()
        if payload_len(tokenizer, candidate) <= MAX_PAYLOAD_TOKENS:
            cur.append(s)
        else:
            if cur:
                pieces.append(" ".join(cur).strip())
                cur = []
            if payload_len(tokenizer, s) <= MAX_PAYLOAD_TOKENS:
                cur = [s]
            else:
                hard = token_ids_payload(tokenizer, s)
                for j in range(0, len(hard), MAX_PAYLOAD_TOKENS):
                    pieces.append(decode_payload(tokenizer, hard[j:j + MAX_PAYLOAD_TOKENS]).strip())
    if cur:
        pieces.append(" ".join(cur).strip())

    return [p for p in pieces if p.strip()]


# Chunk packing + overlap inside a Unit

def pick_prefix_for_unit(unit: Unit) -> str:
    if not ADD_SECTION_PREFIX_LINE:
        return ""
    cand = ""
    if unit.section_paths:
        cand = section_leaf(unit.section_paths[-1])
    if not cand and unit.section_ids:
        cand = unit.section_ids[-1]
    cand = clean_title_for_prefix(cand)
    if not cand:
        return ""
    return SECTION_PREFIX_STYLE.format(title=cand).strip()


def pack_unit_into_chunks(
    tokenizer,
    unit: Unit,
    doc_chunk_counter: int,
    is_slides: bool,
) -> Tuple[List[Chunk], int]:
    overlap_req = SLIDES_OVERLAP_TOKENS if is_slides else DEFAULT_OVERLAP_TOKENS

    # Expand blocks into safe "pieces" (strings), preserving block boundaries.
    pieces: List[Tuple[str, str, str]] = []  # (piece_text, block_id, block_type)
    for b in unit.blocks:
        if b.block_type in HEADING_TYPES:
            continue
        for p in split_atomic_block_if_needed(tokenizer, b):
            if p.strip():
                pieces.append((p.strip(), b.block_id, b.block_type))

    if not pieces:
        return [], doc_chunk_counter

    prefix = pick_prefix_for_unit(unit)

    # Pack into raw chunks (BODY ONLY), then attach prefix and overlap.
    raw_chunks: List[Dict[str, Any]] = []
    cur_parts: List[str] = []
    cur_block_ids: List[str] = []
    cur_block_types: List[str] = []

    def cur_body() -> str:
        return "\n\n".join(cur_parts).strip()

    def flush_raw():
        nonlocal cur_parts, cur_block_ids, cur_block_types
        body = cur_body()
        if body:
            raw_chunks.append(
                dict(
                    body=body,
                    block_ids=list(cur_block_ids),
                    block_types=list(cur_block_types),
                    page_start=unit.page_start,
                    page_end=unit.page_end,
                )
            )
        cur_parts = []
        cur_block_ids = []
        cur_block_types = []

    for (ptext, bid, btype) in pieces:
        candidate_parts = cur_parts + [ptext]
        cand_body = "\n\n".join(candidate_parts).strip()
        cand_len = payload_len(tokenizer, cand_body)

        if cand_len <= TARGET_PAYLOAD_TOKENS:
            cur_parts.append(ptext)
            cur_block_ids.append(bid)
            cur_block_types.append(btype)
            continue

        if cand_len <= MAX_PAYLOAD_TOKENS:
            cur_parts.append(ptext)
            cur_block_ids.append(bid)
            cur_block_types.append(btype)
            if cand_len >= TARGET_PAYLOAD_TOKENS:
                flush_raw()
            continue

        # Would exceed cap: flush current and start new
        flush_raw()
        cur_parts.append(ptext)
        cur_block_ids.append(bid)
        cur_block_types.append(btype)

        # Safety: if a single piece somehow exceeds cap, hard split
        body2 = cur_body()
        if payload_len(tokenizer, body2) > MAX_PAYLOAD_TOKENS:
            ids = token_ids_payload(tokenizer, body2)
            for i in range(0, len(ids), MAX_PAYLOAD_TOKENS):
                part = decode_payload(tokenizer, ids[i:i + MAX_PAYLOAD_TOKENS]).strip()
                if part:
                    raw_chunks.append(
                        dict(
                            body=part,
                            block_ids=list(cur_block_ids),
                            block_types=list(cur_block_types),
                            page_start=unit.page_start,
                            page_end=unit.page_end,
                        )
                    )
            cur_parts = []
            cur_block_ids = []
            cur_block_types = []

    flush_raw()

    # Apply overlap within this unit, computed from BODY ONLY.
    out_chunks: List[Chunk] = []
    prev_body_ids: Optional[List[int]] = None
    prev_chunk_id: Optional[str] = None

    for raw in raw_chunks:
        body = raw["body"].strip()
        body_ids = token_ids_payload(tokenizer, body)

        overlap_used = 0
        overlap_from = None
        final_body = body

        if prev_body_ids is not None and overlap_req > 0:
            ov_ids = prev_body_ids[-overlap_req:] if overlap_req <= len(prev_body_ids) else prev_body_ids[:]

            # ensure overlap + current body <= MAX_PAYLOAD_TOKENS
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
                        final_body = (ov_text + "\n\n" + body).strip()
                        overlap_used = len(ov_ids2)
                        overlap_from = prev_chunk_id

        # Compose final text with prefix (prefix NOT part of overlap calc)
        final_text = (prefix + "\n\n" + final_body).strip() if prefix else final_body

        # Hard cap safety including prefix
        final_ids = token_ids_payload(tokenizer, final_text)
        if len(final_ids) > MAX_PAYLOAD_TOKENS:
            # If prefix pushes over, drop prefix first
            if prefix:
                final_text = final_body
                prefix_out = ""
            else:
                prefix_out = prefix
            final_ids = token_ids_payload(tokenizer, final_text)

            # If still too long, hard trim tokens
            if len(final_ids) > MAX_PAYLOAD_TOKENS:
                final_text = decode_payload(tokenizer, final_ids[:MAX_PAYLOAD_TOKENS]).strip()
                final_ids = token_ids_payload(tokenizer, final_text)
        else:
            prefix_out = prefix

        chunk_id = f"{unit.doc_id}_chunk{doc_chunk_counter:06d}"
        doc_chunk_counter += 1

        out_chunks.append(
            Chunk(
                chunk_id=chunk_id,
                doc_id=unit.doc_id,
                source_file=unit.source_file,
                chunk_idx=-1,  # set later per doc
                page_start=raw["page_start"],
                page_end=raw["page_end"],
                section_paths=stable_uniq(unit.section_paths),
                section_ids=stable_uniq(unit.section_ids),
                block_ids=raw["block_ids"],
                block_types=raw["block_types"],
                overlap_tokens_requested=overlap_req,
                overlap_tokens_used=overlap_used,
                overlap_from_chunk_id=overlap_from,
                prefix=prefix_out,
                body=final_body,
                text=final_text,
                token_len_payload=len(final_ids),
                text_hash=sha1_text(final_text),
            )
        )

        # Base overlap on the NON-OVERLAPPED body (cleaner, avoids overlap growth)
        prev_body_ids = token_ids_payload(tokenizer, body)
        prev_chunk_id = chunk_id

    return out_chunks, doc_chunk_counter


# IO + Main

def load_blocks_file(path: Path) -> List[Block]:
    rows = read_jsonl(path)
    blocks: List[Block] = []
    for r in rows:
        blocks.append(
            Block(
                doc_id=str(r.get("doc_id", "") or ""),
                source_file=str(r.get("source_file", "") or ""),
                block_id=str(r.get("block_id", "") or ""),
                order_idx=int(r.get("order_idx", 0) or 0),
                block_type=str(r.get("block_type", "") or ""),
                page_start=safe_int(r.get("page_start")),
                page_end=safe_int(r.get("page_end")),
                section_path=r.get("section_path"),
                text=str(r.get("text", "") or ""),
                char_len=int(r.get("char_len", 0) or 0),
                text_hash=str(r.get("text_hash", "") or ""),
                section_title=r.get("section_title"),
                section_id=r.get("section_id"),
                table_struct=r.get("table_struct"),
            )
        )
    blocks.sort(key=lambda b: (b.order_idx, b.block_id))
    return blocks


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    in_files = sorted(PROCESSED_DIR.glob(IN_GLOB))
    if not in_files:
        raise SystemExit(f"No input files found: {PROCESSED_DIR / IN_GLOB}")

    tokenizer = AutoTokenizer.from_pretrained(
        CANONICAL_TOKENIZER_NAME,
        trust_remote_code=CANONICAL_TOKENIZER_TRUST_REMOTE_CODE,
        use_fast=True,
    )

    all_chunks: List[Dict[str, Any]] = []
    doc_reports: Dict[str, Any] = {}
    global_token_lens: List[int] = []
    global_block_type_counts = Counter()

    for blocks_path in in_files:
        blocks = load_blocks_file(blocks_path)
        if not blocks:
            continue

        doc_id = blocks[0].doc_id
        source_file = blocks[0].source_file
        is_slides = looks_like_slides_filename(source_file)

        # 1) candidate segments by section boundary (TOC dropped here)
        segs = segment_blocks(tokenizer, blocks, is_slides=is_slides)

        # 2) coalesce tiny segments forward into units
        units = coalesce_segments(tokenizer, segs)

        # 3) chunk within each unit + overlap within each unit
        doc_chunk_counter = 0
        doc_chunks: List[Chunk] = []
        for unit in units:
            chs, doc_chunk_counter = pack_unit_into_chunks(
                tokenizer=tokenizer,
                unit=unit,
                doc_chunk_counter=doc_chunk_counter,
                is_slides=is_slides,
            )
            doc_chunks.extend(chs)

        # 4) drop tiny caption-only chunks (and any remaining TOC-like chunks), then renumber chunk_idx
        kept_doc_chunks: List[Chunk] = []
        dropped = 0
        for ch in doc_chunks:
            if should_drop_chunk(ch):
                dropped += 1
                continue
            kept_doc_chunks.append(ch)

        for i, ch in enumerate(kept_doc_chunks):
            ch.chunk_idx = i
            all_chunks.append(asdict(ch))
            global_token_lens.append(ch.token_len_payload)
            global_block_type_counts.update(ch.block_types)

        token_lens = [c.token_len_payload for c in kept_doc_chunks]
        doc_reports[doc_id] = {
            "doc_id": doc_id,
            "source_file": source_file,
            "is_slides": bool(is_slides),
            "num_blocks": len(blocks),
            "num_segments": len(segs),
            "num_units": len(units),
            "num_chunks": len(kept_doc_chunks),
            "num_chunks_dropped": dropped,
            "chunk_token_len_min": min(token_lens) if token_lens else None,
            "chunk_token_len_med": sorted(token_lens)[len(token_lens) // 2] if token_lens else None,
            "chunk_token_len_max": max(token_lens) if token_lens else None,
            "block_type_counts": dict(Counter(b.block_type for b in blocks)),
            "chunk_block_type_counts": dict(Counter(t for c in kept_doc_chunks for t in c.block_types)),
            "prefix_enabled": bool(ADD_SECTION_PREFIX_LINE),
        }

    # Global report
    token_lens_sorted = sorted(global_token_lens)
    report: Dict[str, Any] = {
        "config": {
            "tokenizer": CANONICAL_TOKENIZER_NAME,
            "model_max_len_including_special": MODEL_MAX_LEN_INCLUDING_SPECIAL,
            "max_payload_tokens": MAX_PAYLOAD_TOKENS,
            "target_payload_tokens": TARGET_PAYLOAD_TOKENS,
            "min_section_tokens": MIN_SECTION_TOKENS,
            "min_final_tokens": MIN_FINAL_TOKENS,
            "default_overlap_tokens": DEFAULT_OVERLAP_TOKENS,
            "slides_overlap_tokens": SLIDES_OVERLAP_TOKENS,
            "add_section_prefix_line": ADD_SECTION_PREFIX_LINE,
            "section_prefix_style": SECTION_PREFIX_STYLE if ADD_SECTION_PREFIX_LINE else None,
            "atomic_block_types": sorted(list(ATOMIC_BLOCK_TYPES)),
            "heading_types_excluded_from_body": sorted(list(HEADING_TYPES)),
            "overlap_is_body_only": True,
            "overlap_sentence_snap": True,
            "drop_table_of_contents": DROP_TABLE_OF_CONTENTS,
            "drop_tiny_caption_chunks": DROP_TINY_CAPTION_CHUNKS,
            "min_tiny_drop_tokens": MIN_TINY_DROP_TOKENS if DROP_TINY_CAPTION_CHUNKS else None,
        },
        "global": {
            "num_docs": len(doc_reports),
            "num_chunks": len(all_chunks),
            "chunk_token_len_min": min(global_token_lens) if global_token_lens else None,
            "chunk_token_len_p50": token_lens_sorted[len(token_lens_sorted) // 2] if token_lens_sorted else None,
            "chunk_token_len_p90": token_lens_sorted[int(0.9 * (len(token_lens_sorted) - 1))] if token_lens_sorted else None,
            "chunk_token_len_max": max(global_token_lens) if global_token_lens else None,
            "chunk_block_type_counts": dict(global_block_type_counts),
        },
        "per_doc": doc_reports,
    }

    write_jsonl(OUT_CHUNKS, all_chunks)
    write_json(OUT_REPORT, report)

    print(f"Wrote chunks:  {OUT_CHUNKS}  ({len(all_chunks)} chunks)")
    print(f"Wrote report:  {OUT_REPORT}")


if __name__ == "__main__":
    main()
