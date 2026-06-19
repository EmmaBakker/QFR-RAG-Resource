#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from collections import Counter, defaultdict

from docling.document_converter import DocumentConverter


# Config
PDF_DIR = Path("data/pdfs")
OUT_DIR = Path("data/processed")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Minimum text thresholds (type-aware)
MIN_TEXT_CHARS_DEFAULT = 30
MIN_TEXT_CHARS_LIST_ITEM = 1
MIN_TEXT_CHARS_HEADING = 3

KEEP_FOOTNOTES = False
KEEP_CAPTIONS = False

# Figure descriptions
KEEP_FIGURE_DESCRIPTIONS = True
FIGURE_LINE_REGEX = r"^\s*(figure|fig\.?)\s*\d+[\s:.\-]"

# Bullet detection (for paragraphs that are actually bullet lists)
BULLET_LINE_REGEX = r"^\s*([•▪\-\u2022]|\d+\.)\s+\S+"

# Tables
STORE_COMPACT_TABLE_STRUCT = True
MERGE_TABLE_CAPTIONS_FROM_TABLE_NODE = True
MERGE_TABLE_FOOTNOTES_FROM_TABLE_NODE = True

# Optional cleanup for stray PDF noise lines
DROP_LINE_REGEXES = [
    r"^\s*page\s+\d+\s*(of\s+\d+)?\s*$",
    r"^\s*\d+\s*$",
]

# Reproducible manual removals (optional)
ENABLE_PATCHES = True

# Incremental processing
SKIP_IF_UNCHANGED = True

# References filtering
DROP_REFERENCES_SECTION = True
REFERENCES_HEADING_REGEX = r"^\s*(references|bibliography|reference list|works cited)\s*$"

# Slide-PDF fix: harvest "orphan" text nodes not linked into body.children
ENABLE_ORPHAN_TEXT_HARVEST = True
LOW_TEXT_PAGE_CHAR_THRESHOLD = 80


# Data model
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

    # Put non-default fields BEFORE default fields (or give them defaults).
    text: str
    char_len: int
    text_hash: str

    # Stable section fields for chunking (defaults OK at end)
    section_title: Optional[str] = None
    section_id: Optional[str] = None

    table_struct: Optional[Dict[str, Any]] = None


# Helpers
def clean_text(s: str) -> str:
    if not s:
        return ""
    s = s.replace("\u00ad", "")  # soft hyphen
    s = re.sub(r"(\w)-\n(\w)", r"\1\2", s)  # dehyphenate across line breaks
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    s = s.strip()

    kept_lines: List[str] = []
    for ln in s.splitlines():
        lns = ln.strip()
        if any(re.match(rx, lns, flags=re.IGNORECASE) for rx in DROP_LINE_REGEXES):
            continue
        kept_lines.append(ln)
    return "\n".join(kept_lines).strip()


def sha1_text(s: str) -> str:
    return hashlib.sha1((s or "").encode("utf-8")).hexdigest()


def safe_int(x: Any) -> Optional[int]:
    try:
        return int(x)
    except Exception:
        return None


def pdf_fingerprint(pdf_path: Path) -> Dict[str, Any]:
    st = pdf_path.stat()
    return {"pdf_size_bytes": int(st.st_size), "pdf_mtime": int(st.st_mtime)}


def stable_doc_id(pdf_path: Path) -> str:
    """Stable doc_id that doesn't change when new PDFs are added."""
    try:
        rel = pdf_path.resolve().relative_to(PDF_DIR.resolve())
        rel_s = rel.as_posix()
    except Exception:
        rel_s = pdf_path.as_posix()
    h = hashlib.sha1(rel_s.encode("utf-8")).hexdigest()[:12]
    return f"doc{h}"


def looks_like_figure_line(text: str) -> bool:
    if not text:
        return False
    for ln in text.splitlines():
        if re.match(FIGURE_LINE_REGEX, ln.strip(), flags=re.IGNORECASE):
            return True
    return False


def looks_like_bullet_line(text: str) -> bool:
    if not text:
        return False
    for ln in text.splitlines():
        if re.match(BULLET_LINE_REGEX, ln.strip()):
            return True
    return False


# Docling $ref resolver + typing
def resolve_ref(doc: Dict[str, Any], ref: str) -> Any:
    """Resolve Docling JSON pointer-like references, e.g. '#/texts/17'."""
    if not isinstance(ref, str) or not ref.startswith("#/"):
        return None
    parts = ref[2:].split("/")
    cur: Any = doc
    for p in parts:
        if isinstance(cur, dict) and p in cur:
            cur = cur[p]
        elif isinstance(cur, list) and p.isdigit():
            idx = int(p)
            if 0 <= idx < len(cur):
                cur = cur[idx]
            else:
                return None
        else:
            return None
    return cur


def node_label(node: Dict[str, Any]) -> str:
    return str(node.get("label") or node.get("type") or "").lower().strip()


def is_table_ref(ref: Optional[str], label: str) -> bool:
    return (ref or "").startswith("#/tables/") or ("table" in label)


def is_figure_ref(ref: Optional[str], label: str) -> bool:
    r = (ref or "").lower()
    return (
        r.startswith("#/figures/")
        or r.startswith("#/images/")
        or r.startswith("#/pictures/")
        or ("figure" in label)
        or ("image" in label)
        or ("picture" in label)
    )


# Provenance (page range)
def extract_page_range(node: Dict[str, Any]) -> Tuple[Optional[int], Optional[int]]:
    page_nums: List[int] = []
    candidates: List[Any] = []

    if isinstance(node.get("provenance"), list):
        candidates.extend(node["provenance"])
    if isinstance(node.get("prov"), list):
        candidates.extend(node["prov"])

    for item in candidates:
        if not isinstance(item, dict):
            continue
        for key in ("page", "page_no", "page_num", "page_number"):
            if key in item:
                v = safe_int(item.get(key))
                if v is not None:
                    page_nums.append(v)

        loc = item.get("location") or item.get("loc")
        if isinstance(loc, dict):
            for key in ("page", "page_no", "page_num", "page_number"):
                if key in loc:
                    v = safe_int(loc.get(key))
                    if v is not None:
                        page_nums.append(v)

    for key in ("page", "page_no", "page_num", "page_number"):
        if key in node:
            v = safe_int(node.get(key))
            if v is not None:
                page_nums.append(v)

    spans = node.get("spans") or node.get("span")
    if isinstance(spans, list):
        for sp in spans:
            if not isinstance(sp, dict):
                continue
            for key in ("page", "page_no", "page_num", "page_number"):
                if key in sp:
                    v = safe_int(sp.get(key))
                    if v is not None:
                        page_nums.append(v)

    if not page_nums:
        return None, None
    return min(page_nums), max(page_nums)


# Node filters + text extraction
def is_furniture(label: str) -> bool:
    return label in {"header", "footer", "page_header", "page_footer", "furniture"}


def extract_text_from_node(node: Dict[str, Any]) -> str:
    for key in ("text", "content", "value", "orig"):
        v = node.get(key)
        if isinstance(v, str) and v.strip():
            return v
    data = node.get("data")
    if isinstance(data, dict):
        for key in ("text", "content", "value"):
            v = data.get(key)
            if isinstance(v, str) and v.strip():
                return v
    return ""


def should_keep_node(node: Dict[str, Any], ref: Optional[str]) -> bool:
    label = node_label(node)

    if is_furniture(label):
        return False

    if label == "footnote" and not KEEP_FOOTNOTES:
        return False

    # Keep caption nodes that look like figure legends even if KEEP_CAPTIONS=False
    if label == "caption" and not KEEP_CAPTIONS:
        if KEEP_FIGURE_DESCRIPTIONS:
            t = clean_text(extract_text_from_node(node))
            if looks_like_figure_line(t):
                return True
        return False

    return True


def block_type_from(label: str, ref: Optional[str]) -> str:
    if is_table_ref(ref, label):
        return "table"
    if label in {"section_header", "heading", "title"}:
        return "heading"
    if label in {"list_item"}:
        return "list_item"
    if label in {"text", "paragraph"}:
        return "paragraph"
    if label == "list":
        return "list"
    if label == "caption":
        return "caption"
    if label == "footnote":
        return "footnote"
    return label or "unknown"


def min_chars_for_type(btype: str) -> int:
    if btype == "list_item":
        return MIN_TEXT_CHARS_LIST_ITEM
    if btype == "heading":
        return MIN_TEXT_CHARS_HEADING
    return MIN_TEXT_CHARS_DEFAULT


# Section path logic
def split_section_path(section_path: Optional[str]) -> List[str]:
    if not section_path:
        return []
    return [p.strip() for p in section_path.split(">") if p.strip()]

def section_title_from_path(section_path: Optional[str]) -> Optional[str]:
    parts = split_section_path(section_path)
    return parts[-1] if parts else None

def section_id_from_title(title: Optional[str]) -> Optional[str]:
    if not title:
        return None
    t = re.sub(r"^\s*\d+\.\s*", "", title).strip()   # drop "1. "
    t = re.sub(r"\s+", " ", t).strip().lower()
    return t or None

def looks_like_slides_filename(name: str) -> bool:
    n = (name or "").lower()
    return ("slide" in n) or ("slides" in n) or ("ppt" in n) or ("presentation" in n)

def compute_section_fields(section_path: Optional[str], source_file: str, page_start: Optional[int]) -> Tuple[Optional[str], Optional[str]]:
    """
    Returns (section_title, section_id)
    """
    title = section_title_from_path(section_path)
    sec_id = section_id_from_title(title)

    # Slide fallback: if section info is missing/unstable, pin to page
    if looks_like_slides_filename(source_file):
        if page_start is not None:
            # keep a real title if we have one, but ensure a stable ID
            sec_id = sec_id or f"slide_{int(page_start)}"
            title = title or f"Slide {int(page_start)}"

    return title, sec_id

def section_path_from_stack(stack: List[Tuple[int, str]]) -> Optional[str]:
    if not stack:
        return None
    joined = " > ".join(title for _, title in stack if title.strip())
    return joined or None


def update_section_stack(stack: List[Tuple[int, str]], level: int, title: str) -> None:
    level = max(1, int(level))
    while stack and stack[-1][0] >= level:
        stack.pop()
    stack.append((level, title))


# Table serialization (header-aware)
def _table_cells(table_node: Dict[str, Any]) -> List[Dict[str, Any]]:
    data = table_node.get("data")
    if not isinstance(data, dict):
        return []
    cells = data.get("table_cells")
    if isinstance(cells, list):
        return [c for c in cells if isinstance(c, dict)]
    return []


def table_to_grid_strings(table_node: Dict[str, Any]) -> Tuple[List[List[str]], Dict[int, List[int]]]:
    data = table_node.get("data")
    if not isinstance(data, dict):
        return [], {}

    n_rows = safe_int(data.get("num_rows")) or 0
    n_cols = safe_int(data.get("num_cols")) or 0

    if n_rows <= 0 or n_cols <= 0:
        grid = data.get("grid")
        if isinstance(grid, list) and grid:
            rows: List[List[str]] = []
            maxc = 0
            for r in grid:
                if not isinstance(r, list):
                    continue
                out = []
                for cell in r:
                    if isinstance(cell, dict):
                        txt = (cell.get("text") or cell.get("content") or "").strip()
                    else:
                        txt = str(cell).strip()
                    out.append(re.sub(r"\s+", " ", txt))
                rows.append(out)
                maxc = max(maxc, len(out))
            rows = [r + [""] * (maxc - len(r)) for r in rows]
            return rows, {}
        return [], {}

    rows = [["" for _ in range(n_cols)] for _ in range(n_rows)]
    header_map: Dict[int, List[int]] = defaultdict(list)

    for c in _table_cells(table_node):
        r0 = safe_int(c.get("start_row_offset_idx"))
        r1 = safe_int(c.get("end_row_offset_idx"))
        c0 = safe_int(c.get("start_col_offset_idx"))
        c1 = safe_int(c.get("end_col_offset_idx"))
        txt = re.sub(r"\s+", " ", str(c.get("text") or "").strip())

        if r0 is None or r1 is None or c0 is None or c1 is None:
            continue
        if not (0 <= r0 < n_rows and 0 < r1 <= n_rows and 0 <= c0 < n_cols and 0 < c1 <= n_cols):
            continue

        if txt:
            rows[r0][c0] = txt

        if c.get("column_header") is True:
            header_map[r0].append(c0)

    header_map = {r: sorted(set(cols)) for r, cols in header_map.items()}
    return rows, header_map


def rows_to_markdown_with_headers(rows: List[List[str]], header_map: Dict[int, List[int]]) -> str:
    if not rows:
        return ""

    n_cols = max(len(r) for r in rows)
    rows = [r + [""] * (n_cols - len(r)) for r in rows]

    header_rows = sorted(header_map.keys()) if header_map else [0]

    merged_header = [""] * n_cols
    for hr in header_rows:
        if not (0 <= hr < len(rows)):
            continue
        for j in range(n_cols):
            piece = (rows[hr][j] or "").strip()
            if not piece:
                continue
            if merged_header[j]:
                if piece.lower() not in merged_header[j].lower():
                    merged_header[j] = f"{merged_header[j]} / {piece}"
            else:
                merged_header[j] = piece

    header_set = set(header_rows)
    body_rows = [rows[i] for i in range(len(rows)) if i not in header_set]

    def md_row(r: List[str]) -> str:
        r = [re.sub(r"\s+", " ", (x or "").strip()) for x in r]
        return "| " + " | ".join(r) + " |"

    out = [md_row(merged_header), md_row(["---"] * n_cols)]
    out += [md_row(r) for r in body_rows if any(x.strip() for x in r)]
    return "\n".join(out).strip()


def table_to_markdown(table_node: Dict[str, Any]) -> str:
    rows, header_map = table_to_grid_strings(table_node)
    return rows_to_markdown_with_headers(rows, header_map)


def extract_table_attached_texts(doc_dict: Dict[str, Any], table_node: Dict[str, Any]) -> Tuple[str, str]:
    caption_parts: List[str] = []
    note_parts: List[str] = []

    caps = table_node.get("captions")
    if isinstance(caps, list):
        for c in caps:
            if isinstance(c, dict) and "$ref" in c:
                n = resolve_ref(doc_dict, c["$ref"])
                if isinstance(n, dict):
                    t = clean_text(extract_text_from_node(n))
                    if t:
                        caption_parts.append(t)

    fns = table_node.get("footnotes")
    if isinstance(fns, list):
        for f in fns:
            if isinstance(f, dict) and "$ref" in f:
                n = resolve_ref(doc_dict, f["$ref"])
                if isinstance(n, dict):
                    t = clean_text(extract_text_from_node(n))
                    if t:
                        note_parts.append(t)

    return clean_text("\n".join(caption_parts)), clean_text("\n".join(note_parts))


# Figure captions (attached)
def _extract_attached_texts_generic(doc_dict: Dict[str, Any], node: Dict[str, Any], keys: List[str]) -> List[str]:
    out: List[str] = []

    for key in keys:
        v = node.get(key)
        if isinstance(v, str):
            t = clean_text(v)
            if t:
                out.append(t)
        elif isinstance(v, dict):
            if "$ref" in v:
                ref_obj = resolve_ref(doc_dict, v["$ref"])
                if isinstance(ref_obj, dict):
                    t = clean_text(extract_text_from_node(ref_obj))
                    if t:
                        out.append(t)
            else:
                t = clean_text(extract_text_from_node(v))
                if t:
                    out.append(t)
        elif isinstance(v, list):
            for item in v:
                if isinstance(item, dict) and "$ref" in item:
                    ref_obj = resolve_ref(doc_dict, item["$ref"])
                    if isinstance(ref_obj, dict):
                        t = clean_text(extract_text_from_node(ref_obj))
                        if t:
                            out.append(t)
                elif isinstance(item, dict):
                    t = clean_text(extract_text_from_node(item))
                    if t:
                        out.append(t)
                elif isinstance(item, str):
                    t = clean_text(item)
                    if t:
                        out.append(t)

    seen = set()
    uniq: List[str] = []
    for t in out:
        k = t.strip()
        if not k or k in seen:
            continue
        seen.add(k)
        uniq.append(t)
    return uniq


def extract_figure_caption_from_node(doc_dict: Dict[str, Any], fig_node: Dict[str, Any]) -> str:
    parts = _extract_attached_texts_generic(
        doc_dict,
        fig_node,
        keys=["captions", "caption", "description", "descriptions", "title", "titles", "alt_text", "alt"],
    )
    data = fig_node.get("data")
    if isinstance(data, dict):
        parts += _extract_attached_texts_generic(
            doc_dict,
            data,
            keys=["caption", "captions", "description", "descriptions", "text", "content", "title"],
        )
    return clean_text("\n".join(parts))


# LIST extraction (node-preserving)
def iter_list_items(list_node: Dict[str, Any]) -> List[Any]:
    candidates: List[Any] = []

    for key in ("children", "items"):
        v = list_node.get(key)
        if isinstance(v, list):
            candidates.extend(v)

    data = list_node.get("data")
    if isinstance(data, dict):
        for key in ("items", "children"):
            v = data.get(key)
            if isinstance(v, list):
                candidates.extend(v)

    return candidates


def list_items_to_nodes(doc_dict: Dict[str, Any], list_node: Dict[str, Any]) -> List[Dict[str, Any]]:
    raw_items = iter_list_items(list_node)
    out: List[Dict[str, Any]] = []

    def handle_item(it: Any) -> None:
        if it is None:
            return

        if isinstance(it, dict) and "$ref" in it:
            n = resolve_ref(doc_dict, it["$ref"])
            handle_item(n)
            return

        if isinstance(it, dict) and node_label(it) == "list":
            for sub in iter_list_items(it):
                handle_item(sub)
            return

        if isinstance(it, dict):
            out.append(it)
            ch = it.get("children")
            if isinstance(ch, list):
                for sub in ch:
                    handle_item(sub)

    for it in raw_items:
        handle_item(it)

    return [n for n in out if isinstance(n, dict)]


# Patches (optional)
def load_patches(patch_path: Path) -> Dict[str, Any]:
    if not patch_path.exists():
        return {}
    try:
        return json.loads(patch_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def apply_patches(blocks: List[Block], patches: Dict[str, Any]) -> List[Block]:
    if not patches:
        return blocks

    drop_ids = set(patches.get("drop_block_ids", []) or [])
    regex_rules = patches.get("drop_regex", []) or []

    out: List[Block] = []
    for b in blocks:
        if b.block_id in drop_ids:
            continue

        drop = False
        for rule in regex_rules:
            if not isinstance(rule, dict):
                continue
            pat = rule.get("pattern")
            field = rule.get("field", "text")
            flags = rule.get("flags", "")
            if not isinstance(pat, str) or not pat:
                continue

            val = getattr(b, field, "") or ""
            re_flags = re.IGNORECASE if "i" in flags.lower() else 0
            if re.search(pat, str(val), flags=re_flags):
                drop = True
                break

        if not drop:
            out.append(b)

    return out


# Post-processing: merge list_item blocks into one list block
def merge_consecutive_list_items(blocks: List[Block]) -> List[Block]:
    out: List[Block] = []
    buf: List[Block] = []

    def flush() -> None:
        nonlocal buf
        if not buf:
            return

        merged_lines: List[str] = []
        for b in buf:
            t = (b.text or "").strip()
            if not t:
                continue
            if re.match(r"^[•▪\-\u2022]\s+", t):
                merged_lines.append(t)
            else:
                merged_lines.append(f"• {t}")

        merged_text = clean_text("\n".join(merged_lines))
        first = buf[0]

        p_starts = [b.page_start for b in buf if b.page_start is not None]
        p_ends = [b.page_end for b in buf if b.page_end is not None]
        page_start = min(p_starts) if p_starts else None
        page_end = max(p_ends) if p_ends else None

        out.append(
            Block(
                doc_id=first.doc_id,
                source_file=first.source_file,
                block_id=first.block_id,
                order_idx=first.order_idx,
                block_type="list",
                page_start=page_start,
                page_end=page_end,
                section_path=first.section_path,
                section_title=first.section_title,
                section_id=first.section_id,
                text=merged_text,
                char_len=len(merged_text),
                text_hash=sha1_text(merged_text),
            )
        )
        buf = []

    for b in blocks:
        if b.block_type == "list_item":
            buf.append(b)
            continue
        flush()
        out.append(b)

    flush()
    return out


# References handling (fix for orphan harvest reintroducing references)
def should_enter_references(title: str) -> bool:
    if not DROP_REFERENCES_SECTION:
        return False
    return bool(re.match(REFERENCES_HEADING_REGEX, (title or "").strip(), flags=re.IGNORECASE))


def drop_references_blocks(blocks: List[Block]) -> List[Block]:
    """
    Post-filter that removes reference/bibliography content even if it came from orphan harvesting.
    Policy:
      - If we find any "References" heading on some page P, drop all blocks with page_start >= P (including the heading).
      - Also drop anything whose section_path contains a references-like segment.
    """
    if not DROP_REFERENCES_SECTION:
        return blocks

    refs_rx = re.compile(REFERENCES_HEADING_REGEX, re.IGNORECASE)

    # 1) Drop by section_path segments if present
    tmp: List[Block] = []
    for b in blocks:
        if b.section_path:
            parts = [p.strip() for p in b.section_path.split(">")]
            if any(refs_rx.match(p) for p in parts):
                continue
        tmp.append(b)
    blocks = tmp

    # 2) Page-based fallback: if a page has a "References" heading, drop that page and all later pages
    ref_pages: List[int] = []
    for b in blocks:
        if b.block_type == "heading" and b.page_start is not None:
            if refs_rx.match((b.text or "").strip()):
                ref_pages.append(int(b.page_start))

    if not ref_pages:
        return blocks

    start_page = min(ref_pages)

    out: List[Block] = []
    for b in blocks:
        if b.page_start is not None and int(b.page_start) >= start_page:
            continue
        out.append(b)
    return out


# Slide-PDF fix: orphan text harvest (schema-tolerant)
def _iter_candidate_text_nodes(doc_dict: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Return a list of dict-like nodes that likely contain text, even if not linked into body.children.
    This is deliberately schema-tolerant for slide PDFs.
    """
    out: List[Dict[str, Any]] = []

    texts = doc_dict.get("texts")
    if isinstance(texts, list):
        out.extend([n for n in texts if isinstance(n, dict)])

    for key in ("pages", "page", "page_objects", "page_elements", "elements"):
        v = doc_dict.get(key)
        if isinstance(v, list):
            for p in v:
                if not isinstance(p, dict):
                    continue
                for kk in ("texts", "text", "children", "items", "elements", "content"):
                    vv = p.get(kk)
                    if isinstance(vv, list):
                        out.extend([n for n in vv if isinstance(n, dict)])

    for k, v in doc_dict.items():
        if k in {"body", "tables", "figures", "images", "pictures"}:
            continue
        if isinstance(v, list) and v and len(v) <= 200000:
            for n in v:
                if isinstance(n, dict):
                    lbl = node_label(n)
                    if lbl in {"text", "paragraph", "list_item", "caption", "section_header", "heading", "title"}:
                        out.append(n)

    return out


def should_harvest_orphans(doc_dict: Dict[str, Any], blocks: List[Block]) -> bool:
    """
    Content-based trigger: only run orphan harvest if we have at least one low-coverage page,
    or if we appear to have extracted almost nothing relative to the number of pages.
    """
    if not ENABLE_ORPHAN_TEXT_HARVEST:
        return False

    # Estimate number of pages if available
    n_pages = None
    pages = doc_dict.get("pages")
    if isinstance(pages, list) and pages:
        n_pages = len(pages)

    # Per-page extracted chars
    per_page = defaultdict(int)
    for b in blocks:
        if b.page_start is not None:
            per_page[int(b.page_start)] += int(b.char_len or 0)

    # If any seen page is low-coverage, harvest
    if any(ch < LOW_TEXT_PAGE_CHAR_THRESHOLD for ch in per_page.values()):
        return True

    # If we know there are many pages but we extracted from very few, harvest
    if n_pages is not None and n_pages >= 5:
        # how many pages have any extracted text?
        covered_pages = len(per_page)
        if covered_pages <= max(1, n_pages // 5):  # <= 20% pages covered
            return True

    # If we extracted very few blocks total, harvest (common in slide parsing failures)
    if len(blocks) < 15:
        return True

    return False


def harvest_orphan_text_blocks(
    doc_dict: Dict[str, Any],
    blocks: List[Block],
    doc_id: str,
    source_file: str,
) -> List[Block]:
    """
    Add additional paragraph/list_item/figure_caption blocks from text nodes that are not reached
    via body.children (common in slide PDFs).
    """
    # Identify low-coverage pages to focus harvesting
    per_page_chars: Dict[int, int] = defaultdict(int)
    for b in blocks:
        if b.page_start is not None:
            per_page_chars[int(b.page_start)] += int(b.char_len or 0)

    # Note: pages with zero extracted text won't be in per_page_chars; the should_harvest_orphans()
    # trigger handles those via doc_dict["pages"] coverage heuristics.
    low_pages = {p for p, ch in per_page_chars.items() if ch < LOW_TEXT_PAGE_CHAR_THRESHOLD}

    existing = {(b.page_start, b.page_end, b.text_hash) for b in blocks}

    # page -> last section_path on that page, based on current blocks
    page_to_section: Dict[int, Optional[str]] = {}
    last_sec: Optional[str] = None
    for b in sorted(blocks, key=lambda x: (x.page_start or 10**9, x.order_idx, x.block_id)):
        if b.section_path:
            last_sec = b.section_path
        if b.page_start is not None:
            page_to_section[int(b.page_start)] = last_sec

    candidates = _iter_candidate_text_nodes(doc_dict)

    new_blocks: List[Block] = []
    seen_counter = 10_000_000

    for idx, node in enumerate(candidates):
        lbl = node_label(node)
        ref = node.get("$ref") if isinstance(node.get("$ref"), str) else None

        if not should_keep_node(node, ref):
            continue

        btype = block_type_from(lbl, ref)
        if btype in {"heading", "table"}:
            continue

        txt = clean_text(extract_text_from_node(node))
        if not txt:
            continue

        if KEEP_FIGURE_DESCRIPTIONS and looks_like_figure_line(txt):
            txt = clean_text(f"Figure description: {txt}")
            btype = "figure_caption"

        if btype in {"paragraph", "text", "unknown"} and looks_like_bullet_line(txt):
            btype = "list_item"

        if len(txt) < min_chars_for_type(btype):
            continue

        p_start, p_end = extract_page_range(node)
        if p_start is None and p_end is None:
            continue

        page_for_filter = p_start or p_end
        if page_for_filter is not None and low_pages and int(page_for_filter) not in low_pages:
            continue

        th = sha1_text(txt)
        key = (p_start, p_end, th)
        if key in existing:
            continue

        sec = page_to_section.get(int(page_for_filter)) if page_for_filter is not None else None

        block_id = f"{doc_id}_b{seen_counter:06d}"
        seen_counter += 1

        sec_title, sec_id = compute_section_fields(sec, source_file, p_start or page_for_filter)

        new_blocks.append(
            Block(
                doc_id=doc_id,
                source_file=source_file,
                block_id=block_id,
                order_idx=1_000_000_000 + idx,
                block_type="paragraph" if btype in {"text", "unknown"} else btype,
                page_start=p_start,
                page_end=p_end,
                section_path=sec,
                section_title=sec_title,
                section_id=sec_id,
                text=txt,
                char_len=len(txt),
                text_hash=th,
            )
        )
        existing.add(key)

    if not new_blocks:
        return blocks

    merged = blocks + new_blocks
    merged.sort(key=lambda b: (b.page_start or 10**9, b.order_idx, b.block_id))
    return merged


# Core extraction (reading order)
def extract_blocks_reading_order(doc_dict: Dict[str, Any], doc_id: str, source_file: str) -> List[Block]:
    blocks: List[Block] = []
    section_stack: List[Tuple[int, str]] = []

    body = doc_dict.get("body") or {}
    children = body.get("children") or []
    if not isinstance(children, list):
        return blocks

    seen = 0

    # dropping References sections during primary traversal
    in_drop_section = False
    drop_section_level: Optional[int] = None

    # page fallback
    last_seen_page: Optional[int] = None

    def update_last_seen(p_start: Optional[int], p_end: Optional[int]) -> None:
        nonlocal last_seen_page
        if p_start is not None:
            last_seen_page = p_start
        elif p_end is not None:
            last_seen_page = p_end

    # paragraph buffer for slide PDFs
    para_buf_texts: List[str] = []
    para_buf_order_idx: Optional[int] = None
    para_buf_section: Optional[str] = None
    para_buf_page: Optional[int] = None
    para_buf_page_end: Optional[int] = None

    def flush_para_buf() -> None:
        nonlocal para_buf_texts, para_buf_order_idx, para_buf_section, para_buf_page, para_buf_page_end, seen

        if not para_buf_texts:
            return

        merged = clean_text("\n".join(para_buf_texts))
        if len(merged) >= MIN_TEXT_CHARS_DEFAULT:
            block_id = f"{doc_id}_b{seen:06d}"
            seen += 1

            sec_title, sec_id = compute_section_fields(para_buf_section, source_file, para_buf_page)

            blocks.append(
                Block(
                    doc_id=doc_id,
                    source_file=source_file,
                    block_id=block_id,
                    order_idx=para_buf_order_idx if para_buf_order_idx is not None else 0,
                    block_type="paragraph",
                    page_start=para_buf_page,
                    page_end=para_buf_page_end if para_buf_page_end is not None else para_buf_page,
                    section_path=para_buf_section,
                    section_title=sec_title,
                    section_id=sec_id,
                    text=merged,
                    char_len=len(merged),
                    text_hash=sha1_text(merged),
                )
            )

        para_buf_texts = []
        para_buf_order_idx = None
        para_buf_section = None
        para_buf_page = None
        para_buf_page_end = None

    def para_key(page_start: Optional[int], section_path: Optional[str]) -> Tuple[Optional[int], Optional[str]]:
        return page_start, section_path

    for order_idx, ch in enumerate(children):
        if not isinstance(ch, dict):
            continue

        ref = ch.get("$ref")
        node: Any = resolve_ref(doc_dict, ref) if ref else ch
        if not isinstance(node, dict):
            continue

        if not should_keep_node(node, ref):
            continue

        label = node_label(node)
        btype = block_type_from(label, ref)

        # Headings
        if label == "section_header" or btype == "heading":
            flush_para_buf()

            level = safe_int(node.get("level")) or 1
            title = clean_text(extract_text_from_node(node))
            if not title or len(title) < MIN_TEXT_CHARS_HEADING:
                continue

            if in_drop_section and drop_section_level is not None:
                if level <= drop_section_level:
                    in_drop_section = False
                    drop_section_level = None
                else:
                    continue

            if should_enter_references(title):
                in_drop_section = True
                drop_section_level = level
                continue

            update_section_stack(section_stack, level, title)
            section_path = section_path_from_stack(section_stack)

            p_start, p_end = extract_page_range(node)
            update_last_seen(p_start, p_end)

            sec_title, sec_id = compute_section_fields(section_path, source_file, p_start)

            block_id = f"{doc_id}_b{seen:06d}"
            seen += 1
            blocks.append(
                Block(
                    doc_id=doc_id,
                    source_file=source_file,
                    block_id=block_id,
                    order_idx=order_idx,
                    block_type="heading",
                    page_start=p_start,
                    page_end=p_end,
                    section_path=section_path,
                    section_title=sec_title,
                    section_id=sec_id,
                    text=title,
                    char_len=len(title),
                    text_hash=sha1_text(title),
                )
            )
            continue


        if in_drop_section:
            continue

        section_path = section_path_from_stack(section_stack)

        # LISTS
        if btype == "list" or label == "list":
            flush_para_buf()

            list_ps, list_pe = extract_page_range(node)
            update_last_seen(list_ps, list_pe)

            item_nodes = list_items_to_nodes(doc_dict, node)
            for it in item_nodes:
                txt = clean_text(extract_text_from_node(it))
                if not txt or len(txt) < MIN_TEXT_CHARS_LIST_ITEM:
                    continue

                p_start, p_end = extract_page_range(it)
                if p_start is None and p_end is None and (list_ps is not None or list_pe is not None):
                    p_start, p_end = list_ps, list_pe
                if p_start is None and p_end is None and last_seen_page is not None:
                    p_start = p_end = last_seen_page

                block_id = f"{doc_id}_b{seen:06d}"
                seen += 1

                sec_title, sec_id = compute_section_fields(section_path, source_file, p_start)

                blocks.append(
                    Block(
                        doc_id=doc_id,
                        source_file=source_file,
                        block_id=block_id,
                        order_idx=order_idx,
                        block_type="list_item",
                        page_start=p_start,
                        page_end=p_end,
                        section_path=section_path,
                        section_title=sec_title,
                        section_id=sec_id,
                        text=txt,
                        char_len=len(txt),
                        text_hash=sha1_text(txt),
                    )
                )

            continue

        # Tables
        if btype == "table":
            flush_para_buf()

            p_start, p_end = extract_page_range(node)
            update_last_seen(p_start, p_end)

            md = clean_text(table_to_markdown(node))

            attached_caption = ""
            attached_notes = ""
            if MERGE_TABLE_CAPTIONS_FROM_TABLE_NODE or MERGE_TABLE_FOOTNOTES_FROM_TABLE_NODE:
                c, n = extract_table_attached_texts(doc_dict, node)
                if MERGE_TABLE_CAPTIONS_FROM_TABLE_NODE:
                    attached_caption = c
                if MERGE_TABLE_FOOTNOTES_FROM_TABLE_NODE:
                    attached_notes = n

            parts: List[str] = []
            if attached_caption:
                parts.append(attached_caption)
            if md:
                parts.append(md)
            if attached_notes:
                parts.append("Notes: " + attached_notes)

            table_text = clean_text("\n\n".join(parts))
            if len(table_text) < MIN_TEXT_CHARS_DEFAULT:
                rows, _ = table_to_grid_strings(node)
                if not rows:
                    continue
                table_text = clean_text(json.dumps(rows, ensure_ascii=False))

            table_struct = None
            if STORE_COMPACT_TABLE_STRUCT:
                rows, header_map = table_to_grid_strings(node)
                table_struct = {"rows": rows, "header_rows": sorted(header_map.keys()) if header_map else []}

            block_id = f"{doc_id}_b{seen:06d}"
            seen += 1
            sec_title, sec_id = compute_section_fields(section_path, source_file, p_start)

            blocks.append(
                Block(
                    doc_id=doc_id,
                    source_file=source_file,
                    block_id=block_id,
                    order_idx=order_idx,
                    block_type="table",
                    page_start=p_start,
                    page_end=p_end,
                    section_path=section_path,
                    section_title=sec_title,
                    section_id=sec_id,
                    text=table_text,
                    char_len=len(table_text),
                    text_hash=sha1_text(table_text),
                    table_struct=table_struct,
                )
            )
            continue

        # Figures / images
        if KEEP_FIGURE_DESCRIPTIONS and is_figure_ref(ref, label):
            flush_para_buf()

            p_start, p_end = extract_page_range(node)
            update_last_seen(p_start, p_end)

            cap = extract_figure_caption_from_node(doc_dict, node)
            if not cap:
                cap = clean_text(extract_text_from_node(node))

            if cap:
                fig_text = clean_text(f"Figure description: {cap}")
                if len(fig_text) >= MIN_TEXT_CHARS_DEFAULT:
                    block_id = f"{doc_id}_b{seen:06d}"
                    seen += 1
                    sec_title, sec_id = compute_section_fields(section_path, source_file, p_start)

                    blocks.append(
                        Block(
                            doc_id=doc_id,
                            source_file=source_file,
                            block_id=block_id,
                            order_idx=order_idx,
                            block_type="figure_caption",
                            page_start=p_start,
                            page_end=p_end,
                            section_path=section_path,
                            section_title=sec_title,
                            section_id=sec_id,
                            text=fig_text,
                            char_len=len(fig_text),
                            text_hash=sha1_text(fig_text),
                        )
                    )
            continue

        # Normal text
        txt = clean_text(extract_text_from_node(node))
        if not txt:
            continue

        if KEEP_FIGURE_DESCRIPTIONS and looks_like_figure_line(txt):
            flush_para_buf()
            txt = clean_text(f"Figure description: {txt}")
            btype = "figure_caption"

        if btype in {"paragraph", "text", "unknown"} and looks_like_bullet_line(txt):
            flush_para_buf()
            btype = "list_item"

        p_start, p_end = extract_page_range(node)
        update_last_seen(p_start, p_end)
        if p_start is None and p_end is None and last_seen_page is not None:
            p_start = p_end = last_seen_page

        # Paragraph buffering: merge consecutive short paragraphs on same page+section
        if btype == "paragraph":
            if len(txt) < MIN_TEXT_CHARS_DEFAULT:
                cur_key = para_key(p_start, section_path)
                buf_key = para_key(para_buf_page, para_buf_section)
                if para_buf_texts and cur_key != buf_key:
                    flush_para_buf()

                if not para_buf_texts:
                    para_buf_order_idx = order_idx
                    para_buf_section = section_path
                    para_buf_page = p_start
                    para_buf_page_end = p_end
                else:
                    if para_buf_page_end is None:
                        para_buf_page_end = p_end
                    elif p_end is not None:
                        para_buf_page_end = max(para_buf_page_end, p_end)

                para_buf_texts.append(txt)
                continue
            else:
                flush_para_buf()

        if len(txt) < min_chars_for_type(btype):
            continue

        block_id = f"{doc_id}_b{seen:06d}"
        seen += 1
        sec_title, sec_id = compute_section_fields(section_path, source_file, p_start)

        blocks.append(
            Block(
                doc_id=doc_id,
                source_file=source_file,
                block_id=block_id,
                order_idx=order_idx,
                block_type=btype,
                page_start=p_start,
                page_end=p_end,
                section_path=section_path,
                section_title=sec_title,
                section_id=sec_id,
                text=txt,
                char_len=len(txt),
                text_hash=sha1_text(txt),
            )
        )

    flush_para_buf()
    return blocks


# IO
def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def write_report(path: Path, report: Dict[str, Any]) -> None:
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


def load_report(path: Path) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def make_tiny_report(doc_id: str, source_file: str, pdf_path: Path, blocks: List[Block]) -> Dict[str, Any]:
    type_counts = Counter(b.block_type for b in blocks)
    with_page = sum(1 for b in blocks if b.page_start is not None)
    with_section_path = sum(1 for b in blocks if b.section_path is not None)
    with_section_id = sum(1 for b in blocks if b.section_id is not None)
    with_section_title = sum(1 for b in blocks if b.section_title is not None)

    return {
        "doc_id": doc_id,
        "source_file": source_file,
        **pdf_fingerprint(pdf_path),
        "num_blocks": len(blocks),
        "block_type_counts": dict(type_counts),
        "blocks_with_page": with_page,
        "blocks_with_section_path": with_section_path,
        "blocks_with_section_id": with_section_id,
        "blocks_with_section_title": with_section_title,
    }


# Main
def main() -> None:
    pdfs = sorted(PDF_DIR.rglob("*.pdf"))
    if not pdfs:
        raise SystemExit(f"No PDFs found under {PDF_DIR.resolve()}")

    converter = DocumentConverter()

    for pdf_path in pdfs:
        doc_id = stable_doc_id(pdf_path)
        source_file = pdf_path.name

        blocks_path = OUT_DIR / f"{doc_id}.blocks.jsonl"
        report_path = OUT_DIR / f"{doc_id}.report.json"

        if SKIP_IF_UNCHANGED and blocks_path.exists() and report_path.exists():
            old = load_report(report_path)
            if isinstance(old, dict):
                fp = pdf_fingerprint(pdf_path)
                if (
                    old.get("source_file") == source_file
                    and old.get("pdf_size_bytes") == fp["pdf_size_bytes"]
                    and old.get("pdf_mtime") == fp["pdf_mtime"]
                ):
                    print(f"Skip (unchanged): {source_file}")
                    continue

        result = converter.convert(str(pdf_path))
        doc_dict = result.document.export_to_dict()

        # 1) Primary extraction
        blocks = extract_blocks_reading_order(doc_dict, doc_id, source_file)

        # 2) Optional orphan harvesting, but only if coverage suggests it's needed
        if should_harvest_orphans(doc_dict, blocks):
            blocks = harvest_orphan_text_blocks(doc_dict, blocks, doc_id, source_file)

        # 3) Merge list items into list blocks
        blocks = merge_consecutive_list_items(blocks)

        # 4) Manual patches
        if ENABLE_PATCHES:
            patches_path = OUT_DIR / f"{doc_id}.patches.json"
            patches = load_patches(patches_path)
            blocks = apply_patches(blocks, patches)

        # 5) Drop references
        blocks = drop_references_blocks(blocks)

        # 6) Finalize ordering for downstream chunking: stable sort then renumber order_idx 0..N-1
        blocks.sort(key=lambda b: (b.page_start if b.page_start is not None else 10**9, b.order_idx, b.block_id))
        for i, b in enumerate(blocks):
            b.order_idx = i

        # 7) Write outputs_old
        write_jsonl(blocks_path, [asdict(b) for b in blocks])

        report = make_tiny_report(doc_id, source_file, pdf_path, blocks)
        write_report(report_path, report)

        print(f"Processed: {source_file} -> {blocks_path.name} ({len(blocks)} blocks)")

    print(f"Done. Outputs in: {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
