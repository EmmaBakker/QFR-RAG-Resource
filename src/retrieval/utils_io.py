from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterator, List, Tuple


def iter_jsonl(path: str) -> Iterator[Dict[str, Any]]:
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def load_chunks_jsonl(path: str) -> Tuple[List[str], List[str]]:
    """
    Reads corpus.chunks.jsonl and returns parallel arrays chunk_ids, texts.
    Requires fields: chunk_id, text
    """
    chunk_ids: List[str] = []
    texts: List[str] = []
    for row in iter_jsonl(path):
        cid = row.get("chunk_id")
        txt = row.get("text")
        if not cid or not isinstance(txt, str) or not txt.strip():
            continue
        chunk_ids.append(str(cid))
        texts.append(txt)

    if len(chunk_ids) != len(texts):
        raise RuntimeError("Internal error: chunk_ids and texts length mismatch.")
    return chunk_ids, texts


def write_json(path: str, obj: Any) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def read_json(path: str) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


# BM25 tokenization (single source of truth)

_BM25_TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:[._/-][A-Za-z0-9]+)*")


def bm25_tokenize_v1(text: str) -> List[str]:
    """
    Stable regex tokenizer designed to preserve technical tokens like:
    QFR, FFRQCA, 3D-QCA, ISO/IEC, v2.0, AHA/ACC, etc.
    """
    return [m.group(0).lower() for m in _BM25_TOKEN_RE.finditer(text)]


# Corpus fingerprint (index integrity)

def corpus_fingerprint(chunk_ids: List[str], texts: List[str]) -> str:
    """
    Hashes (chunk_id + text) pairs in order; detects any change in corpus content or ordering.
    Useful to prove an index matches the exact corpus used during evaluation.
    """
    h = hashlib.sha256()
    for cid, txt in zip(chunk_ids, texts):
        h.update(cid.encode("utf-8"))
        h.update(b"\n")
        h.update(txt.encode("utf-8"))
        h.update(b"\n---\n")
    return h.hexdigest()
