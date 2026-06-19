#!/usr/bin/env python3
"""
verify_index.py

Integrity checks + smoke-test retrieval for your BM25 and Dense (FAISS) indexes.

What it verifies (high level):
- Required files exist
- chunk_ids.json length matches index size
- meta.json is readable and internally consistent
- (Optional but recommended) corpus fingerprint matches meta's stored fingerprint
- Dense: embedder output dimension matches FAISS index dimension
- Smoke test: runs a few queries and checks you get k results with finite scores

Examples:
  # Verify BM25 index (also checks corpus fingerprint if meta contains it)
  python -m src.retrieval.verify_index --mode bm25 --index_dir indexes/bm25 --corpus data/processed/corpus.chunks.jsonl

  # Verify dense index (model_key can be inferred from meta.json if present)
  python -m src.retrieval.verify_index --mode dense --index_dir indexes/dense/qwen3_8b --corpus data/processed/corpus.chunks.jsonl

  # Auto-detect mode from meta.json
  python -m src.retrieval.verify_index --mode auto --index_dir indexes/dense/qwen3_8b --corpus data/processed/corpus.chunks.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .determinism import set_deterministic
from .models import build_embedder
from .utils_io import read_json, load_chunks_jsonl, corpus_fingerprint, bm25_tokenize_v1

logger = logging.getLogger(__name__)


# Helpers
class VerifyError(RuntimeError):
    pass


@dataclass
class VerifyResult:
    ok: bool
    mode: str
    index_dir: str
    messages: List[str]


def _require(path: Path, name: str) -> None:
    if not path.exists():
        raise VerifyError(f"Missing required file: {name} ({path})")


def _is_finite_number(x: float) -> bool:
    return not (math.isnan(x) or math.isinf(x))


def _load_json_list(path: Path) -> List[Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise VerifyError(f"{path.name} must be a JSON list, got: {type(data)}")
    return data


def _infer_mode_from_meta(meta: Dict[str, Any]) -> str:
    t = str(meta.get("type", "")).lower()
    if "bm25" in t:
        return "bm25"
    if "dense" in t or "faiss" in t:
        return "dense"
    # Fallback heuristic: if faiss.index exists -> dense, if bm25.pkl exists -> bm25
    return "unknown"


# BM25 verification
def verify_bm25(index_dir: Path, corpus_path: Optional[Path], k: int, smoke_queries: List[str]) -> VerifyResult:
    messages: List[str] = []
    _require(index_dir / "bm25.pkl", "bm25.pkl")
    _require(index_dir / "chunk_ids.json", "chunk_ids.json")
    _require(index_dir / "meta.json", "meta.json")

    meta = read_json(str(index_dir / "meta.json"))
    if not isinstance(meta, dict):
        raise VerifyError("meta.json must be a JSON object")

    chunk_ids = _load_json_list(index_dir / "chunk_ids.json")
    if len(chunk_ids) == 0:
        raise VerifyError("chunk_ids.json is empty")

    with (index_dir / "bm25.pkl").open("rb") as f:
        bm25 = pickle.load(f)

    # Basic type/shape checks
    if not hasattr(bm25, "get_scores"):
        raise VerifyError("bm25.pkl does not look like a rank_bm25 BM25Okapi object (missing get_scores).")

    # rank_bm25 stores corpus size in internal lists; best we can do is compare score vector length
    test_scores = bm25.get_scores(["test"])
    if len(test_scores) != len(chunk_ids):
        raise VerifyError(
            f"BM25 corpus size mismatch: bm25.get_scores(...) returned {len(test_scores)} scores "
            f"but chunk_ids has {len(chunk_ids)} entries."
        )
    messages.append(f"BM25 size OK: {len(chunk_ids)} chunks.")

    # Optional: fingerprint check
    fp_meta = meta.get("corpus_fingerprint_sha256")
    if corpus_path is not None and corpus_path.exists():
        corpus_chunk_ids, texts = load_chunks_jsonl(str(corpus_path))
        if len(corpus_chunk_ids) != len(chunk_ids):
            raise VerifyError(
                f"Corpus chunk count ({len(corpus_chunk_ids)}) differs from index chunk_ids.json ({len(chunk_ids)})."
            )
        # Ensure ordering matches exactly (this matters!)
        if corpus_chunk_ids != chunk_ids:
            raise VerifyError("chunk_ids.json does not match corpus chunk_id ordering. Index/corpus are misaligned.")
        fp_now = corpus_fingerprint(corpus_chunk_ids, texts)
        messages.append(f"Corpus fingerprint computed: {fp_now[:12]}...")

        if fp_meta:
            if fp_now != fp_meta:
                raise VerifyError(
                    "Corpus fingerprint mismatch: meta.json fingerprint differs from current corpus. "
                    "This index was built from a different corpus version."
                )
            messages.append("Corpus fingerprint matches meta.json.")
        else:
            messages.append("meta.json has no corpus_fingerprint_sha256; computed fingerprint but could not compare.")
    else:
        messages.append("No corpus provided (or file missing); skipping fingerprint/ordering checks.")

    # Smoke test retrieval
    for q in smoke_queries:
        toks = bm25_tokenize_v1(q)
        scores = bm25.get_scores(toks)
        if len(scores) != len(chunk_ids):
            raise VerifyError("BM25 scores length changed unexpectedly during smoke test.")

        top = np.argsort(-scores)[:k]
        if len(top) == 0:
            raise VerifyError("BM25 smoke test returned no results (unexpected).")

        top_scores = [float(scores[i]) for i in top]
        if not all(_is_finite_number(s) for s in top_scores):
            raise VerifyError("BM25 smoke test returned non-finite scores.")
        messages.append(f"BM25 smoke OK for query={q!r} (top1={top_scores[0]:.6f}).")

    return VerifyResult(ok=True, mode="bm25", index_dir=str(index_dir), messages=messages)


# Dense (FAISS) verification
def verify_dense(
    index_dir: Path,
    corpus_path: Optional[Path],
    model_key: Optional[str],
    device: Optional[str],
    k: int,
    smoke_queries: List[str],
) -> VerifyResult:
    messages: List[str] = []

    _require(index_dir / "faiss.index", "faiss.index")
    _require(index_dir / "chunk_ids.json", "chunk_ids.json")
    _require(index_dir / "meta.json", "meta.json")

    try:
        import faiss  # type: ignore
    except Exception as e:
        raise VerifyError("faiss is required to verify dense indexes. Install faiss-cpu or faiss-gpu.") from e

    meta = read_json(str(index_dir / "meta.json"))
    if not isinstance(meta, dict):
        raise VerifyError("meta.json must be a JSON object")

    chunk_ids = _load_json_list(index_dir / "chunk_ids.json")
    if len(chunk_ids) == 0:
        raise VerifyError("chunk_ids.json is empty")

    index = faiss.read_index(str(index_dir / "faiss.index"))

    # Size checks
    ntotal = int(index.ntotal)
    if ntotal != len(chunk_ids):
        raise VerifyError(f"FAISS ntotal={ntotal} but chunk_ids.json has {len(chunk_ids)} entries.")
    messages.append(f"FAISS size OK: ntotal={ntotal}.")

    # Dimensionality checks
    dim = int(index.d)
    meta_dim = meta.get("dim")
    if meta_dim is not None and int(meta_dim) != dim:
        raise VerifyError(f"meta.json dim={meta_dim} but faiss index dim={dim}.")
    messages.append(f"FAISS dim OK: d={dim}.")

    # Model key consistency
    meta_mk = meta.get("model_key")
    if model_key is None:
        model_key = str(meta_mk) if meta_mk else None

    if model_key is None:
        raise VerifyError(
            "model_key not provided and meta.json has no model_key. "
            "Pass --model_key explicitly."
        )

    if meta_mk and str(meta_mk) != str(model_key):
        raise VerifyError(f"Index meta.model_key={meta_mk} but you requested model_key={model_key}.")
    messages.append(f"Model key OK: {model_key}.")

    # Optional: fingerprint + chunk ordering check
    fp_meta = meta.get("corpus_fingerprint_sha256")
    if corpus_path is not None and corpus_path.exists():
        corpus_chunk_ids, texts = load_chunks_jsonl(str(corpus_path))
        if len(corpus_chunk_ids) != len(chunk_ids):
            raise VerifyError(
                f"Corpus chunk count ({len(corpus_chunk_ids)}) differs from index chunk_ids.json ({len(chunk_ids)})."
            )
        if corpus_chunk_ids != chunk_ids:
            raise VerifyError("chunk_ids.json does not match corpus chunk_id ordering. Index/corpus are misaligned.")
        fp_now = corpus_fingerprint(corpus_chunk_ids, texts)
        messages.append(f"Corpus fingerprint computed: {fp_now[:12]}...")

        if fp_meta:
            if fp_now != fp_meta:
                raise VerifyError(
                    "Corpus fingerprint mismatch: meta.json fingerprint differs from current corpus. "
                    "This index was built from a different corpus version."
                )
            messages.append("Corpus fingerprint matches meta.json.")
        else:
            messages.append("meta.json has no corpus_fingerprint_sha256; computed fingerprint but could not compare.")
    else:
        messages.append("No corpus provided (or file missing); skipping fingerprint/ordering checks.")

    # Build embedder and verify output dimension matches FAISS dimension
    embedder = build_embedder(
        model_key,
        device=device,
        batch_size=int(meta.get("batch_size", 16)),
        max_length=int(meta.get("max_length", 512)),
    )

    probe = embedder.embed_queries(["dimension probe"])
    if probe.ndim != 2 or probe.shape[0] != 1:
        raise VerifyError(f"Embedder probe returned unexpected shape: {probe.shape}")
    if int(probe.shape[1]) != dim:
        raise VerifyError(f"Embedder dim={probe.shape[1]} but FAISS dim={dim}. Wrong model or wrong index.")
    messages.append(f"Embedder dimension matches FAISS (dim={dim}).")

    # Smoke test retrieval
    for q in smoke_queries:
        qv = embedder.embed_queries([q]).astype(np.float32)
        scores, idxs = index.search(qv, k)
        if idxs.shape[1] == 0:
            raise VerifyError("Dense smoke test returned no results (unexpected).")

        out_scores = [float(scores[0, j]) for j in range(min(k, idxs.shape[1]))]
        if not all(_is_finite_number(s) for s in out_scores):
            raise VerifyError("Dense smoke test returned non-finite scores.")
        messages.append(f"Dense smoke OK for query={q!r} (top1={out_scores[0]:.6f}).")

    return VerifyResult(ok=True, mode="dense", index_dir=str(index_dir), messages=messages)


# CLI
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["auto", "dense", "bm25"], default="auto")
    ap.add_argument("--index_dir", required=True)
    ap.add_argument("--corpus", default=None, help="Optional: data/processed/corpus.chunks.jsonl (enables fingerprint+ordering checks)")
    ap.add_argument("--model_key", default=None, help="Dense only. If omitted, inferred from meta.json when possible.")
    ap.add_argument("--device", default="cpu", help="Dense embedding device. Use 'cpu' or 'cuda'. Default: cpu.")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--seed", type=int, default=224)
    ap.add_argument(
        "--smoke_queries",
        default=None,
        help="Optional JSON list of queries. If omitted, uses a small default set.",
    )
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    set_deterministic(args.seed, deterministic_torch=False)

    index_dir = Path(args.index_dir)
    if not index_dir.exists():
        raise SystemExit(f"index_dir does not exist: {index_dir}")

    corpus_path = Path(args.corpus) if args.corpus else None

    if args.smoke_queries:
        try:
            smoke_queries = json.loads(args.smoke_queries)
            if not isinstance(smoke_queries, list) or not all(isinstance(x, str) for x in smoke_queries):
                raise ValueError
        except Exception:
            raise SystemExit("--smoke_queries must be a JSON list of strings, e.g. '[\"q1\",\"q2\"]'")
    else:
        smoke_queries = [
            "QFR computation frame rate",
            "angiographic projections for 3D reconstruction",
            "contrast injection requirements",
        ]

    # Determine mode if auto
    mode = args.mode
    meta: Dict[str, Any] = {}
    meta_path = index_dir / "meta.json"
    if meta_path.exists():
        try:
            meta = read_json(str(meta_path))
        except Exception:
            meta = {}

    if mode == "auto":
        guessed = _infer_mode_from_meta(meta) if isinstance(meta, dict) else "unknown"
        if guessed == "unknown":
            # heuristic fallback
            if (index_dir / "faiss.index").exists():
                guessed = "dense"
            elif (index_dir / "bm25.pkl").exists():
                guessed = "bm25"
        if guessed not in ("dense", "bm25"):
            raise SystemExit("Could not auto-detect mode. Pass --mode dense or --mode bm25.")
        mode = guessed

    try:
        if mode == "bm25":
            res = verify_bm25(index_dir=index_dir, corpus_path=corpus_path, k=args.k, smoke_queries=smoke_queries)
        else:
            res = verify_dense(
                index_dir=index_dir,
                corpus_path=corpus_path,
                model_key=args.model_key,
                device=args.device,
                k=args.k,
                smoke_queries=smoke_queries,
            )
    except VerifyError as e:
        logger.error(f"VERIFY FAILED: {e}")
        raise SystemExit(2)

    print("\n=== VERIFY OK ===")
    print(f"mode      : {res.mode}")
    print(f"index_dir : {res.index_dir}")
    for m in res.messages:
        print(f"- {m}")


if __name__ == "__main__":
    main()
