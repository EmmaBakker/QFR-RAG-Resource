#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import pickle
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np

# Try to import faiss early
try:
    import faiss  # type: ignore
except ImportError:
    faiss = None

from .determinism import set_deterministic
from .models import build_embedder
from .utils_io import read_json, bm25_tokenize_v1

logger = logging.getLogger(__name__)


class BaseRetriever:
    def search(self, query: str, k: int) -> List[Tuple[str, float]]:
        raise NotImplementedError


class DenseRetriever(BaseRetriever):
    """
    Exact dense retrieval using FAISS IndexFlatIP (cosine via L2-normalized vectors).

    Safety checks:
    - required files exist
    - index meta model_key matches requested model_key (when available)
    - embedder dim matches FAISS index dim
    """
    def __init__(self, model_key: str, index_dir: str, device: Optional[str] = None):
        if faiss is None:
            raise ImportError("faiss is required for DenseRetriever. Install faiss-cpu or faiss-gpu.")

        d = Path(index_dir)
        logger.info(f"Loading Dense Index from {d}")

        faiss_path = d / "faiss.index"
        ids_path = d / "chunk_ids.json"
        meta_path = d / "meta.json"

        if not faiss_path.exists():
            raise FileNotFoundError(f"Missing {faiss_path}")
        if not ids_path.exists():
            raise FileNotFoundError(f"Missing {ids_path}")
        if not meta_path.exists():
            raise FileNotFoundError(f"Missing {meta_path}")

        self.index = faiss.read_index(str(faiss_path))
        self.chunk_ids: List[str] = json.loads(ids_path.read_text(encoding="utf-8"))
        self.meta = read_json(str(meta_path))

        # If meta records model_key, enforce match
        meta_mk = self.meta.get("model_key")
        if meta_mk is not None and str(meta_mk) != str(model_key):
            raise ValueError(f"Index built with model_key={meta_mk} but requested model_key={model_key}")

        # Reuse meta settings to avoid accidental mismatches
        bs = int(self.meta.get("batch_size", 16))
        ml = int(self.meta.get("max_length", 512))

        self.embedder = build_embedder(
            model_key,
            device=device,
            batch_size=bs,
            max_length=ml,
        )

        # Dimensionality check (fail fast)
        probe = self.embedder.embed_queries(["dim_probe"]).astype(np.float32)
        if probe.ndim != 2 or probe.shape[0] != 1:
            raise RuntimeError(f"Unexpected embedder output shape: {probe.shape}")
        if int(probe.shape[1]) != int(self.index.d):
            raise ValueError(f"Embedder dim={probe.shape[1]} but FAISS index dim={self.index.d}")

        # Size sanity check
        if len(self.chunk_ids) != int(self.index.ntotal):
            raise ValueError(f"chunk_ids={len(self.chunk_ids)} but faiss.ntotal={self.index.ntotal}")

    def search(self, query: str, k: int) -> List[Tuple[str, float]]:
        qv = self.embedder.embed_queries([query]).astype(np.float32)
        scores, idxs = self.index.search(qv, k)

        results: List[Tuple[str, float]] = []
        for j in range(min(k, idxs.shape[1])):
            ii = int(idxs[0, j])
            if 0 <= ii < len(self.chunk_ids):
                results.append((self.chunk_ids[ii], float(scores[0, j])))
        return results


class BM25Retriever(BaseRetriever):
    """
    Lexical retrieval using rank_bm25 BM25Okapi.

    Loads meta.json if present (useful for later integrity checks), but does not require it.
    """
    def __init__(self, index_dir: str):
        d = Path(index_dir)
        logger.info(f"Loading BM25 Index from {d}")

        pkl_path = d / "bm25.pkl"
        ids_path = d / "chunk_ids.json"
        meta_path = d / "meta.json"

        if not pkl_path.exists():
            raise FileNotFoundError(f"Missing {pkl_path}")
        if not ids_path.exists():
            raise FileNotFoundError(f"Missing {ids_path}")

        with pkl_path.open("rb") as f:
            self.bm25 = pickle.load(f)

        self.chunk_ids: List[str] = json.loads(ids_path.read_text(encoding="utf-8"))
        self.meta = read_json(str(meta_path)) if meta_path.exists() else {}

        # Basic alignment check: score vector length should match chunk_ids length
        try:
            test_scores = self.bm25.get_scores(["probe"])
            if len(test_scores) != len(self.chunk_ids):
                raise ValueError(
                    f"BM25 score length={len(test_scores)} does not match chunk_ids={len(self.chunk_ids)}"
                )
        except Exception as e:
            raise RuntimeError("Loaded bm25.pkl does not behave like rank_bm25 BM25Okapi") from e

    def search(self, query: str, k: int) -> List[Tuple[str, float]]:
        toks = bm25_tokenize_v1(query)
        scores = self.bm25.get_scores(toks)
        top_indices = np.argsort(-scores)[:k]
        return [(self.chunk_ids[i], float(scores[i])) for i in top_indices]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["dense", "bm25"], required=True)
    ap.add_argument("--model_key", default=None, help="Required for dense (e.g., qwen3_8b)")
    ap.add_argument("--index_dir", required=True)
    ap.add_argument("--query", required=True)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--device", default=None)
    ap.add_argument("--seed", type=int, default=224)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    set_deterministic(args.seed, deterministic_torch=False)

    if args.mode == "bm25":
        retriever: BaseRetriever = BM25Retriever(args.index_dir)
    else:
        if not args.model_key:
            raise SystemExit("--model_key is required for dense retrieval")
        retriever = DenseRetriever(args.model_key, args.index_dir, device=args.device)

    hits = retriever.search(args.query, args.k)

    print(f"\nResults for: {args.query}")
    print("-" * 50)
    for rank, (cid, score) in enumerate(hits, start=1):
        print(f"{rank:02d}\t{score:.6f}\t{cid}")


if __name__ == "__main__":
    main()
