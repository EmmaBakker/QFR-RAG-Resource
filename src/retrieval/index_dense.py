#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np

from .determinism import set_deterministic
from .models import build_embedder, MODEL_SPECS
from .utils_io import load_chunks_jsonl, write_json, corpus_fingerprint

logger = logging.getLogger(__name__)


def build_faiss_ip_index(vectors: np.ndarray):
    """
    Cosine similarity = inner product on L2-normalized vectors.
    We use FlatIP for exact search (deterministic given vectors).
    """
    import faiss
    dim = vectors.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(vectors)
    return index


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_key", required=True, choices=sorted(MODEL_SPECS.keys()))
    ap.add_argument("--corpus", default="data/processed/corpus.chunks.jsonl")
    ap.add_argument("--index_root", default="indexes/dense")
    ap.add_argument("--device", default=None)
    ap.add_argument("--batch_size", type=int, default=None)
    ap.add_argument("--max_length", type=int, default=None)
    ap.add_argument("--seed", type=int, default=224)
    ap.add_argument("--deterministic_torch", action="store_true", help="Enable strict torch determinism (may error on some kernels).")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    set_deterministic(args.seed, deterministic_torch=args.deterministic_torch)

    spec = MODEL_SPECS[args.model_key]
    out_dir = Path(args.index_root) / spec.model_key
    out_dir.mkdir(parents=True, exist_ok=True)

    chunk_ids, texts = load_chunks_jsonl(args.corpus)
    if not chunk_ids:
        raise SystemExit("No chunks found. corpus must have fields: chunk_id, text")

    fp = corpus_fingerprint(chunk_ids, texts)
    logger.info(f"Loaded {len(chunk_ids)} chunks from {args.corpus}")
    logger.info(f"Corpus fingerprint: {fp[:12]}...")

    embedder = build_embedder(
        args.model_key,
        device=args.device,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )

    # Embed corpus
    t0 = time.time()
    vecs_list: list[np.ndarray] = []
    outer_bs = 256  # outer loop chunking
    total_texts = len(texts)

    for i in range(0, total_texts, outer_bs):
        batch = texts[i:i + outer_bs]

        # Print before starting the batch so you know it's working
        print(f">>> [{spec.model_key}] Starting batch {i} to {min(i + outer_bs, total_texts)} of {total_texts}...")

        v = embedder.embed_documents(batch)  # normalized if spec.normalize=True
        vecs_list.append(v)

        # Calculate time spent
        elapsed = time.time() - t0
        avg_speed = (i + len(batch)) / elapsed
        remaining = (total_texts - (i + len(batch))) / avg_speed if avg_speed > 0 else 0

        print(f"DONE: {min(i + outer_bs, total_texts)}/{total_texts} ({100 * (i + len(batch)) / total_texts:.1f}%) | "
              f"Speed: {avg_speed:.2f} docs/sec | "
              f"Est. remaining: {remaining / 60:.1f} min")

    vectors = np.vstack(vecs_list).astype(np.float32)
    logger.info(f"[{spec.model_key}] vectors shape={vectors.shape} time={time.time() - t0:.1f}s")

    # Sanity checks
    if vectors.shape[0] != len(chunk_ids):
        raise RuntimeError(f"Vector count mismatch: vectors={vectors.shape[0]} chunk_ids={len(chunk_ids)}")

    # FAISS index
    index = build_faiss_ip_index(vectors)
    import faiss
    faiss.write_index(index, str(out_dir / "faiss.index"))

    (out_dir / "chunk_ids.json").write_text(json.dumps(chunk_ids, indent=2), encoding="utf-8")

    meta = {
        "type": "dense_faiss_flatip_cosine",
        "model_key": spec.model_key,
        "kind": spec.kind,
        "doc_hf_id": spec.doc_hf_id,
        "query_hf_id": spec.effective_query_hf_id(),
        "adapter_id": spec.adapter_id,
        "pool": spec.pool,
        "query_prompt_name": spec.query_prompt_name,
        "normalize": spec.normalize,
        "max_length": int(embedder.max_length),
        "batch_size": int(embedder.batch_size),
        "num_chunks": len(chunk_ids),
        "dim": int(vectors.shape[1]),
        "corpus_path": str(Path(args.corpus).as_posix()),
        "corpus_fingerprint_sha256": fp,
        "seed": args.seed,
        "deterministic_torch": bool(args.deterministic_torch),
    }
    write_json(str(out_dir / "meta.json"), meta)

    logger.info(f"[{spec.model_key}] wrote dense index: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
