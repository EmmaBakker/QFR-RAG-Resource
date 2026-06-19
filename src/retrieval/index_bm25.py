#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import pickle
from pathlib import Path

from rank_bm25 import BM25Okapi

from .determinism import set_deterministic
from .utils_io import load_chunks_jsonl, write_json, bm25_tokenize_v1, corpus_fingerprint

logger = logging.getLogger(__name__)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="data/processed/corpus.chunks.jsonl")
    ap.add_argument("--out_dir", default="indexes/bm25")
    ap.add_argument("--seed", type=int, default=224)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    set_deterministic(args.seed, deterministic_torch=False)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    chunk_ids, texts = load_chunks_jsonl(args.corpus)
    if not chunk_ids:
        raise SystemExit("No chunks found. corpus must have fields: chunk_id, text")

    fp = corpus_fingerprint(chunk_ids, texts)
    logger.info(f"Loaded {len(chunk_ids)} chunks from {args.corpus}")
    logger.info(f"Corpus fingerprint: {fp[:12]}...")

    tokenized = [bm25_tokenize_v1(t) for t in texts]
    bm25 = BM25Okapi(tokenized)

    with (out_dir / "bm25.pkl").open("wb") as f:
        pickle.dump(bm25, f)

    (out_dir / "chunk_ids.json").write_text(json.dumps(chunk_ids, indent=2), encoding="utf-8")

    meta = {
        "type": "bm25_okapi",
        "tokenizer": "bm25_tokenize_v1_regex",
        "num_chunks": len(chunk_ids),
        "corpus_path": str(Path(args.corpus).as_posix()),
        "corpus_fingerprint_sha256": fp,
        "seed": args.seed,
    }
    write_json(str(out_dir / "meta.json"), meta)

    logger.info(f"Wrote BM25 index to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
