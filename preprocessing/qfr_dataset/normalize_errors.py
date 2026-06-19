#!/usr/bin/env python3
"""Normalise transcription/OCR errors in pre-processed QFR corpus blocks.

Reads ``*.blocks.jsonl`` files under a given directory, applies a
search-and-replace mapping loaded from a JSON file, and writes the
corrected text back in-place (or to a parallel ``.normalized.jsonl`` file
with ``--no-inplace``).

Typical usage
-------------
Run from the repository root::

    python -m preprocessing.qfr_dataset.normalize_errors

Override defaults::

    python -m preprocessing.qfr_dataset.normalize_errors \\
        --blocks-dir  data/processed \\
        --mapping-path preprocessing/qfr_dataset/text_normalization.json \\
        --inplace
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple


# ---------------------------------------------------------------------------
# Text normaliser
# ---------------------------------------------------------------------------

class TextNormalizer:
    def __init__(self, replacements: Dict[str, str]):
        self.replacements = replacements

    @classmethod
    def from_json(cls, path: str | Path) -> "TextNormalizer":
        path = Path(path)
        data = json.loads(path.read_text(encoding="utf-8"))
        reps = data.get("replacements", {})
        if not isinstance(reps, dict) or not reps:
            raise ValueError("JSON must contain a non-empty 'replacements' object.")
        reps = {str(k): str(v) for k, v in reps.items()}
        return cls(replacements=reps)

    def normalize(self, text: str) -> Tuple[str, Dict[str, int]]:
        out = text or ""
        stats: Dict[str, int] = {}
        # Apply longest keys first to avoid partial-overlap issues.
        for src in sorted(self.replacements.keys(), key=len, reverse=True):
            dst = self.replacements[src]
            if src and src in out:
                count = out.count(src)
                out = out.replace(src, dst)
                stats[src] = count
        return out, stats


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def sha1_text(s: str) -> str:
    import hashlib
    return hashlib.sha1((s or "").encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# CLI & main
# ---------------------------------------------------------------------------

_DEFAULT_BLOCKS_DIR = Path("data/processed")
_DEFAULT_MAPPING_PATH = Path("preprocessing/qfr_dataset/text_normalization.json")


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Normalise transcription/OCR errors in QFR corpus blocks."
    )
    ap.add_argument(
        "--blocks-dir",
        default=str(_DEFAULT_BLOCKS_DIR),
        help=f"Directory containing *.blocks.jsonl files. Default: {_DEFAULT_BLOCKS_DIR}",
    )
    ap.add_argument(
        "--mapping-path",
        default=str(_DEFAULT_MAPPING_PATH),
        help=(
            "Path to the JSON file with a 'replacements' mapping. "
            f"Default: {_DEFAULT_MAPPING_PATH}"
        ),
    )
    ap.add_argument(
        "--inplace",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Write corrections back to the original file (default). "
            "Use --no-inplace to write to *.normalized.jsonl instead."
        ),
    )
    return ap


def main() -> None:
    args = build_arg_parser().parse_args()

    blocks_dir = Path(args.blocks_dir)
    mapping_path = Path(args.mapping_path)
    inplace: bool = args.inplace

    if not mapping_path.exists():
        raise FileNotFoundError(
            f"Mapping file not found: {mapping_path}\n"
            "Pass --mapping-path to point to the correct location."
        )
    if not blocks_dir.exists():
        raise FileNotFoundError(
            f"Blocks directory not found: {blocks_dir}\n"
            "Pass --blocks-dir to point to the correct location."
        )

    normalizer = TextNormalizer.from_json(mapping_path)

    jsonl_files = sorted(blocks_dir.rglob("*.blocks.jsonl"))
    if not jsonl_files:
        raise RuntimeError(
            f"No *.blocks.jsonl files found under {blocks_dir.resolve()}"
        )

    print(f"Found {len(jsonl_files)} *.blocks.jsonl files in {blocks_dir}")
    for path in jsonl_files:
        total_changes = 0
        out_lines = []

        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            obj = json.loads(line)
            text = obj.get("text", "")
            new_text, stats = normalizer.normalize(text)
            if stats:
                obj["text"] = new_text
                obj["char_len"] = len(new_text)
                obj["text_hash"] = sha1_text(new_text)
                total_changes += sum(stats.values())
            out_lines.append(json.dumps(obj, ensure_ascii=False))

        if total_changes == 0:
            continue

        out_path = path if inplace else path.with_suffix(".normalized.jsonl")
        out_path.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
        print(f"{path.name}: {total_changes} replacements → {out_path}")

    print("Done.")


if __name__ == "__main__":
    main()
