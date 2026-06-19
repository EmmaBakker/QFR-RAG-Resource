from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Union

logger = logging.getLogger(__name__)


def _iter_jsonl(path: Union[str, Path]) -> Iterable[dict]:
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSONL at {p.as_posix()}:{line_no}: {e}") from e


@dataclass
class CorpusStore:
    """
    Loads a chunked corpus JSONL file into memory for fast lookup by chunk_id.

    Required JSONL fields per row:
      - chunk_id: str
      - text: str

    strict:
      - True: fail fast on malformed rows / duplicates
      - False: skip malformed rows and keep last duplicate
    """
    corpus_path: Union[str, Path]
    strict: bool = True
    log_every: int = 200000

    by_id: Dict[str, str] | None = None

    def load(self) -> None:
        p = Path(self.corpus_path)
        if not p.exists():
            raise FileNotFoundError(f"Corpus not found: {p.as_posix()}")

        by_id: Dict[str, str] = {}
        n_rows = 0
        n_skipped = 0
        n_dupes = 0

        for row in _iter_jsonl(p):
            n_rows += 1
            if self.log_every and (n_rows % self.log_every == 0):
                logger.info(f"Corpus load progress: rows={n_rows:,} chunks={len(by_id):,} skipped={n_skipped:,} dupes={n_dupes:,}")

            cid = row.get("chunk_id")
            txt = row.get("text")

            if not cid or not isinstance(cid, str):
                n_skipped += 1
                if self.strict:
                    raise ValueError(f"Row missing valid 'chunk_id' at row={n_rows}")
                continue

            if txt is None:
                txt = ""
            if not isinstance(txt, str):
                n_skipped += 1
                if self.strict:
                    raise ValueError(f"Row chunk_id={cid!r} has non-string 'text' at row={n_rows}")
                continue

            if cid in by_id:
                n_dupes += 1
                if self.strict:
                    raise ValueError(f"Duplicate chunk_id={cid!r} encountered at row={n_rows}")
                # non-strict: keep last
            by_id[cid] = txt

        if not by_id and self.strict:
            raise ValueError(f"Corpus loaded zero chunks from {p.as_posix()} (rows={n_rows}, skipped={n_skipped})")

        self.by_id = by_id
        logger.info(f"Corpus loaded: chunks={len(by_id):,} rows={n_rows:,} skipped={n_skipped:,} dupes={n_dupes:,} from={p.as_posix()}")

    def __len__(self) -> int:
        return 0 if self.by_id is None else len(self.by_id)

    def get_text(self, chunk_id: str) -> Optional[str]:
        if self.by_id is None:
            raise RuntimeError("CorpusStore not loaded. Call load() before get_text().")
        return self.by_id.get(str(chunk_id))

    def must_get_text(self, chunk_id: str) -> str:
        txt = self.get_text(chunk_id)
        if txt is None:
            raise KeyError(f"chunk_id not found in corpus: {chunk_id}")
        return txt