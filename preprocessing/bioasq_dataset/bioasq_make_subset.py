#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
import hashlib
from collections import defaultdict, Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple


RAW_PATH_DEFAULT = Path("data/bioasq/raw/training13b.json")
OUT_EVAL_DEFAULT = Path("data/bioasq_subset/eval")
OUT_PROC_DEFAULT = Path("data/bioasq_subset/processed")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def pmid_from_url(url: str) -> Optional[str]:
    url = (url or "").strip()
    if "pubmed" not in url.lower():
        return None
    parts = [p for p in url.replace("?", "/").split("/") if p]
    for i in range(len(parts) - 1, -1, -1):
        if parts[i].isdigit():
            return parts[i]
    return None


def normalize_question_type(qtype: str) -> str:
    qtype = (qtype or "").strip().lower()
    if qtype in {"yesno", "factoid", "list", "summary"}:
        return qtype
    return qtype or "unknown"


def normalize_documents(doc_urls: List[str]) -> List[str]:
    out: List[str] = []
    seen: Set[str] = set()
    for url in doc_urls or []:
        pmid = pmid_from_url(url)
        if pmid is None:
            continue
        did = f"pmid:{pmid}"
        if did not in seen:
            seen.add(did)
            out.append(did)
    return out


def normalize_snippets(snippets: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for sn in snippets or []:
        doc_url = str(sn.get("document") or "").strip()
        pmid = pmid_from_url(doc_url)
        out.append({
            "document": f"pmid:{pmid}" if pmid else doc_url,
            "text": str(sn.get("text") or "").strip(),
            "beginSection": sn.get("beginSection"),
            "endSection": sn.get("endSection"),
            "offsetInBeginSection": sn.get("offsetInBeginSection"),
            "offsetInEndSection": sn.get("offsetInEndSection"),
        })
    return out


def largest_remainder_allocation(
    counts_by_type: Dict[str, int],
    n_total: int,
    min_per_type: int = 0,
) -> Dict[str, int]:
    types = sorted(counts_by_type.keys())
    total_available = sum(counts_by_type.values())
    if n_total > total_available:
        raise ValueError(
            f"Requested n_total={n_total}, but only {total_available} questions are available."
        )

    # Start with minimum per type where possible
    allocation: Dict[str, int] = {}
    remaining = n_total

    eligible_types = [t for t in types if counts_by_type[t] > 0]
    if min_per_type > 0:
        for t in eligible_types:
            take = min(min_per_type, counts_by_type[t])
            allocation[t] = take
            remaining -= take
    else:
        allocation = {t: 0 for t in types}

    if remaining < 0:
        raise ValueError(
            f"min_per_type={min_per_type} is too large for n_total={n_total}."
        )

    residual_capacity = {
        t: counts_by_type[t] - allocation.get(t, 0)
        for t in types
    }
    residual_total = sum(residual_capacity.values())

    if remaining == 0 or residual_total == 0:
        return allocation

    quotas: Dict[str, float] = {
        t: remaining * (residual_capacity[t] / residual_total)
        for t in types
    }

    floors: Dict[str, int] = {
        t: min(int(math.floor(quotas[t])), residual_capacity[t])
        for t in types
    }

    for t in types:
        allocation[t] = allocation.get(t, 0) + floors[t]

    assigned = sum(floors.values())
    leftovers = remaining - assigned

    remainders: List[Tuple[str, float]] = sorted(
        ((t, quotas[t] - floors[t]) for t in types if residual_capacity[t] > floors[t]),
        key=lambda x: (-x[1], x[0]),
    )

    i = 0
    while leftovers > 0 and i < len(remainders):
        t = remainders[i][0]
        if allocation[t] < counts_by_type[t]:
            allocation[t] += 1
            leftovers -= 1
        i += 1

    if leftovers != 0:
        # fallback fill if rounding/capacity edge case
        for t in types:
            while leftovers > 0 and allocation[t] < counts_by_type[t]:
                allocation[t] += 1
                leftovers -= 1

    return allocation


def parse_explicit_counts(s: Optional[str]) -> Optional[Dict[str, int]]:
    if not s:
        return None
    out: Dict[str, int] = {}
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise ValueError(
                f"Invalid --per_type_counts item '{part}'. Expected format like yesno=50."
            )
        k, v = part.split("=", 1)
        out[normalize_question_type(k)] = int(v)
    return out


def sample_subset(
    questions: List[Dict[str, Any]],
    n_total: Optional[int],
    per_type_counts: Optional[Dict[str, int]],
    seed: int,
    min_per_type: int,
) -> List[Dict[str, Any]]:
    rng = random.Random(seed)

    by_type: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for q in questions:
        qtype = normalize_question_type(str(q.get("type") or ""))
        by_type[qtype].append(q)

    for qtype in by_type:
        rng.shuffle(by_type[qtype])

    available_counts = {t: len(v) for t, v in by_type.items()}

    if per_type_counts is not None:
        allocation = {}
        for t, req in per_type_counts.items():
            avail = available_counts.get(t, 0)
            if req > avail:
                raise ValueError(
                    f"Requested {req} questions for type '{t}', but only {avail} are available."
                )
            allocation[t] = req
        # include zero for other types
        for t in available_counts:
            allocation.setdefault(t, 0)
    else:
        if n_total is None:
            raise ValueError("Either n_total or per_type_counts must be provided.")
        allocation = largest_remainder_allocation(
            counts_by_type=available_counts,
            n_total=n_total,
            min_per_type=min_per_type,
        )

    selected: List[Dict[str, Any]] = []
    for t in sorted(allocation.keys()):
        k = allocation[t]
        if k <= 0:
            continue
        selected.extend(by_type[t][:k])

    rng.shuffle(selected)
    return selected


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, default=RAW_PATH_DEFAULT)
    ap.add_argument("--out_eval", type=Path, default=OUT_EVAL_DEFAULT)
    ap.add_argument("--out_proc", type=Path, default=OUT_PROC_DEFAULT)

    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--n_total", type=int, default=None)
    group.add_argument(
        "--per_type_counts",
        type=str,
        default=None,
        help="Example: yesno=125,factoid=125,list=125,summary=125",
    )

    ap.add_argument("--seed", type=int, default=224)
    ap.add_argument(
        "--min_per_type",
        type=int,
        default=10,
        help="Only used with --n_total. Ensures at least this many per type when possible.",
    )

    args = ap.parse_args()

    raw_path = args.raw
    out_eval = args.out_eval
    out_proc = args.out_proc
    out_eval.mkdir(parents=True, exist_ok=True)
    out_proc.mkdir(parents=True, exist_ok=True)

    data = json.loads(raw_path.read_text(encoding="utf-8"))
    questions: List[Dict[str, Any]] = data["questions"]

    explicit_counts = parse_explicit_counts(args.per_type_counts)

    selected = sample_subset(
        questions=questions,
        n_total=args.n_total,
        per_type_counts=explicit_counts,
        seed=args.seed,
        min_per_type=args.min_per_type,
    )

    q_out = out_eval / "queries.jsonl"
    qrels_out = out_eval / "qrels.tsv"
    gold_out = out_eval / "gold_answers.jsonl"
    ids_out = out_proc / "subset_query_ids.txt"
    pmids_out = out_proc / "pmids.txt"
    meta_out = out_proc / "subset_meta.json"

    all_pmids: Set[str] = set()
    selected_ids: List[str] = []
    type_counts = Counter()

    with (
        q_out.open("w", encoding="utf-8") as fq,
        qrels_out.open("w", encoding="utf-8") as fr,
        gold_out.open("w", encoding="utf-8") as fg,
    ):
        fr.write("query_id\tdoc_id\trelevance\n")

        for q in selected:
            qid = str(q["id"])
            qtext = str(q.get("body") or "").strip()
            qtype = normalize_question_type(str(q.get("type") or ""))

            docs = normalize_documents(q.get("documents", []) or [])
            snippets = normalize_snippets(q.get("snippets", []) or [])

            selected_ids.append(qid)
            type_counts[qtype] += 1
            all_pmids.update(d.replace("pmid:", "") for d in docs)

            fq.write(json.dumps({
                "query_id": qid,
                "text": qtext,
                "type": qtype,
            }, ensure_ascii=False) + "\n")

            for did in docs:
                fr.write(f"{qid}\t{did}\t1\n")

            fg.write(json.dumps({
                "query_id": qid,
                "question": qtext,
                "type": qtype,
                "documents": docs,
                "snippets": snippets,
                "exact_answer": q.get("exact_answer"),
                "ideal_answer": q.get("ideal_answer"),
            }, ensure_ascii=False) + "\n")

    ids_out.write_text("\n".join(selected_ids) + "\n", encoding="utf-8")
    pmids_out.write_text("\n".join(sorted(all_pmids)) + "\n", encoding="utf-8")

    meta = {
        "source_file": str(raw_path),
        "sha256": sha256_file(raw_path),
        "sampling": {
            "mode": "explicit_per_type" if explicit_counts is not None else "stratified_proportional",
            "n_total": args.n_total,
            "per_type_counts": explicit_counts,
            "min_per_type": args.min_per_type if explicit_counts is None else None,
            "seed": args.seed,
        },
        "num_selected_questions": len(selected),
        "type_counts": dict(type_counts),
        "num_unique_pmids": len(all_pmids),
    }
    meta_out.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"Wrote {q_out}")
    print(f"Wrote {qrels_out}")
    print(f"Wrote {gold_out}")
    print(f"Wrote {ids_out}")
    print(f"Wrote {pmids_out} ({len(all_pmids)} PMIDs)")
    print(f"Wrote {meta_out}")


if __name__ == "__main__":
    main()