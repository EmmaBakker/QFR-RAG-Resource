#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generic inter-judge reliability script.

It supports:
  - scalar metrics, e.g. nugget recall, faithfulness, relevance
  - categorical metrics, e.g. correctness labels, taxonomy labels, safety labels

Example, direct use:
  python scripts/inter_judge_reliability.py \
    --judge-a runs/model_x/eval_gpt/per_example_eval.jsonl \
    --judge-b runs/model_x/eval_claude/per_example_eval.jsonl \
    --scalar-metrics nuggets.macro_avg_strict_recall nuggets.macro_avg_soft_recall \
    --categorical-metrics correctness.judge_correct taxonomy.case \
    --out-dir outputs/inter_judge_reliability/model_x
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


# ---------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------

def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                obj = json.loads(line)
                if isinstance(obj, dict):
                    rows.append(obj)
    return rows


def read_json(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)

    if isinstance(obj, list):
        return [x for x in obj if isinstance(x, dict)]

    if isinstance(obj, dict):
        # Common wrapper keys used by evaluation scripts.
        for key in ["examples", "rows", "items", "data", "per_example", "per_example_eval"]:
            val = obj.get(key)
            if isinstance(val, list):
                return [x for x in val if isinstance(x, dict)]
            if isinstance(val, dict):
                rows = []
                for rid, rec in val.items():
                    if isinstance(rec, dict):
                        row = dict(rec)
                        row.setdefault("id", rid)
                        rows.append(row)
                return rows

        # Otherwise treat a dictionary of records as {id: record}.
        rows = []
        for rid, rec in obj.items():
            if isinstance(rec, dict):
                row = dict(rec)
                row.setdefault("id", rid)
                rows.append(row)
        if rows:
            return rows

    raise ValueError(f"Unsupported JSON structure: {path}")


def read_csv(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def find_default_input_file(directory: Path) -> Path:
    candidates = [
        "per_example_eval.jsonl",
        "per_example_eval.json",
        "adversarial_eval_per_item.csv",
        "results.jsonl",
        "results.json",
        "results.csv",
    ]
    for name in candidates:
        p = directory / name
        if p.exists():
            return p

    # Fall back to the first plausible file in the directory.
    for suffix in ["*.jsonl", "*.json", "*.csv"]:
        matches = sorted(directory.glob(suffix))
        if matches:
            return matches[0]

    raise FileNotFoundError(f"No JSONL, JSON, or CSV file found in {directory}")


def load_rows(path_like: str) -> List[Dict[str, Any]]:
    path = Path(path_like)
    if path.is_dir():
        path = find_default_input_file(path)

    if not path.exists():
        raise FileNotFoundError(path)

    if path.suffix.lower() == ".jsonl":
        return read_jsonl(path)
    if path.suffix.lower() == ".json":
        return read_json(path)
    if path.suffix.lower() == ".csv":
        return read_csv(path)

    raise ValueError(f"Unsupported file extension for {path}")


def rows_by_id(rows: Sequence[Dict[str, Any]], id_field: str) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        rid = get_value(row, id_field)
        if rid is None:
            continue
        rid_s = str(rid).strip()
        if rid_s:
            out[rid_s] = row
    return out


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    columns: List[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


# ---------------------------------------------------------------------
# Value handling
# ---------------------------------------------------------------------

def get_value(row: Dict[str, Any], path: str) -> Any:
    """
    Read either a direct field name or a nested dot path.

    If a CSV column literally contains dots, the direct column is preferred.
    """
    if path in row:
        return row[path]

    cur: Any = row
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def safe_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    if isinstance(x, bool):
        return float(x)
    try:
        if isinstance(x, str) and x.strip().lower() in {"", "none", "null", "nan"}:
            return None
        v = float(x)
        if math.isnan(v):
            return None
        return v
    except Exception:
        return None


def normalize_label(x: Any) -> Optional[str]:
    if x is None:
        return None
    if isinstance(x, float) and math.isnan(x):
        return None
    if isinstance(x, bool):
        return "true" if x else "false"

    s = str(x).strip()
    if not s or s.lower() in {"none", "null", "nan"}:
        return None

    lower = s.lower()
    if lower in {"true", "1", "yes", "y"}:
        return "true"
    if lower in {"false", "0", "no", "n"}:
        return "false"
    return s


def mean(values: Iterable[float]) -> Optional[float]:
    vals = [float(v) for v in values if v is not None and not math.isnan(float(v))]
    return sum(vals) / len(vals) if vals else None


def round_or_none(x: Optional[float], digits: int = 4) -> Optional[float]:
    return None if x is None else round(float(x), digits)


# ---------------------------------------------------------------------
# Reliability metrics
# ---------------------------------------------------------------------

def pearson(xs: List[float], ys: List[float]) -> Optional[float]:
    if len(xs) < 2:
        return None

    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den_x = math.sqrt(sum((x - mx) ** 2 for x in xs))
    den_y = math.sqrt(sum((y - my) ** 2 for y in ys))

    if den_x == 0 or den_y == 0:
        return None
    return num / (den_x * den_y)


def rankdata(values: List[float]) -> List[float]:
    """
    Average ranks for ties. Ranks start at 1.
    """
    indexed = sorted(enumerate(values), key=lambda x: x[1])
    ranks = [0.0] * len(values)
    i = 0

    while i < len(indexed):
        j = i
        while j + 1 < len(indexed) and indexed[j + 1][1] == indexed[i][1]:
            j += 1

        avg_rank = (i + 1 + j + 1) / 2
        for k in range(i, j + 1):
            original_index = indexed[k][0]
            ranks[original_index] = avg_rank
        i = j + 1

    return ranks


def spearman(xs: List[float], ys: List[float]) -> Optional[float]:
    if len(xs) < 2:
        return None
    return pearson(rankdata(xs), rankdata(ys))


def mean_abs_diff(xs: List[float], ys: List[float]) -> Optional[float]:
    if not xs:
        return None
    return sum(abs(x - y) for x, y in zip(xs, ys)) / len(xs)


def cohen_kappa(y_a: List[str], y_b: List[str]) -> Optional[float]:
    if not y_a or len(y_a) != len(y_b):
        return None

    labels = sorted(set(y_a) | set(y_b))
    if len(labels) < 2:
        return None

    n = len(y_a)
    observed = sum(a == b for a, b in zip(y_a, y_b)) / n

    counts_a = {lab: 0 for lab in labels}
    counts_b = {lab: 0 for lab in labels}
    for a, b in zip(y_a, y_b):
        counts_a[a] += 1
        counts_b[b] += 1

    expected = sum((counts_a[lab] / n) * (counts_b[lab] / n) for lab in labels)
    if expected == 1:
        return None

    return (observed - expected) / (1 - expected)


def compare_scalar(
    rows_a: Dict[str, Dict[str, Any]],
    rows_b: Dict[str, Dict[str, Any]],
    metric_path: str,
    large_diff_threshold: float,
    restrict_ids: Optional[set[str]] = None,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    common_ids = sorted(set(rows_a) & set(rows_b))
    if restrict_ids is not None:
        common_ids = [rid for rid in common_ids if rid in restrict_ids]

    xs: List[float] = []
    ys: List[float] = []
    disagreements: List[Dict[str, Any]] = []

    for rid in common_ids:
        x = safe_float(get_value(rows_a[rid], metric_path))
        y = safe_float(get_value(rows_b[rid], metric_path))
        if x is None or y is None:
            continue

        xs.append(x)
        ys.append(y)

        diff = x - y
        if abs(diff) >= large_diff_threshold:
            disagreements.append({
                "id": rid,
                "metric": metric_path,
                "judge_a_value": x,
                "judge_b_value": y,
                "difference_a_minus_b": diff,
                "absolute_difference": abs(diff),
            })

    diff_values = [x - y for x, y in zip(xs, ys)]

    result = {
        "n": len(xs),
        "judge_a_mean": round_or_none(mean(xs)),
        "judge_b_mean": round_or_none(mean(ys)),
        "mean_difference_a_minus_b": round_or_none(mean(diff_values)),
        "mean_absolute_difference": round_or_none(mean_abs_diff(xs, ys)),
        "pearson_r": round_or_none(pearson(xs, ys)),
        "spearman_rho": round_or_none(spearman(xs, ys)),
        "judge_a_higher_n": sum(1 for x, y in zip(xs, ys) if x > y),
        "judge_b_higher_n": sum(1 for x, y in zip(xs, ys) if y > x),
        "equal_n": sum(1 for x, y in zip(xs, ys) if x == y),
    }
    return result, disagreements


def compare_categorical(
    rows_a: Dict[str, Dict[str, Any]],
    rows_b: Dict[str, Dict[str, Any]],
    metric_path: str,
    restrict_ids: Optional[set[str]] = None,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    common_ids = sorted(set(rows_a) & set(rows_b))
    if restrict_ids is not None:
        common_ids = [rid for rid in common_ids if rid in restrict_ids]

    y_a: List[str] = []
    y_b: List[str] = []
    disagreements: List[Dict[str, Any]] = []

    for rid in common_ids:
        a = normalize_label(get_value(rows_a[rid], metric_path))
        b = normalize_label(get_value(rows_b[rid], metric_path))
        if a is None or b is None:
            continue

        y_a.append(a)
        y_b.append(b)

        if a != b:
            disagreements.append({
                "id": rid,
                "metric": metric_path,
                "judge_a_label": a,
                "judge_b_label": b,
            })

    labels = sorted(set(y_a) | set(y_b))
    n = len(y_a)

    if n == 0:
        return {
            "n": 0,
            "exact_agreement": None,
            "disagreement_rate": None,
            "cohen_kappa": None,
            "pabak_2po_minus1": None,
            "labels": labels,
            "single_label_case": False,
            "confusion": {},
        }, disagreements

    exact = sum(a == b for a, b in zip(y_a, y_b)) / n
    kappa = cohen_kappa(y_a, y_b)
    confusion = {a: {b: 0 for b in labels} for a in labels}
    for a, b in zip(y_a, y_b):
        confusion[a][b] += 1

    result = {
        "n": n,
        "exact_agreement": round_or_none(exact),
        "disagreement_rate": round_or_none(1 - exact),
        "cohen_kappa": round_or_none(kappa),
        "pabak_2po_minus1": round_or_none(2 * exact - 1),
        "labels": labels,
        "single_label_case": len(labels) < 2,
        "confusion": confusion,
    }
    return result, disagreements


def make_group_ids(
    rows_a: Dict[str, Dict[str, Any]],
    rows_b: Dict[str, Dict[str, Any]],
    subgroup_field: str,
) -> Dict[str, set[str]]:
    groups: Dict[str, set[str]] = defaultdict(set)

    for rid in sorted(set(rows_a) & set(rows_b)):
        value = get_value(rows_a[rid], subgroup_field)
        if value is None:
            value = get_value(rows_b[rid], subgroup_field)
        label = normalize_label(value)
        if label is not None:
            groups[label].add(rid)

    return groups


# ---------------------------------------------------------------------
# Run logic
# ---------------------------------------------------------------------

def parse_metric_list(value: str) -> List[str]:
    if not value:
        return []
    value = value.replace(";", ",")
    return [x.strip() for x in value.split(",") if x.strip()]


def load_manifest(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def direct_manifest_from_args(args: argparse.Namespace) -> List[Dict[str, str]]:
    if not args.judge_a or not args.judge_b:
        raise ValueError("Provide either --manifest or both --judge-a and --judge-b.")

    return [{
        "name": args.name,
        "judge_a": args.judge_a,
        "judge_b": args.judge_b,
        "id_field": args.id_field,
        "scalar_metrics": ",".join(args.scalar_metrics or []),
        "categorical_metrics": ",".join(args.categorical_metrics or []),
        "subgroup_fields": ",".join(args.subgroup_fields or []),
    }]


def add_summary_row(
    rows: List[Dict[str, Any]],
    *,
    pair_name: str,
    metric_type: str,
    metric: str,
    result: Dict[str, Any],
    subgroup_field: Optional[str] = None,
    subgroup_value: Optional[str] = None,
) -> None:
    row = {
        "pair": pair_name,
        "metric_type": metric_type,
        "metric": metric,
        "subgroup_field": subgroup_field,
        "subgroup_value": subgroup_value,
        "n": result.get("n"),
    }

    for key, value in result.items():
        if key in {"confusion", "labels"}:
            row[key] = json.dumps(value, ensure_ascii=False)
        else:
            row[key] = value

    rows.append(row)


def run_pair(
    *,
    pair: Dict[str, str],
    out_rows: List[Dict[str, Any]],
    scalar_disagreements: List[Dict[str, Any]],
    categorical_disagreements: List[Dict[str, Any]],
    warnings: List[str],
    large_diff_threshold: float,
) -> None:
    pair_name = pair.get("name") or "judge_pair"
    id_field = pair.get("id_field") or "id"
    scalar_metrics = parse_metric_list(pair.get("scalar_metrics", ""))
    categorical_metrics = parse_metric_list(pair.get("categorical_metrics", ""))
    subgroup_fields = parse_metric_list(pair.get("subgroup_fields", ""))

    if not scalar_metrics and not categorical_metrics:
        warnings.append(f"{pair_name}: no scalar_metrics or categorical_metrics specified; skipped.")
        return

    try:
        rows_a = rows_by_id(load_rows(pair["judge_a"]), id_field)
        rows_b = rows_by_id(load_rows(pair["judge_b"]), id_field)
    except Exception as e:
        warnings.append(f"{pair_name}: could not load judge files: {repr(e)}")
        return

    common_n = len(set(rows_a) & set(rows_b))
    if common_n == 0:
        warnings.append(f"{pair_name}: no shared IDs between judge files.")
        return

    for metric in scalar_metrics:
        result, disagreements = compare_scalar(rows_a, rows_b, metric, large_diff_threshold)
        add_summary_row(out_rows, pair_name=pair_name, metric_type="scalar", metric=metric, result=result)

        for d in disagreements:
            d.update({"pair": pair_name})
            scalar_disagreements.append(d)

        for subgroup_field in subgroup_fields:
            groups = make_group_ids(rows_a, rows_b, subgroup_field)
            for subgroup_value, ids in sorted(groups.items()):
                sub_result, _ = compare_scalar(rows_a, rows_b, metric, large_diff_threshold, restrict_ids=ids)
                if sub_result.get("n", 0) > 0:
                    add_summary_row(
                        out_rows,
                        pair_name=pair_name,
                        metric_type="scalar",
                        metric=metric,
                        result=sub_result,
                        subgroup_field=subgroup_field,
                        subgroup_value=subgroup_value,
                    )

    for metric in categorical_metrics:
        result, disagreements = compare_categorical(rows_a, rows_b, metric)
        add_summary_row(out_rows, pair_name=pair_name, metric_type="categorical", metric=metric, result=result)

        for d in disagreements:
            d.update({"pair": pair_name})
            categorical_disagreements.append(d)

        for subgroup_field in subgroup_fields:
            groups = make_group_ids(rows_a, rows_b, subgroup_field)
            for subgroup_value, ids in sorted(groups.items()):
                sub_result, _ = compare_categorical(rows_a, rows_b, metric, restrict_ids=ids)
                if sub_result.get("n", 0) > 0:
                    add_summary_row(
                        out_rows,
                        pair_name=pair_name,
                        metric_type="categorical",
                        metric=metric,
                        result=sub_result,
                        subgroup_field=subgroup_field,
                        subgroup_value=subgroup_value,
                    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute generic inter-judge reliability for two judge-output files."
    )

    parser.add_argument("--manifest", help="CSV manifest with one or more judge-output pairs.")

    parser.add_argument("--judge-a", help="Path to first judge output file or directory.")
    parser.add_argument("--judge-b", help="Path to second judge output file or directory.")
    parser.add_argument("--name", default="judge_pair", help="Name for the direct judge pair.")

    parser.add_argument("--id-field", default="id", help="ID field shared by both judge outputs.")
    parser.add_argument("--scalar-metrics", nargs="*", default=[], help="Scalar metric paths to compare.")
    parser.add_argument("--categorical-metrics", nargs="*", default=[], help="Categorical metric paths to compare.")
    parser.add_argument("--subgroup-fields", nargs="*", default=[], help="Optional subgroup fields, e.g. question_type.")

    parser.add_argument("--out-dir", default="outputs/inter_judge_reliability")
    parser.add_argument("--large-diff-threshold", type=float, default=0.25)

    args = parser.parse_args()

    if args.manifest:
        pairs = load_manifest(Path(args.manifest))
    else:
        pairs = direct_manifest_from_args(args)

    out_dir = Path(args.out_dir)
    summary_rows: List[Dict[str, Any]] = []
    scalar_disagreements: List[Dict[str, Any]] = []
    categorical_disagreements: List[Dict[str, Any]] = []
    warnings: List[str] = []

    for pair in pairs:
        run_pair(
            pair=pair,
            out_rows=summary_rows,
            scalar_disagreements=scalar_disagreements,
            categorical_disagreements=categorical_disagreements,
            warnings=warnings,
            large_diff_threshold=args.large_diff_threshold,
        )

    write_csv(out_dir / "inter_judge_reliability_summary.csv", summary_rows)
    write_csv(out_dir / "scalar_large_disagreements.csv", scalar_disagreements)
    write_csv(out_dir / "categorical_disagreements.csv", categorical_disagreements)
    write_json(out_dir / "inter_judge_reliability_summary.json", summary_rows)
    write_json(out_dir / "warnings.json", warnings)

    print("Done.")
    print(f"Wrote outputs to: {out_dir}")
    print(f"Summary rows: {len(summary_rows)}")
    print(f"Scalar large disagreements: {len(scalar_disagreements)}")
    print(f"Categorical disagreements: {len(categorical_disagreements)}")
    if warnings:
        print(f"Warnings: {len(warnings)}. See {out_dir / 'warnings.json'}")


if __name__ == "__main__":
    main()