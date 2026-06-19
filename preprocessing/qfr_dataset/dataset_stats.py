#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from collections import Counter


def load_jsonl(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    encodings_to_try = ["utf-8", "utf-8-sig", "cp1252", "latin-1"]

    last_error = None

    for encoding in encodings_to_try:
        try:
            records = []
            with path.open("r", encoding=encoding) as f:
                for line in f:
                    if line.strip():
                        records.append(json.loads(line))
            return records
        except UnicodeDecodeError as e:
            last_error = e

    raise UnicodeDecodeError(
        last_error.encoding,
        last_error.object,
        last_error.start,
        last_error.end,
        f"Could not decode {path} with encodings: {encodings_to_try}",
    )


def count_task_b_slots(row):
    """
    Your Task B/C format:
      "required_slots": 2
    """
    required_slots = row.get("required_slots")

    if isinstance(required_slots, int):
        return required_slots

    if isinstance(required_slots, list):
        return len(required_slots)

    if isinstance(required_slots, dict):
        return len(required_slots)

    # Fallback: count fields such as slot_A, slot_B, slot_C
    return sum(
        1 for key in row.keys()
        if key.startswith("slot_") and key != "slot_nuggets"
    )


def count_task_b_nuggets(row):
    """
    Your Task B/C format:
      "slot_nuggets": {
          "slot_A": [...],
          "slot_B": [...]
      }
    """
    slot_nuggets = row.get("slot_nuggets", {})

    if not isinstance(slot_nuggets, dict):
        return 0

    total = 0
    for nuggets in slot_nuggets.values():
        if isinstance(nuggets, list):
            total += len(nuggets)
        elif isinstance(nuggets, dict):
            total += len(nuggets)
        elif isinstance(nuggets, str) and nuggets.strip():
            total += 1

    return total


def print_task_a_stats(records):
    print("\n=== Task A statistics ===")
    print(f"Total questions: {len(records)}")

    # Optional: print taxonomy/category counts if present
    for field in ["task", "category", "type", "question_type", "taxonomy"]:
        values = [r.get(field) for r in records if r.get(field)]
        if values:
            print(f"\n{field} counts:")
            for label, count in Counter(values).most_common():
                print(f"  {label}: {count}")


def print_task_b_stats(records):
    slot_counts = [count_task_b_slots(r) for r in records]
    nugget_counts = [count_task_b_nuggets(r) for r in records]

    total_questions = len(records)
    total_slots = sum(slot_counts)
    total_nuggets = sum(nugget_counts)

    print("\n=== Task B statistics ===")
    print(f"Total questions: {total_questions}")

    print(f"\nTotal slots: {total_slots}")
    print(f"Average slots per question: {total_slots / total_questions:.2f}")

    print(f"\nTotal nuggets: {total_nuggets}")
    print(f"Average nuggets per question: {total_nuggets / total_questions:.2f}")

    if total_slots > 0:
        print(f"Average nuggets per slot: {total_nuggets / total_slots:.2f}")

    print("\nSlot count distribution:")
    for n_slots, count in Counter(slot_counts).most_common():
        print(f"  {n_slots} slots: {count} questions")

    print("\nNugget count distribution:")
    for n_nuggets, count in Counter(nugget_counts).most_common():
        print(f"  {n_nuggets} nuggets: {count} questions")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_a", required=True, help="Path to Task A JSONL file")
    parser.add_argument("--task_b", required=True, help="Path to Task B/C JSONL file")
    args = parser.parse_args()

    task_a = load_jsonl(Path(args.task_a))
    task_b = load_jsonl(Path(args.task_b))

    print_task_a_stats(task_a)
    print_task_b_stats(task_b)


if __name__ == "__main__":
    main()