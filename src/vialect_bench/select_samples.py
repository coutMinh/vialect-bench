from __future__ import annotations

import argparse
import csv
import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any

from datasets import load_dataset


DATASETS = {
    "sentiment": {
        "dataset_id": "tridm/UIT-VSMEC",
        "split": "train",
        "label_field": "Emotion",
    },
    "nli": {
        "dataset_id": "uitnlp/ViANLI",
        "split": "train",
        "label_field": "label",
    },
    "qa": {
        "dataset_id": "taidng/UIT-ViQuAD2.0",
        "split": "train",
        "label_field": "is_impossible",
    },
}


MAX_SELECTION_POOL = 3000
TOKEN_RE = re.compile(r"\w+", flags=re.UNICODE)


def to_builtin(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): to_builtin(item) for key, item in value.items()}
    if isinstance(value, list):
        return [to_builtin(item) for item in value]
    if isinstance(value, tuple):
        return [to_builtin(item) for item in value]
    return value


def get_label(task: str, row: dict[str, Any]) -> str:
    if task == "qa":
        return "unanswerable" if row.get("is_impossible") else "answerable"
    return str(row[DATASETS[task]["label_field"]])


def tokens(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def token_set(text: str) -> set[str]:
    return set(tokens(text))


def jaccard(left: set[str], right: set[str]) -> float:
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def selection_text(task: str, row: dict[str, Any]) -> str:
    if task == "sentiment":
        return str(row.get("Sentence", ""))
    if task == "nli":
        return f"{row.get('premise', '')}\n{row.get('hypothesis', '')}"
    if task == "qa":
        return str(row.get("question", ""))
    raise ValueError(f"Unsupported task: {task}")


def passes_filter(task: str, row: dict[str, Any]) -> bool:
    if task == "sentiment":
        sentence = str(row.get("Sentence", ""))
        return len(tokens(sentence)) >= 5

    if task == "qa":
        question = str(row.get("question", ""))
        answers = row.get("answers") or {}
        answer_texts = answers.get("text") or []
        question_len = len(tokens(question))
        return (
            not row.get("is_impossible")
            and 6 <= question_len <= 25
            and len(answer_texts) > 0
            and all(len(tokens(str(answer))) <= 12 for answer in answer_texts)
        )

    return True


def select_least_similar(rows: list[dict[str, Any]], task: str, sample_size: int, seed: int) -> list[int]:
    if sample_size > len(rows):
        raise SystemExit(
            f"Task '{task}' has only {len(rows)} rows after filtering, "
            f"but --sample-size is {sample_size}."
        )

    rng = random.Random(seed)
    row_features = [(row["_source_index"], token_set(selection_text(task, row))) for row in rows]
    first_position = rng.randrange(len(rows))
    selected_positions = [first_position]
    remaining_positions = set(range(len(rows)))
    remaining_positions.remove(first_position)

    while len(selected_positions) < sample_size:
        best_position = None
        best_similarity = float("inf")

        for position in remaining_positions:
            candidate_tokens = row_features[position][1]
            max_similarity = max(
                jaccard(candidate_tokens, row_features[selected_position][1])
                for selected_position in selected_positions
            )
            if max_similarity < best_similarity:
                best_similarity = max_similarity
                best_position = position

        assert best_position is not None
        selected_positions.append(best_position)
        remaining_positions.remove(best_position)

    return [row_features[position][0] for position in selected_positions]


def normalize_row(task: str, dataset_id: str, source_index: int, row: dict[str, Any]) -> dict:
    row = to_builtin(row)
    return {"_task": task, "_source_dataset": dataset_id, "_source_index": source_index, **row}


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_summary_csv(path: Path, summary_rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "task",
                "dataset",
                "num_samples",
                "num_candidates",
                "selection_pool",
                "seed",
                "labels",
            ],
        )
        writer.writeheader()
        writer.writerows(summary_rows)


def select_for_task(
    task: str,
    sample_size: int,
    output_dir: Path,
    max_source_rows: int | None,
    split_override: str | None,
    seed: int,
) -> dict:
    if task not in DATASETS:
        known = ", ".join(sorted(DATASETS))
        raise SystemExit(f"Unknown task '{task}'. Known tasks: {known}")

    task_config = DATASETS[task]
    dataset_id = task_config["dataset_id"]
    split = split_override or task_config["split"]
    source_split = f"{split}[:{max_source_rows}]" if max_source_rows else split

    dataset = load_dataset(dataset_id, split=source_split)
    candidates = [
        normalize_row(task, dataset_id, source_index, dataset[source_index])
        for source_index in range(len(dataset))
        if passes_filter(task, dataset[source_index])
    ]
    selection_pool = candidates
    if len(selection_pool) > MAX_SELECTION_POOL:
        rng = random.Random(seed)
        selection_pool = rng.sample(selection_pool, MAX_SELECTION_POOL)

    source_indices = select_least_similar(selection_pool, task, sample_size, seed)
    selected = [row for row in candidates if row["_source_index"] in set(source_indices)]
    selected.sort(key=lambda row: source_indices.index(row["_source_index"]))

    out_path = output_dir / f"{task}_{sample_size}.jsonl"
    write_jsonl(out_path, selected)

    labels = Counter(get_label(task, row) for row in selected)
    return {
        "task": task,
        "dataset": dataset_id,
        "num_samples": len(selected),
        "num_candidates": len(candidates),
        "selection_pool": len(selection_pool),
        "seed": seed,
        "labels": json.dumps(dict(labels), ensure_ascii=False),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Select low-similarity pilot samples for Vialect-Bench tasks."
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=["sentiment", "nli", "qa"],
        help="Task names to sample: sentiment, nli, qa.",
    )
    parser.add_argument("--sample-size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=Path("data/pilot_samples"))
    parser.add_argument(
        "--max-source-rows",
        type=int,
        default=None,
        help="Optional cap for quick smoke tests.",
    )
    parser.add_argument("--split", default=None, help="Override configured source split.")
    args = parser.parse_args()

    summaries = [
        select_for_task(
            task=task,
            sample_size=args.sample_size,
            output_dir=args.output_dir,
            max_source_rows=args.max_source_rows,
            split_override=args.split,
            seed=args.seed,
        )
        for task in args.tasks
    ]
    write_summary_csv(args.output_dir / "selection_summary.csv", summaries)
    print(json.dumps(summaries, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
