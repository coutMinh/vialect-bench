from __future__ import annotations

import argparse
import csv
import json
import random
import re
import unicodedata
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
    "mcqa": {
        "dataset_id": "uitnlp/vimmrc2.0",
        "split": "train",
        "label_field": "answers",
    },
}


MAX_SELECTION_POOL = 3000
MAX_MCQA_QUESTIONS_PER_TOPIC = 2
TOKEN_RE = re.compile(r"\w+", flags=re.UNICODE)
SENTIMENT_VULGAR_WORDS = [
    "đéo", "đụ", "lồn", "địt", "đm", "dm", "dmm", "cc", "fuck", "shit",
    "vãi", "vãi_chưởng", "chảnh chó", "con mẹ nó",
]


def is_vulgar(text: str) -> bool:
    lower = text.lower()
    for word in SENTIMENT_VULGAR_WORDS:
        i = lower.find(word)
        while i != -1:
            before_ok = i == 0 or not lower[i - 1].isalnum()
            after_ok = i + len(word) >= len(lower) or not lower[i + len(word)].isalnum()
            if before_ok and after_ok:
                return True
            i = lower.find(word, i + 1)
    return False
MCQA_TITLE_EXCLUDE_KEYWORDS = {
    "bac",
    "chu dong tu",
    "luom",
    "thanh giong",
    "son tinh",
    "thuy tinh",
    "y ec xanh",
    "mtao",
    "quan am",
    "ra ma",
    "xi ta",
    "ga li le",
    "nguyen",
    "tran dai nghia",
    "hai ba trung",
    "con rong chau tien",
    "bop nat qua cam",
    "ong trang",
    "trang",
}
MCQA_GENERAL_TITLE_KEYWORDS = {
    "bon mua",
    "cay",
    "chim",
    "ga",
    "gau",
    "ca",
    "cau chuyen",
    "con",
    "chiec",
    "chuyen",
    "mua",
    "hoa",
    "song",
    "bien",
    "rung",
    "dong",
    "dien thoai",
    "doi giay",
    "mau giay",
    "ban tay",
    "ban tin",
    "bai hat",
    "bai hoc",
    "bai tap",
    "bai kiem tra",
    "bai tho",
    "bai doc",
    "dong ho",
    "cai cau",
    "cua tung",
    "cuc nuoc da",
    "cuon so tay",
    "nang",
    "dat",
    "dem",
    "dua ghe",
    "huong",
    "mat troi",
    "bau troi",
    "canh dieu",
    "hat mam",
    "truong",
    "lop",
    "cho",
    "que",
    "kho bau",
    "phong nha",
    "thuoc la",
}


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
    if task == "mcqa":
        return str(row.get("answer", ""))
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
    if task == "mcqa":
        return f"{row.get('title', '')}\n{row.get('question', '')}"
    raise ValueError(f"Unsupported task: {task}")


def passes_filter(task: str, row: dict[str, Any]) -> bool:
    if task == "sentiment":
        sentence = str(row.get("Sentence", ""))
        return len(tokens(sentence)) >= 5 and not is_vulgar(sentence)

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

    if task == "mcqa":
        return is_valid_mcqa_question(row)

    return True


def strip_accents(text: str) -> str:
    normalized = unicodedata.normalize("NFD", text)
    return "".join(ch for ch in normalized if unicodedata.category(ch) != "Mn")


def normalized_title(title: str) -> str:
    text = str(title).replace("\u200b", " ").replace("-", " ")
    text = strip_accents(text).lower()
    return re.sub(r"\s+", " ", text).strip()


def is_general_mcqa_topic(row: dict[str, Any]) -> bool:
    title = normalized_title(row.get("title", ""))
    topic_text = normalized_title(
        " ".join(
            [
                str(row.get("title", "")),
                str(row.get("article", "")),
                " ".join(str(question) for question in row.get("questions", []) or []),
            ]
        )
    )
    return (
        mcqa_grade_ok(row.get("grade"))
        and not any(keyword in topic_text for keyword in MCQA_TITLE_EXCLUDE_KEYWORDS)
        and any(keyword in title for keyword in MCQA_GENERAL_TITLE_KEYWORDS)
    )


def mcqa_grade_ok(grade: Any) -> bool:
    try:
        return int(str(grade).strip()) <= 3
    except ValueError:
        return False


def is_valid_mcqa_question(row: dict[str, Any]) -> bool:
    return (
        is_general_mcqa_topic(row)
        and len(tokens(str(row.get("article", "")))) >= 40
        and len(tokens(str(row.get("question", "")))) >= 5
        and isinstance(row.get("options"), list)
        and len(row.get("options") or []) == 4
        and str(row.get("answer", "")).strip() in {"A", "B", "C", "D"}
    )


def sample_key(row: dict[str, Any]) -> Any:
    return row.get("_sample_id", row["_source_index"])


def select_least_similar(
    rows: list[dict[str, Any]],
    task: str,
    sample_size: int,
    seed: int,
    max_per_topic: int | None = None,
) -> list[Any]:
    if sample_size > len(rows):
        raise SystemExit(
            f"Task '{task}' has only {len(rows)} rows after filtering, "
            f"but --sample-size is {sample_size}."
        )

    rng = random.Random(seed)
    row_features = [(sample_key(row), token_set(selection_text(task, row))) for row in rows]
    first_position = rng.randrange(len(rows))
    selected_positions = [first_position]
    topic_counts = Counter()
    topic_counts[rows[first_position].get("_topic_source_index", rows[first_position]["_source_index"])] += 1
    remaining_positions = set(range(len(rows)))
    remaining_positions.remove(first_position)

    while len(selected_positions) < sample_size:
        best_position = None
        best_similarity = float("inf")

        for position in remaining_positions:
            topic_key = rows[position].get("_topic_source_index", rows[position]["_source_index"])
            if max_per_topic is not None and topic_counts[topic_key] >= max_per_topic:
                continue
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
        topic_key = rows[best_position].get("_topic_source_index", rows[best_position]["_source_index"])
        topic_counts[topic_key] += 1
        remaining_positions.remove(best_position)

    return [row_features[position][0] for position in selected_positions]


def normalize_row(task: str, dataset_id: str, source_index: int, row: dict[str, Any]) -> dict:
    row = to_builtin(row)
    return {"_task": task, "_source_dataset": dataset_id, "_source_index": source_index, **row}


def build_mcqa_question_rows(dataset, dataset_id: str) -> list[dict[str, Any]]:
    rows = []
    for topic_index in range(len(dataset)):
        topic = to_builtin(dataset[topic_index])
        if not is_general_mcqa_topic(topic):
            continue

        questions = topic.get("questions") or []
        options = topic.get("options") or []
        answers = topic.get("answers") or []
        types = topic.get("types") or []
        for question_index, (question, option_set, answer) in enumerate(zip(questions, options, answers)):
            question_row = {
                "_task": "mcqa",
                "_source_dataset": dataset_id,
                "_source_index": topic_index,
                "_sample_id": f"{topic_index}:{question_index}",
                "_topic_source_index": topic_index,
                "_question_index": question_index,
                "title": topic.get("title"),
                "article": topic.get("article"),
                "grade": topic.get("grade"),
                "author": topic.get("author"),
                "files": topic.get("files"),
                "isProse": topic.get("isProse"),
                "question": question,
                "options": option_set,
                "answer": answer,
                "type": types[question_index] if question_index < len(types) else None,
            }
            if is_valid_mcqa_question(question_row):
                rows.append(question_row)
    return rows


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
    if task == "mcqa":
        candidates = build_mcqa_question_rows(dataset, dataset_id)
    else:
        candidates = [
            normalize_row(task, dataset_id, source_index, dataset[source_index])
            for source_index in range(len(dataset))
            if passes_filter(task, dataset[source_index])
        ]
    selection_pool = candidates
    if len(selection_pool) > MAX_SELECTION_POOL:
        rng = random.Random(seed)
        selection_pool = rng.sample(selection_pool, MAX_SELECTION_POOL)

    source_indices = select_least_similar(
        selection_pool,
        task,
        sample_size,
        seed,
        max_per_topic=MAX_MCQA_QUESTIONS_PER_TOPIC if task == "mcqa" else None,
    )
    selected = [row for row in candidates if sample_key(row) in set(source_indices)]
    selected.sort(key=lambda row: source_indices.index(sample_key(row)))

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
        default=["sentiment", "nli", "qa", "mcqa"],
        help="Task names to sample: sentiment, nli, qa, mcqa.",
    )
    parser.add_argument("--sample-size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=Path("data"))
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
