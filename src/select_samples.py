from __future__ import annotations

import argparse
import csv
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any

import openpyxl
from datasets import load_dataset


Row = dict[str, Any]
Trigger = dict[str, str]
TriggerIndex = dict[str, list[Row]]


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


MAX_MCQA_QUESTIONS_PER_TOPIC = 2
LEXICON_CATEGORY_PREFIXES = {
    "1. Interrogatives": "Interrogative",
    "2. Demonstratives & Deixis": "Demonstrative",
    "3. Negation & Function Words": "Negation/Function",
    "4. Discourse Particles": "Particle",
    "5. Connectives & Aspect Markers": "Connective/Aspect",
    "6. Pronouns": "Pronoun",
    "7. Kinship & Honorifics": "Kinship",
    "8. Predicate Vocabulary": "Predicate",
    "9. Templates & Idioms": "Template/Idiom",
}
TOTAL_LEXICON_CATEGORIES = len(LEXICON_CATEGORY_PREFIXES)
TOKEN_RE = re.compile(r"\w+", flags=re.UNICODE)
SENTIMENT_VULGAR_WORDS = [
    "đéo", "đụ", "lồn", "địt", "đm", "dm", "dmm", "cc", "fuck", "shit",
    "vãi", "vãi_chưởng", "chảnh chó", "con mẹ nó", "dume", "lol", "đũy", "lòn",
]
EMOJI_RE = re.compile(
    "["
    "\U0001f300-\U0001f5ff"
    "\U0001f600-\U0001f64f"
    "\U0001f680-\U0001f6ff"
    "\U0001f700-\U0001f77f"
    "\U0001f780-\U0001f7ff"
    "\U0001f800-\U0001f8ff"
    "\U0001f900-\U0001f9ff"
    "\U0001fa00-\U0001fa6f"
    "\U0001fa70-\U0001faff"
    "\u2600-\u27bf"
    ":)"
    ":("
    "]"
)


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


def has_emoji(text: str) -> bool:
    return bool(EMOJI_RE.search(text))


def load_lexicon(xlsx_path: Path) -> list[Row]:
    wb = openpyxl.load_workbook(xlsx_path, read_only=True)
    ws = wb[wb.sheetnames[0]]
    entries: list[Row] = []
    current_category = None

    for row in ws.iter_rows(min_row=3, values_only=True):
        col_b = row[1]
        if col_b is not None and isinstance(col_b, str):
            for prefix, category in LEXICON_CATEGORY_PREFIXES.items():
                if col_b.strip().startswith(prefix):
                    current_category = category
                    break
            continue

        standard = row[3]
        if standard is None or current_category is None:
            continue

        entries.append({"standard": str(standard).strip(), "category": current_category})

    wb.close()
    return entries


def build_trigger_index(entries: list[Row]) -> TriggerIndex:
    index: TriggerIndex = {}
    for entry in entries:
        standard = str(entry["standard"]).lower()
        forms = {standard, *(part.strip() for part in re.split(r"\s*/\s*", standard))}
        for form in forms:
            if len(form) >= 2:
                index.setdefault(form, []).append(entry)
    return index


def find_lexicon_triggers(
    text: str,
    trigger_index: TriggerIndex,
) -> list[Trigger]:
    text_lower = normalize_for_match(text)
    found: list[Trigger] = []
    seen = set()

    for key in sorted(trigger_index, key=len, reverse=True):
        pattern = term_pattern(key) if len(key) <= 4 else re.compile(re.escape(key), flags=re.IGNORECASE)
        if not pattern.search(text_lower):
            continue

        for entry in trigger_index[key]:
            dedup_key = (entry["standard"], entry["category"])
            if dedup_key in seen:
                continue
            seen.add(dedup_key)
            found.append(
                {
                    "standard": str(entry["standard"]),
                    "category": str(entry["category"]),
                    "matched_word": key,
                }
            )

    return found


def lexicon_hits_by_category(triggers: list[Trigger]) -> dict[str, list[str]]:
    hits: dict[str, list[str]] = {}
    for trigger in triggers:
        hits.setdefault(trigger["category"], []).append(trigger["matched_word"])
    return hits


def normalize_for_match(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def term_pattern(term: str) -> re.Pattern[str]:
    return re.compile(rf"(?<!\w){re.escape(term.lower())}(?!\w)", flags=re.IGNORECASE)


MCQA_TITLE_EXCLUDE_KEYWORDS = {
    "bac",
    "chu dong tu",
    "luom",
    "thanh giong",
    "y ec xanh",
    "mtao",
    "quan am",
    "ra ma",
    "xi ta",
    "ga li le",
    "nguyen",
    "tran dai nghia",
    "hai ba trung",
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


def get_label(task: str, row: Row) -> str:
    if task == "qa":
        return "unanswerable" if row.get("is_impossible") else "answerable"
    if task == "mcqa":
        return str(row.get("answer", ""))
    return str(row[DATASETS[task]["label_field"]])


def tokens(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def sample_text(task: str, row: Row) -> str:
    if task == "sentiment":
        return str(row.get("Sentence", ""))
    if task == "nli":
        return f"{row.get('premise', '')}\n{row.get('hypothesis', '')}"
    if task == "qa":
        return str(row.get("question", ""))
    if task == "mcqa":
        return str(row.get("question", ""))
    raise ValueError(f"Unsupported task: {task}")


def lexicon_score(
    task: str,
    row: Row,
    trigger_index: TriggerIndex,
) -> tuple[int, dict[str, Any]]:
    text = sample_text(task, row)
    triggers = find_lexicon_triggers(text, trigger_index)
    hits = lexicon_hits_by_category(triggers)
    category_count = len(hits)
    trigger_count = len(triggers)
    score = category_count

    metadata = {
        "lexicon_category_score": score,
        "lexicon_trigger_count": trigger_count,
        "lexicon_category_count": category_count,
        "lexicon_hits": hits,
        "dialect_triggers": triggers,
        "dialect_categories": sorted(hits),
    }
    return score, metadata


def passes_lexicon_filter(
    task: str,
    row: Row,
    min_lexicon_categories: int,
    min_tokens: int,
    trigger_index: TriggerIndex,
) -> bool:
    text = sample_text(task, row)
    _, metadata = lexicon_score(task, row, trigger_index)
    if len(tokens(text)) <= min_tokens:
        return False
    if metadata["lexicon_category_count"] < min_lexicon_categories:
        return False
    if task == "sentiment" and (has_emoji(text) or is_vulgar(text)):
        return False
    return True


def passes_filter(task: str, row: Row) -> bool:
    if task == "sentiment":
        sentence = str(row.get("Sentence", ""))
        return len(tokens(sentence)) >= 5 and not is_vulgar(sentence) and not has_emoji(sentence)

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


def is_general_mcqa_topic(row: Row) -> bool:
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
    )


def mcqa_grade_ok(grade: Any) -> bool:
    try:
        return int(str(grade).strip()) <= 3
    except ValueError:
        return False


def is_valid_mcqa_question(row: Row) -> bool:
    return (
        is_general_mcqa_topic(row)
        and len(tokens(str(row.get("article", "")))) >= 30
        and len(tokens(str(row.get("question", "")))) >= 5
        and isinstance(row.get("options"), list)
        and len(row.get("options") or []) == 4
        and str(row.get("answer", "")).strip() in {"A", "B", "C", "D"}
    )


def balanced_label_targets(rows: list[Row], task: str, sample_size: int) -> dict[str, int]:
    labels = sorted({get_label(task, row) for row in rows})
    label_pool_sizes = Counter(get_label(task, row) for row in rows)
    if not labels:
        return {}

    base = sample_size // len(labels)
    remainder = sample_size % len(labels)
    targets = {label: base + (1 if i < remainder else 0) for i, label in enumerate(labels)}

    overflow = 0
    for label in labels:
        if targets[label] > label_pool_sizes[label]:
            overflow += targets[label] - label_pool_sizes[label]
            targets[label] = label_pool_sizes[label]

    while overflow > 0:
        made_progress = False
        for label in labels:
            if targets[label] < label_pool_sizes[label]:
                targets[label] += 1
                overflow -= 1
                made_progress = True
                if overflow == 0:
                    break
        if not made_progress:
            break

    return targets


def select_high_lexicon_coverage(
    rows: list[Row],
    task: str,
    sample_size: int,
    min_lexicon_categories: int,
    min_tokens: int,
    trigger_index: TriggerIndex,
    max_per_topic: int | None = None,
) -> list[Row]:
    candidates = []
    for row in rows:
        text = sample_text(task, row)
        _, metadata = lexicon_score(task, row, trigger_index)
        if (
            len(tokens(text)) > min_tokens
            and metadata["lexicon_category_count"] >= min_lexicon_categories
            and not (task == "sentiment" and (has_emoji(text) or is_vulgar(text)))
        ):
            row["_lexicon_metadata"] = metadata
            candidates.append(row)
    if sample_size > len(candidates):
        raise SystemExit(
            f"Task '{task}' has only {len(candidates)} rows with enough lexicon categories, "
            f"but --sample-size is {sample_size}."
        )

    def ranking_key(row: Row) -> tuple[int, int, int]:
        return (
            row["_lexicon_metadata"]["lexicon_category_count"],
            row["_lexicon_metadata"]["lexicon_trigger_count"],
            len(tokens(sample_text(task, row))),
        )

    grouped: dict[str, list[Row]] = {}
    for row in candidates:
        grouped.setdefault(get_label(task, row), []).append(row)
    for label_rows in grouped.values():
        label_rows.sort(key=ranking_key, reverse=True)

    selected: list[Row] = []
    topic_counts = Counter()
    label_targets = balanced_label_targets(candidates, task, sample_size)

    for label in sorted(label_targets):
        added = 0
        for row in grouped.get(label, []):
            if added >= label_targets[label]:
                break
            topic_key = row.get("_topic_source_index", row["_source_index"])
            if max_per_topic is not None and topic_counts[topic_key] >= max_per_topic:
                continue
            selected.append(row)
            topic_counts[topic_key] += 1
            added += 1

    if len(selected) < sample_size:
        raise SystemExit(
            f"Task '{task}' could select only {len(selected)} rows after label/topic limits, "
            f"but --sample-size is {sample_size}."
        )
    selected.sort(key=ranking_key, reverse=True)
    return selected


def normalize_row(task: str, dataset_id: str, source_index: int, row: Row) -> Row:
    row = to_builtin(row)
    return {"_task": task, "_source_dataset": dataset_id, "_source_index": source_index, **row}


def build_mcqa_question_rows(dataset, dataset_id: str) -> list[Row]:
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


def write_jsonl(path: Path, rows: list[Row]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_summary_csv(path: Path, summary_rows: list[Row]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "task",
                "dataset",
                "num_samples",
                "num_candidates",
                "num_eligible_candidates",
                "labels",
                "category_coverage",
                "avg_lexicon_categories",
                "avg_lexicon_triggers",
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
    min_lexicon_categories: int,
    min_tokens: int,
    trigger_index: TriggerIndex,
) -> Row:
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
    selected = select_high_lexicon_coverage(
        candidates,
        task,
        sample_size,
        min_lexicon_categories=min_lexicon_categories,
        min_tokens=min_tokens,
        trigger_index=trigger_index,
        max_per_topic=MAX_MCQA_QUESTIONS_PER_TOPIC if task == "mcqa" else None,
    )
    for row in selected:
        metadata = row.pop("_lexicon_metadata", None)
        if metadata is None:
            _, metadata = lexicon_score(task, row, trigger_index)
        row["_lexicon_category_score"] = metadata["lexicon_category_score"]
        row["_lexicon_hits"] = metadata["lexicon_hits"]
        row["_dialect_triggers"] = metadata["dialect_triggers"]
        row["_dialect_categories"] = metadata["dialect_categories"]
        row["_num_dialect_triggers"] = metadata["lexicon_trigger_count"]
        row["_num_dialect_categories"] = metadata["lexicon_category_count"]

    out_path = output_dir / f"{task}_{sample_size}.jsonl"
    write_jsonl(out_path, selected)

    labels = Counter(get_label(task, row) for row in selected)
    selected_categories = Counter(
        category
        for row in selected
        for category in row.get("_dialect_categories", [])
    )
    category_coverage = len(selected_categories) / TOTAL_LEXICON_CATEGORIES * 100
    avg_lexicon_categories = sum(
        row.get("_num_dialect_categories", 0) for row in selected
    ) / max(len(selected), 1)
    avg_lexicon_triggers = sum(
        row.get("_num_dialect_triggers", 0) for row in selected
    ) / max(len(selected), 1)
    eligible_candidate_count = sum(
        passes_lexicon_filter(task, row, min_lexicon_categories, min_tokens, trigger_index)
        for row in candidates
    )
    return {
        "task": task,
        "dataset": dataset_id,
        "num_samples": len(selected),
        "num_candidates": len(candidates),
        "num_eligible_candidates": eligible_candidate_count,
        "labels": json.dumps(dict(labels), ensure_ascii=False),
        "category_coverage": f"{category_coverage:.0f}%",
        "avg_lexicon_categories": f"{avg_lexicon_categories:.1f}",
        "avg_lexicon_triggers": f"{avg_lexicon_triggers:.1f}",
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Select label-balanced samples with high dialect lexicon category coverage."
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=["sentiment", "nli", "qa", "mcqa"],
        help="Task names to sample: sentiment, nli, qa, mcqa.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=100,
        help="Number of samples per task.",
    )
    parser.add_argument(
        "--lexicon",
        type=Path,
        default=Path("data/DIALECT_LEXICON_v2.xlsx"),
        help="Path to dialect lexicon Excel file.",
    )
    parser.add_argument(
        "--min-lexicon-categories",
        type=int,
        default=1,
        help="Minimum number of lexicon categories required for selected candidates.",
    )
    parser.add_argument(
        "--min-tokens",
        type=int,
        default=7,
        help="Minimum number of tokens required for selected candidate text.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("data"))
    parser.add_argument(
        "--max-source-rows",
        type=int,
        default=None,
        help="Optional cap for quick smoke tests.",
    )
    parser.add_argument("--split", default=None, help="Override configured source split.")
    args = parser.parse_args()

    if args.sample_size < 0:
        raise SystemExit("sample size must be non-negative.")

    entries = load_lexicon(args.lexicon)
    trigger_index = build_trigger_index(entries)
    print(f"Loaded {len(entries)} lexicon entries and {len(trigger_index)} trigger keys from {args.lexicon}")

    summaries = [
        select_for_task(
            task=task,
            sample_size=args.sample_size,
            output_dir=args.output_dir,
            max_source_rows=args.max_source_rows,
            split_override=args.split,
            min_lexicon_categories=args.min_lexicon_categories,
            min_tokens=args.min_tokens,
          trigger_index=trigger_index,
        )
        for task in args.tasks
    ]
    write_summary_csv(args.output_dir / "selection_summary.csv", summaries)
    print(json.dumps(summaries, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
