from __future__ import annotations

import argparse
import json
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


TASK_MAP = {
    "SENT": "sentiment",
    "NLI": "nli",
    "QA": "qa",
    "MCQA": "mcqa",
}

DIALECTS = ["PNN", "PNT1", "PNT2", "PNT3", "PNT4", "PNB"]
OPTION_RE = re.compile(r"(?m)^\s*([A-D])\.\s*(.+?)\s*$")
SAMPLE_ID_RE = re.compile(r"^(SENT|NLI|QA|MCQA)_(\d+)(?:_(\d+))?$")


def load_json(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list in {path}")
    return data


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def normalize_text(value: Any) -> str:
    text = unicodedata.normalize("NFC", str(value or ""))
    return re.sub(r"\s+", " ", text).strip()


def parse_sample_id(sample_id: str) -> tuple[str, int, int | None]:
    match = SAMPLE_ID_RE.fullmatch(sample_id)
    if not match:
        raise ValueError(f"Unsupported sample_id format: {sample_id}")
    task, source_index, question_index = match.groups()
    return task, int(source_index), int(question_index) if question_index else None


def parse_mcqa(text: str) -> tuple[str, list[str]]:
    matches = list(OPTION_RE.finditer(text))
    if not matches:
        return text.strip(), []

    question = text[: matches[0].start()].strip()
    options_by_label = {match.group(1): match.group(2).strip() for match in matches}
    options = [options_by_label.get(label, "") for label in ["A", "B", "C", "D"]]
    return question, options


def mcqa_text(question: str, options: list[str]) -> str:
    option_lines = [f"{label}. {text}" for label, text in zip(["A", "B", "C", "D"], options)]
    return normalize_text(question + "\n" + "\n".join(option_lines))


def normalize_nli_label(label: Any) -> str:
    value = str(label or "").strip().lower()
    label_map = {
        "entailment": "entailment",
        "neutral": "neutral",
        "contradiction": "contradiction",
        "contradict": "contradiction",
    }
    return label_map.get(value, str(label or "").strip())


def source_maps(
    sentiment_path: Path,
    nli_path: Path,
    qa_path: Path,
    mcqa_path: Path,
) -> dict[str, dict[Any, dict[str, Any]]]:
    sentiment = {int(row["_source_index"]): row for row in load_jsonl(sentiment_path)}
    nli = {int(row["_source_index"]): row for row in load_jsonl(nli_path)}
    qa = {int(row["_source_index"]): row for row in load_jsonl(qa_path)}
    mcqa = {
        (int(row["_source_index"]), int(row["_question_index"])): row
        for row in load_jsonl(mcqa_path)
    }
    return {"SENT": sentiment, "NLI": nli, "QA": qa, "MCQA": mcqa}


def source_for_row(
    row: dict[str, Any],
    sources: dict[str, dict[Any, dict[str, Any]]],
) -> dict[str, Any]:
    task_code, source_index, question_index = parse_sample_id(str(row["sample_id"]))
    key: Any = (source_index, question_index) if task_code == "MCQA" else source_index
    source = sources[task_code].get(key)
    if source is None:
        raise ValueError(f"{row['sample_id']} was not found in the selected 100-sample files")
    return source


def validate_against_source(row: dict[str, Any], source: dict[str, Any]) -> None:
    task_code, _, _ = parse_sample_id(str(row["sample_id"]))

    if task_code == "SENT":
        if normalize_text(row.get("original_text")) != normalize_text(source.get("Sentence")):
            raise ValueError(f"Text mismatch for {row['sample_id']}")
        if normalize_text(row.get("label")) != normalize_text(source.get("Emotion")):
            raise ValueError(f"Label mismatch for {row['sample_id']}")
        return

    if task_code == "NLI":
        if normalize_text(row.get("original_text")) != normalize_text(source.get("premise")):
            raise ValueError(f"Premise mismatch for {row['sample_id']}")
        if normalize_text(row.get("hypothesis")) != normalize_text(source.get("hypothesis")):
            raise ValueError(f"Hypothesis mismatch for {row['sample_id']}")
        if normalize_nli_label(row.get("label")) != normalize_nli_label(source.get("label")):
            raise ValueError(f"Label mismatch for {row['sample_id']}")
        return

    if task_code == "QA":
        if normalize_text(row.get("original_text")) != normalize_text(source.get("question")):
            raise ValueError(f"Question mismatch for {row['sample_id']}")
        answers = [normalize_text(text) for text in (source.get("answers") or {}).get("text", [])]
        answer = normalize_text(row.get("label") or row.get("hypothesis"))
        if answer and answer not in answers:
            raise ValueError(f"Answer mismatch for {row['sample_id']}")
        return

    if task_code == "MCQA":
        question, options = parse_mcqa(str(row.get("original_text", "")))
        if mcqa_text(question, options) != mcqa_text(source.get("question", ""), source.get("options") or []):
            raise ValueError(f"Question/options mismatch for {row['sample_id']}")
        if normalize_text(row.get("label")).upper() != normalize_text(source.get("answer")).upper():
            raise ValueError(f"Answer label mismatch for {row['sample_id']}")
        answer_index = "ABCD".find(normalize_text(row.get("label")).upper())
        source_options = source.get("options") or []
        if 0 <= answer_index < len(source_options):
            if normalize_text(row.get("hypothesis")) != normalize_text(source_options[answer_index]):
                raise ValueError(f"Answer text mismatch for {row['sample_id']}")


def empty_variant(task: str) -> dict[str, Any]:
    if task == "sentiment":
        return {"text": ""}
    if task == "nli":
        return {"hypothesis": ""}
    if task == "qa":
        return {"question": ""}
    if task == "mcqa":
        return {"question": "", "options": []}
    raise ValueError(f"Unsupported task: {task}")


def base_case(
    row: dict[str, Any],
    source: dict[str, Any],
    dataset_part: str,
) -> dict[str, Any]:
    task_code, source_index, question_index = parse_sample_id(str(row["sample_id"]))
    task = TASK_MAP[task_code]
    sample_id = str(row["sample_id"])
    common = {
        "id": sample_id,
        "task": task,
        "source_dataset": source.get("_source_dataset", "vialectbench_annotated"),
        "source_index": source_index,
        "dataset_part": dataset_part,
        "domain": row.get("domain"),
        "confidence": row.get("confidence"),
        "final_status": row.get("final_status"),
        "review_count": row.get("review_count"),
        "annotation_source": "human_annotated",
        "variants": {},
    }
    if question_index is not None:
        common["question_index"] = question_index

    if task == "sentiment":
        common["label"] = source.get("Emotion")
        common["variants"]["standard"] = {"text": source.get("Sentence", "")}
    elif task == "nli":
        common["label"] = normalize_nli_label(source.get("label"))
        common["source_uid"] = source.get("uid")
        common["variants"]["standard"] = {
            "premise": source.get("premise", ""),
            "hypothesis": source.get("hypothesis", ""),
        }
    elif task == "qa":
        answers = source.get("answers") or {}
        common["answers"] = answers.get("text") or [row.get("label") or row.get("hypothesis")]
        common["source_id"] = source.get("id")
        common["source_title"] = source.get("title")
        common["variants"]["standard"] = {
            "context": source.get("context", ""),
            "question": source.get("question", ""),
        }
    elif task == "mcqa":
        common["label"] = str(source.get("answer", "")).strip().upper()
        common["source_title"] = source.get("title")
        common["variants"]["standard"] = {
            "context": source.get("article", ""),
            "question": source.get("question", ""),
            "options": source.get("options") or [],
        }
    else:
        raise ValueError(f"Unsupported task: {row.get('task')}")

    for dialect in DIALECTS:
        common["variants"][dialect] = empty_variant(task)

    return common


def dialect_variant(row: dict[str, Any]) -> dict[str, Any]:
    task = TASK_MAP.get(str(row["task"]).upper(), str(row["task"]).lower())
    dialect_text = row.get("dialect_text", "")

    if task == "sentiment":
        return {"text": dialect_text}
    if task == "nli":
        return {"hypothesis": dialect_text}
    if task == "qa":
        return {"question": dialect_text}
    if task == "mcqa":
        question, options = parse_mcqa(str(dialect_text))
        return {"question": question, "options": options}
    raise ValueError(f"Unsupported task: {row.get('task')}")


def convert_rows(
    rows: list[dict[str, Any]],
    sources: dict[str, dict[Any, dict[str, Any]]],
    dataset_part: str,
    validate_sources: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    cases: dict[tuple[str, str], dict[str, Any]] = {}
    annotators: dict[tuple[str, str], set[str]] = defaultdict(set)
    dialect_statuses: dict[tuple[str, str], dict[str, str]] = defaultdict(dict)
    annotated_dialects: dict[tuple[str, str], set[str]] = defaultdict(set)
    duplicate_counts: Counter[tuple[str, str, str]] = Counter()

    for row in rows:
        task_code = str(row["task"]).upper()
        key = (task_code, str(row["sample_id"]))
        source = source_for_row(row, sources)
        if validate_sources:
            validate_against_source(row, source)
        if key not in cases:
            cases[key] = base_case(row, source, dataset_part)

        dialect = str(row.get("target_dialect", "")).strip()
        if not dialect:
            continue
        if dialect not in DIALECTS:
            raise ValueError(f"Unsupported dialect '{dialect}' for {row['sample_id']}")

        duplicate_counts[(task_code, str(row["sample_id"]), dialect)] += 1
        cases[key]["variants"][dialect] = dialect_variant(row)
        annotated_dialects[key].add(dialect)
        if row.get("annotator_id"):
            annotators[key].add(str(row["annotator_id"]))
        dialect_statuses[key][dialect] = str(row.get("final_status", ""))

    output = []
    for key in sorted(cases):
        case = cases[key]
        filled = sorted(annotated_dialects[key])
        case["dialects"] = list(DIALECTS)
        case["annotated_dialects"] = filled
        case["num_dialects"] = len(filled)
        case["num_dialect_slots"] = len(DIALECTS)
        case["annotator_count"] = len(annotators[key])
        case["dialect_statuses"] = {
            dialect: dialect_statuses[key].get(dialect, "") for dialect in DIALECTS
        }
        output.append(case)

    duplicate_groups = sum(1 for count in duplicate_counts.values() if count > 1)
    summary = {
        "annotation_rows": len(rows),
        "cases": len(output),
        "duplicate_sample_dialect_groups": duplicate_groups,
    }
    return output, summary


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert human-annotated VialectBench JSON rows into probe_dialects JSONL."
    )
    parser.add_argument("--input", type=Path, default=Path("data/vialectbench_finalized_1.json"))
    parser.add_argument("--output", type=Path, default=Path("data/Finalized/probe_dialects.jsonl"))
    parser.add_argument("--dataset-part", default="finalized")
    parser.add_argument("--sentiment", type=Path, default=Path("data/sentiment_100.jsonl"))
    parser.add_argument("--nli", type=Path, default=Path("data/nli_100.jsonl"))
    parser.add_argument("--qa", type=Path, default=Path("data/qa_100.jsonl"))
    parser.add_argument("--mcqa", type=Path, default=Path("data/mcqa_100.jsonl"))
    parser.add_argument(
        "--no-validate-sources",
        action="store_true",
        help="Skip consistency checks against the selected 100-sample source files.",
    )
    args = parser.parse_args()

    rows = load_json(args.input)
    sources = source_maps(
        sentiment_path=args.sentiment,
        nli_path=args.nli,
        qa_path=args.qa,
        mcqa_path=args.mcqa,
    )
    cases, summary = convert_rows(
        rows,
        sources=sources,
        dataset_part=args.dataset_part,
        validate_sources=not args.no_validate_sources,
    )
    write_jsonl(args.output, cases)
    print(
        f"Wrote {len(cases)} cases to {args.output} "
        f"from {summary['annotation_rows']} annotation rows; "
        f"duplicate sample+dialect groups: {summary['duplicate_sample_dialect_groups']}"
    )


if __name__ == "__main__":
    main()
