from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any


TASK_MAP = {
    "SENT": "sentiment",
    "NLI": "nli",
    "QA": "qa",
    "MCQA": "mcqa",
}

OPTION_RE = re.compile(r"(?m)^\s*([A-D])\.\s*(.+?)\s*$")


def load_json(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list in {path}")
    return data


def parse_mcqa(text: str) -> tuple[str, list[str]]:
    matches = list(OPTION_RE.finditer(text))
    if not matches:
        return text.strip(), []

    question = text[: matches[0].start()].strip()
    options_by_label = {match.group(1): match.group(2).strip() for match in matches}
    options = [options_by_label.get(label, "") for label in ["A", "B", "C", "D"]]
    return question, options


def base_case(row: dict[str, Any], dataset_part: str) -> dict[str, Any]:
    task = TASK_MAP.get(str(row["task"]).upper(), str(row["task"]).lower())
    sample_id = str(row["sample_id"])
    common = {
        "id": sample_id,
        "task": task,
        "source_dataset": "vialectbench_annotated",
        "source_index": sample_id,
        "dataset_part": dataset_part,
        "domain": row.get("domain"),
        "confidence": row.get("confidence"),
        "final_status": row.get("final_status"),
        "review_count": row.get("review_count"),
        "annotation_source": "human_annotated",
        "variants": {},
    }

    if task == "sentiment":
        common["label"] = row.get("label")
        common["variants"]["standard"] = {"text": row.get("original_text", "")}
    elif task == "nli":
        common["label"] = normalize_nli_label(row.get("label"))
        common["variants"]["standard"] = {
            "premise": row.get("original_text", ""),
            "hypothesis": row.get("hypothesis", ""),
        }
    elif task == "qa":
        common["answers"] = [row.get("label") or row.get("hypothesis")]
        common["variants"]["standard"] = {
            "context": "",
            "question": row.get("original_text", ""),
        }
    elif task == "mcqa":
        question, options = parse_mcqa(str(row.get("original_text", "")))
        common["label"] = str(row.get("label", "")).strip().upper()
        common["variants"]["standard"] = {
            "context": "",
            "question": question,
            "options": options,
        }
    else:
        raise ValueError(f"Unsupported task: {row.get('task')}")

    return common


def normalize_nli_label(label: Any) -> str:
    value = str(label or "").strip().lower()
    label_map = {
        "entailment": "entailment",
        "neutral": "neutral",
        "contradiction": "contradiction",
        "contradict": "contradiction",
    }
    return label_map.get(value, str(label or "").strip())


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


def convert_rows(rows: list[dict[str, Any]], dataset_part: str) -> list[dict[str, Any]]:
    cases: dict[tuple[str, str], dict[str, Any]] = {}
    annotators: dict[tuple[str, str], set[str]] = defaultdict(set)
    dialect_statuses: dict[tuple[str, str], dict[str, str]] = defaultdict(dict)

    for row in rows:
        key = (str(row["task"]).upper(), str(row["sample_id"]))
        if key not in cases:
            cases[key] = base_case(row, dataset_part)

        dialect = str(row.get("target_dialect", "")).strip()
        if not dialect:
            continue

        cases[key]["variants"][dialect] = dialect_variant(row)
        if row.get("annotator_id"):
            annotators[key].add(str(row["annotator_id"]))
        dialect_statuses[key][dialect] = str(row.get("final_status", ""))

    output = []
    for key in sorted(cases):
        case = cases[key]
        dialects = sorted(name for name in case["variants"] if name != "standard")
        case["dialects"] = dialects
        case["num_dialects"] = len(dialects)
        case["annotator_count"] = len(annotators[key])
        case["dialect_statuses"] = dialect_statuses[key]
        output.append(case)

    return output


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert human-annotated VialectBench JSON rows into probe_dialects JSONL."
    )
    parser.add_argument("--input", type=Path, default=Path("data/vialectbench_finalized.json"))
    parser.add_argument("--output", type=Path, default=Path("data/Finalized/probe_dialects.jsonl"))
    parser.add_argument("--dataset-part", default="finalized")
    args = parser.parse_args()

    rows = load_json(args.input)
    cases = convert_rows(rows, dataset_part=args.dataset_part)
    write_jsonl(args.output, cases)
    print(f"Wrote {len(cases)} cases to {args.output}")


if __name__ == "__main__":
    main()
