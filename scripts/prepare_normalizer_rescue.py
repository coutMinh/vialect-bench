"""Extract the exact direct-prompting failures used by EDA's rescue table.

This is read-only with respect to the benchmark and cached predictions. It
writes a small worklist under .local/rebuttal/normalizer-rescue/ by default.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import json
import math
from pathlib import Path
import re


DIALECTS = {"PNB", "PNN", "PNT1", "PNT2", "PNT3", "PNT4"}


def normalize_answer(value: object) -> str:
    if isinstance(value, list):
        value = value[0] if value else ""
    if value is None:
        return ""
    text = str(value).lower().strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[\.,;:!?\(\)\[\]\{\}\"'`]+", "", text)
    return text.strip()


def exact_match(prediction: object, gold: object) -> bool:
    golds = gold if isinstance(gold, list) else [gold]
    pred_norm = normalize_answer(prediction)
    return any(pred_norm == normalize_answer(item) for item in golds)


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def build_worklist() -> tuple[list[dict], list[dict], dict]:
    probe_path = Path("data/Finalized/probe_dialects.jsonl")
    cases = {str(case["id"]): case for case in read_jsonl(probe_path)}
    prediction_dir = Path("outputs/direct_prompting")
    predictions = sorted(prediction_dir.glob("*.jsonl"))
    if len(cases) != 400 or len(predictions) != 10:
        raise ValueError("Expected 400 probe cases and 10 cached model files")

    targets: list[dict] = []
    all_counts = Counter()
    for path in predictions:
        model = path.stem
        by_case = defaultdict(dict)
        for row in read_jsonl(path):
            if not str(row.get("raw_output") or "").strip() and not str(row.get("prediction") or "").strip():
                continue
            task = str(row["task"]).lower()
            case_id = str(row["id"])
            variant = row.get("dialect_group") or row.get("variant") or "standard"
            key = (task, case_id)
            if variant in by_case[key]:
                raise ValueError(f"Duplicate prediction: {model}, {key}, {variant}")
            by_case[key][variant] = row
            all_counts[model] += 1

        for (task, case_id), variants in by_case.items():
            standard = variants.get("standard")
            dialect_rows = [(name, row) for name, row in variants.items() if name in DIALECTS]
            if standard is None or not exact_match(standard.get("prediction"), standard.get("gold")):
                continue
            failures = [(name, row) for name, row in dialect_rows if not exact_match(row.get("prediction"), row.get("gold"))]
            if not dialect_rows or len(failures) < math.ceil(len(dialect_rows) / 2):
                continue
            if case_id not in cases or cases[case_id]["task"] != task:
                raise ValueError(f"Missing or mismatched probe case: {task}, {case_id}")
            for variant, row in failures:
                targets.append({
                    "model_slug": model,
                    "task": task,
                    "id": case_id,
                    "variant": variant,
                    "gold": row.get("gold"),
                    "direct_prediction": row.get("prediction"),
                    "standard_prediction": standard.get("prediction"),
                })

    targets.sort(key=lambda row: (row["model_slug"], row["task"], row["id"], row["variant"]))
    unique_keys = sorted({(row["task"], row["id"], row["variant"]) for row in targets})
    unique_variants = [
        {"task": task, "id": case_id, "variant": variant}
        for task, case_id, variant in unique_keys
    ]
    counts = Counter(row["model_slug"] for row in targets)
    summary = {
        "input_probe": str(probe_path),
        "input_predictions": str(prediction_dir),
        "selection": "standard exact match correct; >= half of available dialect variants exact match wrong; include only those wrong dialect rows",
        "prediction_rows_read_by_model": dict(sorted(all_counts.items())),
        "hard_failure_rows_by_model": dict(sorted(counts.items())),
        "hard_failure_rows": len(targets),
        "unique_case_dialect_variants": len(unique_variants),
        "unique_source_cases": len({row["id"] for row in unique_variants}),
    }
    return targets, unique_variants, summary


def write_model_cases(output_dir: Path, targets: list[dict], normalized_path: Path) -> None:
    cases = {str(case["id"]): case for case in read_jsonl(Path("data/Finalized/probe_dialects.jsonl"))}
    normalized = {}
    revisions = set()
    for row in read_jsonl(normalized_path):
        key = (row["task"], str(row["id"]), row["variant"])
        if key in normalized:
            raise ValueError(f"Duplicate normalized variant: {key}")
        source = cases[key[1]]["variants"][key[2]]
        if row["source"] != source:
            raise ValueError(f"Normalizer source mismatch: {key}")
        normalized[key] = row["normalized"]
        revisions.add((row["normalizer_id"], row["normalizer_revision"]))
    needed = {(row["task"], row["id"], row["variant"]) for row in targets}
    if normalized.keys() != needed or len(revisions) != 1:
        raise ValueError("Normalized rows do not exactly match the rescue cohort or model revision")

    by_model = defaultdict(lambda: defaultdict(dict))
    for row in targets:
        by_model[row["model_slug"]][row["id"]][row["variant"]] = normalized[
            (row["task"], row["id"], row["variant"])
        ]

    model_dir = output_dir / "model_cases"
    model_dir.mkdir(parents=True, exist_ok=True)
    for model, by_case in sorted(by_model.items()):
        prepared = []
        for case_id, variants in by_case.items():
            case = deepcopy(cases[case_id])
            standard = case["variants"]["standard"]
            if case["task"] == "nli":
                case["premise"] = standard["premise"]
            elif case["task"] in {"qa", "mcqa"}:
                case["context"] = standard["context"]
            case["variants"] = variants
            prepared.append(case)
        write_jsonl(model_dir / f"{model}.jsonl", prepared)
    print(f"Wrote {len(targets)} model rows from {len(needed)} normalized variants across {len(by_model)} model case files")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path(".local/rebuttal/normalizer-rescue"))
    parser.add_argument("--normalized-rows", type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    targets, unique_variants, summary = build_worklist()
    write_jsonl(args.output_dir / "hard_failure_rows.jsonl", targets)
    write_jsonl(args.output_dir / "unique_case_dialects.jsonl", unique_variants)
    (args.output_dir / "worklist_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.normalized_rows:
        write_model_cases(args.output_dir, targets, args.normalized_rows)


if __name__ == "__main__":
    main()
