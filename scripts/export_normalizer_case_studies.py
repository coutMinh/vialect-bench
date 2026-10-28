"""Export inspectable full-run examples with both models' raw responses."""
from __future__ import annotations

import json
from pathlib import Path

from analyze_normalizer_full import MODEL_ORDER, dialect_fields, exact, key, metric, read


CASES = {
    ("sentiment", "SENT_1239", "PNT3"): "dialect term restoration",
    ("qa", "QA_22620", "PNT1"): "partial question normalization",
    ("nli", "NLI_1465", "PNT4"): "unsupported negation",
    ("mcqa", "MCQA_0081_3", "PNT2"): "answer option corruption",
    ("sentiment", "SENT_5315", "PNT3"): "text omission",
}


def main():
    root = Path(__file__).resolve().parents[1]
    probe = {str(row["id"]): row for row in read(root / "data/Finalized/probe_dialects.jsonl")}
    inputs = {key(row): row for row in read(root / "outputs/normalizer_full/normalized_case_dialects.jsonl")}
    output = root / "outputs/normalizer_full/selected_case_studies.jsonl"
    rows = []
    for slug in MODEL_ORDER:
        name = f"{slug}.jsonl"
        direct = {key(row): row for row in read(root / "outputs/direct_prompting" / name)}
        normalized = {key(row): row for row in read(root / "outputs/normalized_direct_prompting_full" / name)}
        for case_key, category in CASES.items():
            task, case_id, dialect = case_key
            source = inputs[case_key]
            old = direct[case_key]
            new = normalized[case_key]
            standard = dialect_fields(task, probe[case_id]["variants"]["standard"])
            rows.append({
                "task": task, "id": case_id, "variant": dialect,
                "failure_or_success_mode": category,
                "source": source["source"], "normalized": source["normalized"],
                "standard": standard,
                "gold": old["gold"], "model": new["model_name"],
                "direct_prediction": old["prediction"],
                "normalized_prediction": new["prediction"],
                "direct_raw_output": old["raw_output"],
                "normalized_raw_output": new["raw_output"],
                "direct_main_score": metric(old),
                "normalized_main_score": metric(new),
                "direct_exact_match": exact(old["prediction"], old["gold"]),
                "normalized_exact_match": exact(new["prediction"], new["gold"]),
            })
    if len(rows) != 50:
        raise ValueError("Expected five cases for each of ten models")
    with output.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Wrote {output} with {len(rows)} model/case comparisons")


if __name__ == "__main__":
    main()
