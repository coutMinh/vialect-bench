"""Check the shared data/result snapshot without models, API calls, or third-party packages."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "docs/rebuttal_manifest.json"
DIALECTS = {"PNB", "PNN", "PNT1", "PNT2", "PNT3", "PNT4"}
TEXT_SUFFIXES = {".json", ".jsonl", ".csv", ".tex", ".md", ".txt", ".svg"}


def digest(path: Path) -> str:
    data = path.read_bytes()
    if path.suffix in TEXT_SUFFIXES:
        data = data.replace(b"\r\n", b"\n")
    return hashlib.sha256(data).hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--update-manifest", action="store_true",
                        help="Accept the current data/results as a new shared snapshot after review.")
    args = parser.parse_args()
    if args.update_manifest:
        paths = sorted(
            p for base in [
                "data/Finalized", "outputs/direct_prompting",
                "outputs/finalized_6_complete_six_intrinsic_analysis",
                "outputs/lexical_substitution_analysis",
                "paper_artifacts/dataset_analysis", "paper_artifacts/direct_prompting",
            ] for p in (ROOT / base).rglob("*") if p.is_file()
        )
        paths += [ROOT / "data" / name for name in [
            "vialectbench_finalized_6.json", "DIALECT_LEXICON_v2.xlsx",
            "sentiment_100.jsonl", "nli_100.jsonl", "qa_100.jsonl", "mcqa_100.jsonl",
            "qualification_10.jsonl", "selection_summary.csv",
        ]]
        snapshot = {
            "description": "Shared rebuttal data and cached results; not proof of original run inputs.",
            "hashing": "SHA-256; CRLF normalized to LF for text files; binary files unchanged.",
            "files": {p.relative_to(ROOT).as_posix(): digest(p) for p in sorted(paths)},
        }
        MANIFEST.write_text(json.dumps(snapshot, indent=2) + "\n", encoding="utf-8")
        print(f"Updated {MANIFEST.relative_to(ROOT)}")

    snapshot = json.loads(MANIFEST.read_text(encoding="utf-8"))
    failures = []
    for name, expected in snapshot["files"].items():
        path = ROOT / name
        if not path.is_file() or digest(path) != expected:
            failures.append(f"Snapshot missing or changed: {name}")

    cases = read_jsonl(ROOT / "data/Finalized/probe_dialects.jsonl")
    ids = {(c["task"], c["id"]) for c in cases}
    if len(cases) != 400 or len(ids) != 400:
        failures.append("Expected 400 unique probe cases.")
    if Counter(c["task"] for c in cases) != {"sentiment": 100, "nli": 100, "qa": 100, "mcqa": 100}:
        failures.append("Expected 100 probe cases per task.")
    for c in cases:
        if set(c["variants"]) != DIALECTS | {"standard"} or not all(c["variants"].values()):
            failures.append(f"Incomplete variants: {c['id']}")
    annotations = json.loads((ROOT / "data/vialectbench_finalized_6.json").read_text(encoding="utf-8"))
    pairs = {(r["sample_id"], r["target_dialect"]) for r in annotations}
    expected_pairs = {(c["id"], dialect) for c in cases for dialect in DIALECTS}
    if len(annotations) != 2400 or pairs != expected_pairs:
        failures.append("Expected 2,400 unique annotation pairs matching the probe IDs.")

    expected_keys = {(task, sample_id, variant) for task, sample_id in ids
                     for variant in DIALECTS | {"standard"}}
    output_paths = sorted((ROOT / "outputs/direct_prompting").glob("*.jsonl"))
    expected_outputs = {name for name in snapshot["files"]
                        if name.startswith("outputs/direct_prompting/") and name.endswith(".jsonl")}
    if {p.relative_to(ROOT).as_posix() for p in output_paths} != expected_outputs:
        failures.append("Model result files differ from the reviewed snapshot; review before updating the manifest.")
    for path in output_paths:
        rows = read_jsonl(path)
        keys = {(r["task"], r["id"], r["variant"]) for r in rows}
        if len(rows) != 2800 or keys != expected_keys:
            failures.append(f"Incomplete or duplicate result coverage: {path.name}")
        if any(r.get("error") for r in rows):
            failures.append(f"Result rows contain errors: {path.name}")
    if failures:
        raise SystemExit("\n".join(failures))
    print(f"OK: {len(snapshot['files'])} snapshot files; 400 cases; 2,400 rewrites; "
          f"{len(output_paths)} models x 2,800 result rows.")


if __name__ == "__main__":
    main()
