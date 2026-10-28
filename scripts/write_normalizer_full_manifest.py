"""Validate and hash the full normalization run for archiving."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from run_normalizer_full_predictions import MODELS


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "outputs/normalizer_full"
RESULTS = ROOT / "outputs/normalized_direct_prompting_full"
SOURCE_COMMIT = "d9314e80bf508b2f533063c2f8f2a1be14f57792"
PROBE_SHA256 = "8e1bf191416d99e14418ea6451815303eb527a5b0c9f2b54e468d8fcc0057501"


TEXT_SUFFIXES = {".csv", ".json", ".jsonl", ".md", ".py", ".tex", ".yaml"}


def canonical_bytes(path):
    data = path.read_bytes()
    return data.replace(b"\r\n", b"\n") if path.suffix in TEXT_SUFFIXES else data


def row_count(path):
    return sum(bool(line.strip()) for line in path.open(encoding="utf-8"))


def record(path):
    data = canonical_bytes(path)
    return {
        "path": path.relative_to(ROOT).as_posix(),
        "sha256": hashlib.sha256(data).hexdigest(),
        "bytes": len(data),
        "rows": row_count(path) if path.suffix == ".jsonl" else None,
    }


def main():
    probe_bytes = (ROOT / "data/Finalized/probe_dialects.jsonl").read_bytes().replace(b"\r\n", b"\n")
    if hashlib.sha256(probe_bytes).hexdigest() != PROBE_SHA256:
        raise ValueError("Probe differs from the pinned source commit")
    inputs = [
        DATA / "normalized_segments.jsonl",
        DATA / "normalized_case_dialects.jsonl",
        DATA / "full_normalized_cases.jsonl",
        DATA / "remote_runner_hashes.json",
        DATA / "prompt_equivalence_report.json",
    ]
    expected = [3294, 2400, 400]
    if [row_count(path) for path in inputs[:3]] != expected:
        raise ValueError("Normalized input coverage is incomplete")
    remote_hashes = json.loads((DATA / "remote_runner_hashes.json").read_text(encoding="utf-8"))
    for name, expected_hash in remote_hashes.items():
        local_bytes = (ROOT / name).read_bytes().replace(b"\r\n", b"\n")
        if hashlib.sha256(local_bytes).hexdigest() != expected_hash:
            raise ValueError(f"Pinned runner/config mismatch: {name}")
    outputs = [RESULTS / f"{slug}.jsonl" for slug in MODELS]
    direct_outputs = [ROOT / "outputs/direct_prompting" / f"{slug}.jsonl" for slug in MODELS]
    if sorted(path.name for path in RESULTS.glob("*.jsonl")) != sorted(path.name for path in outputs):
        raise ValueError("Unexpected full-model result set")
    for path in outputs:
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        keys = {(row["task"], str(row["id"]), row["variant"]) for row in rows}
        if len(rows) != 2400 or len(keys) != 2400 or any(not str(row.get("raw_output") or "").strip() for row in rows):
            raise ValueError(f"Incomplete raw outputs: {path.name}")
    if any(row_count(path) != 2800 for path in direct_outputs):
        raise ValueError("Original direct-prompting coverage is incomplete")
    configs = [ROOT / "configs/direct_prompting" / f"{slug}.yaml" for slug in MODELS]
    configs.extend([ROOT / "configs/models.yaml", ROOT / "configs/probe.yaml"])
    runners = [ROOT / name for name in (
        "src/probe_models.py", "src/probe_gemma3.py", "src/probe_openai.py", "src/prompting.py",
    )]
    derived = [DATA / name for name in (
        "text_change_summary.csv", "field_match_summary.csv",
        "case_level_outcomes.csv", "analysis_summary.json",
        "gpt_usage_summary.json", "selected_case_studies.jsonl",
    )]
    derived += sorted((ROOT / "paper_artifacts/direct_prompting").glob("paper_table_normalizer_full*"))
    if len(derived) < 10 or any(not path.exists() for path in derived):
        raise ValueError("Run the full analysis before writing its manifest")
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_commit": SOURCE_COMMIT,
        "probe_sha256_lf": PROBE_SHA256,
        "normalizer_id": "tarudesu/mbart-large-50",
        "normalizer_revision": "83c5b2a3b5078b54e4357c690f42b0bdaa167d08",
        "text_hash_normalization": "CRLF to LF; binary files unchanged",
        "source_cases": 400,
        "dialect_variants_per_case": 6,
        "model_count": len(MODELS),
        "raw_model_rows": 2400 * len(MODELS),
        "inputs": [record(path) for path in inputs],
        "outputs": [record(path) for path in outputs],
        "direct_comparison_outputs": [record(path) for path in direct_outputs],
        "configs": [record(path) for path in configs],
        "runners": [record(path) for path in runners],
        "execution_logs": record(DATA / "run_logs.tar.gz"),
        "derived_analysis": [record(path) for path in derived],
    }
    target = DATA / "run_manifest.json"
    target.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {target} with {manifest['raw_model_rows']} validated raw model rows")


if __name__ == "__main__":
    main()
