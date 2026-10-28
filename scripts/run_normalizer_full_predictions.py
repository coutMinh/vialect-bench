"""Run the original direct-prompting model configurations on normalized inputs.

This wrapper only changes the case file, output path, and strategy metadata.
`normalized_direct` uses the same prompt templates as `direct`.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys


MODELS = (
    "qwen_2_5_0_5b_instruct", "qwen_2_5_3b_instruct", "qwen_2_5_7b_instruct",
    "llama_3_1_8b_instruct", "mistral_7b_instruct_v0_3", "gemma_2_9b_it",
    "gemma_3_4b_it", "vistral_7b_chat", "seallm_7b_chat", "gpt_4o",
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=Path("outputs/normalizer_full/full_normalized_cases.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/normalized_direct_prompting_full"))
    parser.add_argument("--models", nargs="+", choices=MODELS, default=MODELS)
    args = parser.parse_args()
    if sum(bool(line.strip()) for line in args.cases.open(encoding="utf-8")) != 400:
        raise ValueError("Expected 400 normalized cases")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for slug in args.models:
        module = "src.probe_openai" if slug == "gpt_4o" else (
            "src.probe_gemma3" if slug == "gemma_3_4b_it" else "src.probe_models"
        )
        output = args.output_dir / f"{slug}.jsonl"
        command = [
            sys.executable, "-u", "-m", module,
            "--config", f"configs/direct_prompting/{slug}.yaml",
            "--cases", str(args.cases),
            "--output", str(output),
            "--prompt-strategy", "normalized_direct",
        ]
        print(f"Running {slug}", flush=True)
        subprocess.run(command, check=True)
        rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines() if line.strip()]
        keys = {(str(row["id"]), row["variant"]) for row in rows}
        if (len(rows) != 2400 or len(keys) != 2400
                or any(not str(row.get("raw_output") or "").strip() for row in rows)):
            raise ValueError(f"Incomplete output for {slug}: {len(rows)} rows")
        print(f"Completed {slug}: 2,400 raw responses", flush=True)


if __name__ == "__main__":
    main()
