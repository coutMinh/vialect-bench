"""Recreate the full ViDia2Std inputs without changing the original probe runners.

Completed segment predictions are reused from normalized_segments.jsonl. The
private checkpoint and HF_TOKEN are needed only if any segments are missing.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path


PROBE_SHA256 = "8e1bf191416d99e14418ea6451815303eb527a5b0c9f2b54e468d8fcc0057501"
MODEL_ID = "tarudesu/mbart-large-50"
MODEL_REVISION = "83c5b2a3b5078b54e4357c690f42b0bdaa167d08"
DIALECTS = ("PNB", "PNN", "PNT1", "PNT2", "PNT3", "PNT4")


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def segments(task, variant):
    if task == "sentiment":
        return [variant["text"]]
    if task == "nli":
        return [variant["hypothesis"]]
    if task == "qa":
        return [variant["question"]]
    if task == "mcqa":
        return [variant["question"], *variant["options"]]
    raise ValueError(f"Unknown task: {task}")


def normalize_missing(texts, output_path):
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise ValueError("HF_TOKEN is required to generate missing segments")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=token, revision=MODEL_REVISION)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_ID, token=token, revision=MODEL_REVISION)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    if hasattr(tokenizer, "src_lang"):
        tokenizer.src_lang = "vi_VN"
    if hasattr(tokenizer, "tgt_lang"):
        tokenizer.tgt_lang = "vi_VN"
    language_ids = getattr(tokenizer, "lang_code_to_id", None)
    with output_path.open("a", encoding="utf-8", newline="\n") as handle:
        for start in range(0, len(texts), 16):
            batch = texts[start:start + 16]
            inputs = tokenizer(batch, return_tensors="pt", max_length=128, truncation=True, padding=True)
            inputs.pop("token_type_ids", None)
            inputs = {key: value.to(device) for key, value in inputs.items()}
            kwargs = {
                "max_length": 128, "do_sample": False, "num_beams": 1,
                "pad_token_id": tokenizer.pad_token_id,
                "eos_token_id": tokenizer.eos_token_id,
            }
            if isinstance(language_ids, dict) and "vi_VN" in language_ids:
                kwargs["forced_bos_token_id"] = language_ids["vi_VN"]
            with torch.inference_mode():
                generated = model.generate(**inputs, **kwargs)
            outputs = tokenizer.batch_decode(generated, skip_special_tokens=True)
            if len(outputs) != len(batch) or any(not text.strip() for text in outputs):
                raise ValueError("Normalizer returned an incomplete batch")
            for source, normalized in zip(batch, outputs, strict=True):
                handle.write(json.dumps({"source": source, "normalized": normalized}, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", type=Path, default=Path("data/Finalized/probe_dialects.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/normalizer_full"))
    args = parser.parse_args()

    probe_bytes = args.probe.read_bytes().replace(b"\r\n", b"\n")
    if hashlib.sha256(probe_bytes).hexdigest() != PROBE_SHA256:
        raise ValueError("Probe differs from the frozen full-run source")
    cases = [json.loads(line) for line in probe_bytes.decode("utf-8").splitlines() if line]
    case_map = {str(case["id"]): case for case in cases}
    if len(cases) != 400 or len(case_map) != 400:
        raise ValueError("Expected 400 unique probe cases")
    keys = sorted((case["task"], str(case["id"]), dialect) for case in cases for dialect in DIALECTS)
    unique = sorted({text for task, case_id, dialect in keys
                     for text in segments(task, case_map[case_id]["variants"][dialect])})
    if len(unique) != 3294:
        raise ValueError(f"Unexpected number of distinct segments: {len(unique)}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    segment_path = args.output_dir / "normalized_segments.jsonl"
    lookup = {}
    if segment_path.exists():
        for row in read_jsonl(segment_path):
            source, normalized = row["source"], row["normalized"]
            if source in lookup and lookup[source] != normalized:
                raise ValueError(f"Conflicting segment output: {source}")
            lookup[source] = normalized
    if not set(lookup).issubset(unique):
        raise ValueError("Segment archive contains an unknown input")
    missing = [text for text in unique if text not in lookup]
    if missing:
        normalize_missing(missing, segment_path)
        lookup = {row["source"]: row["normalized"] for row in read_jsonl(segment_path)}
    if len(lookup) != len(unique):
        raise ValueError("Incomplete normalizer segment coverage")

    normalized_rows = []
    for task, case_id, dialect in keys:
        source = case_map[case_id]["variants"][dialect]
        if task == "sentiment":
            normalized = {"text": lookup[source["text"]]}
        elif task == "nli":
            normalized = {"hypothesis": lookup[source["hypothesis"]]}
        elif task == "qa":
            normalized = {"question": lookup[source["question"]]}
        else:
            normalized = {
                "question": lookup[source["question"]],
                "options": [lookup[option] for option in source["options"]],
            }
        normalized_rows.append({
            "task": task, "id": case_id, "variant": dialect,
            "source": source, "normalized": normalized,
            "normalizer_id": MODEL_ID, "normalizer_revision": MODEL_REVISION,
        })
    write_jsonl(args.output_dir / "normalized_case_dialects.jsonl", normalized_rows)

    normalized_map = {(row["task"], row["id"], row["variant"]): row["normalized"]
                      for row in normalized_rows}
    prepared_cases = []
    for case in cases:
        prepared = copy.deepcopy(case)
        standard = prepared["variants"]["standard"]
        if prepared["task"] == "nli":
            prepared["premise"] = standard["premise"]
        elif prepared["task"] in {"qa", "mcqa"}:
            prepared["context"] = standard["context"]
        prepared["variants"] = {
            dialect: normalized_map[(prepared["task"], str(prepared["id"]), dialect)]
            for dialect in DIALECTS
        }
        prepared_cases.append(prepared)
    write_jsonl(args.output_dir / "full_normalized_cases.jsonl", prepared_cases)
    print(f"Prepared {len(normalized_rows)} normalized variants from {len(unique)} distinct segments; "
          f"{sum(row['source'] != row['normalized'] for row in normalized_rows)} changed")


if __name__ == "__main__":
    main()
