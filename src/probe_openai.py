from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

from .probe_models import (
    DATASET_PART_CASES,
    PROMPT_STRATEGIES,
    build_probe_items,
    find_model,
    gold_value,
    load_cases,
    load_run_config,
    resolve_cases_path,
)
from .prompting import parse_prediction


def load_existing_keys(path: Path) -> set[tuple[str, str]]:
    if not path.exists():
        return set()

    keys = set()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("id") and row.get("variant"):
                keys.add((str(row["id"]), str(row["variant"])))
    return keys


def extract_output_text(response: Any) -> str:
    text = getattr(response, "output_text", None)
    if text is not None:
        return str(text)

    parts = []
    for item in getattr(response, "output", []) or []:
        for content in getattr(item, "content", []) or []:
            content_text = getattr(content, "text", None)
            if content_text:
                parts.append(str(content_text))
    return "\n".join(parts)


def usage_dict(response: Any) -> dict[str, Any]:
    usage = getattr(response, "usage", None)
    if usage is None:
        return {}
    if hasattr(usage, "model_dump"):
        return usage.model_dump()
    if isinstance(usage, dict):
        return usage
    return {
        key: getattr(usage, key)
        for key in ["input_tokens", "output_tokens", "total_tokens"]
        if hasattr(usage, key)
    }


def call_openai(
    client: Any,
    model_id: str,
    prompt: str,
    max_output_tokens: int,
    retries: int,
    retry_sleep: float,
) -> tuple[str, dict[str, Any]]:
    last_error = None
    for attempt in range(retries + 1):
        started = time.perf_counter()
        try:
            response = client.responses.create(
                model=model_id,
                input=prompt,
                temperature=0,
                max_output_tokens=max(16, max_output_tokens),
            )
            latency_sec = time.perf_counter() - started
            metadata = {
                "api_response_id": getattr(response, "id", None),
                "api_usage": usage_dict(response),
                "latency_sec": latency_sec,
                "retry_count": attempt,
            }
            return extract_output_text(response), metadata
        except Exception as exc:  # noqa: BLE001 - surface API errors in output.
            last_error = exc
            if attempt >= retries:
                break
            time.sleep(retry_sleep * (2**attempt))

    raise RuntimeError(f"OpenAI request failed after {retries + 1} attempts: {last_error}")


def output_row(
    item: Any,
    model_name: str,
    model_id: str,
    raw: str,
    dataset_part: str,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    case = item.case
    parsed = parse_prediction(case["task"], raw)
    return {
        "id": case.get("id"),
        "task": case["task"],
        "dataset_part": case.get("dataset_part") or dataset_part,
        "model_name": model_name,
        "model_id": model_id,
        "provider": "openai",
        "prompt_strategy": item.prompt_strategy,
        "variant": item.variant_name,
        "dialect_group": item.variant_name if item.variant_name != "standard" else "standard",
        "gold": gold_value(case),
        "prediction": parsed,
        "raw_output": raw.strip(),
        **metadata,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run OpenAI API probes on standard and dialect variants."
    )
    parser.add_argument("--config", type=Path, default=Path("configs/direct_prompting/gpt_4o.yaml"))
    parser.add_argument("--models", type=Path)
    parser.add_argument("--cases", type=Path)
    parser.add_argument(
        "--dataset-part",
        choices=sorted(DATASET_PART_CASES),
        help="Dataset split/part to use when --cases is not provided.",
    )
    parser.add_argument("--model-name")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--max-new-tokens", type=int)
    parser.add_argument(
        "--prompt-strategy",
        choices=sorted(PROMPT_STRATEGIES),
        help="Prompt setting to use: direct or normalized_direct.",
    )
    parser.add_argument("--include-empty-variants", action="store_true")
    parser.add_argument("--limit", type=int, help="Optional cap for smoke tests.")
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--retry-sleep", type=float, default=2.0)
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Do not skip rows already present in the output JSONL.",
    )
    args = parser.parse_args()

    try:
        from openai import OpenAI
    except ImportError as exc:
        raise SystemExit("OpenAI probing requires: pip install openai") from exc

    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is not set.")

    run_config = load_run_config(args.config)
    dataset_part = args.dataset_part or run_config.get("dataset_part", "finalized")
    models_path = args.models or Path(run_config.get("models", "configs/models.yaml"))
    cases_path = resolve_cases_path(
        args.cases,
        run_config,
        dataset_part,
        dataset_part_from_cli=args.dataset_part is not None,
    )
    model_name = args.model_name or run_config.get("model_name")
    output_path = args.output or Path(run_config.get("output", "outputs/openai_probe.jsonl"))
    max_new_tokens = args.max_new_tokens or int(run_config.get("max_new_tokens", 64))
    prompt_strategy = args.prompt_strategy or run_config.get("prompt_strategy", "direct")
    skip_empty_variants = not (
        args.include_empty_variants or bool(run_config.get("include_empty_variants", False))
    )

    if not model_name:
        raise SystemExit("Missing model_name. Set it in config or pass --model-name.")
    if prompt_strategy not in PROMPT_STRATEGIES:
        known = ", ".join(sorted(PROMPT_STRATEGIES))
        raise SystemExit(f"Unknown prompt_strategy '{prompt_strategy}'. Known: {known}")

    model_spec = find_model(models_path, model_name)
    if model_spec.get("provider") != "openai":
        raise SystemExit(
            f"Model '{model_name}' is not configured as provider: openai. "
            "Use src.probe_models for local Hugging Face models."
        )

    cases = load_cases(cases_path)
    probe_items = [
        item
        for case in cases
        for item in build_probe_items(
            case,
            skip_empty_variants=skip_empty_variants,
            prompt_strategy=prompt_strategy,
        )
    ]
    if args.limit is not None:
        probe_items = probe_items[: args.limit]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    completed = set() if args.no_resume else load_existing_keys(output_path)
    pending_items = [
        item
        for item in probe_items
        if (str(item.case.get("id")), str(item.variant_name)) not in completed
    ]

    print(
        f"Loaded {len(cases)} cases -> {len(probe_items)} probe items; "
        f"{len(pending_items)} pending for {model_name}.",
        flush=True,
    )

    client = OpenAI()
    with output_path.open("a", encoding="utf-8") as handle:
        for index, item in enumerate(pending_items, start=1):
            print(
                f"[{index}/{len(pending_items)}] {item.case.get('id')} {item.variant_name}",
                flush=True,
            )
            try:
                raw, metadata = call_openai(
                    client,
                    model_spec["model_id"],
                    item.prompt,
                    max_output_tokens=max_new_tokens,
                    retries=args.retries,
                    retry_sleep=args.retry_sleep,
                )
                row = output_row(
                    item,
                    model_spec["name"],
                    model_spec["model_id"],
                    raw,
                    dataset_part=dataset_part,
                    metadata=metadata,
                )
            except Exception as exc:  # noqa: BLE001 - keep failed item observable.
                row = {
                    "id": item.case.get("id"),
                    "task": item.case["task"],
                    "dataset_part": item.case.get("dataset_part") or dataset_part,
                    "model_name": model_spec["name"],
                    "model_id": model_spec["model_id"],
                    "provider": "openai",
                    "prompt_strategy": item.prompt_strategy,
                    "variant": item.variant_name,
                    "dialect_group": item.variant_name if item.variant_name != "standard" else "standard",
                    "gold": gold_value(item.case),
                    "prediction": "",
                    "raw_output": "",
                    "api_error": str(exc),
                }
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()


if __name__ == "__main__":
    main()
