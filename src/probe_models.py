from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .prompting import build_prompt, parse_prediction


@dataclass(frozen=True)
class ProbeItem:
    case: dict
    variant_name: str
    prompt: str


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_run_config(path: Path | None) -> dict:
    if path is None:
        return {}
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def load_text_generator(model_id: str):
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
    except ImportError as exc:
        raise SystemExit(
            "Model probing requires optional dependencies: transformers and torch."
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    )
    return pipeline("text-generation", model=model, tokenizer=tokenizer)


def generate(generator, prompt: str, max_new_tokens: int) -> str:
    result = generator(
        [{"role": "user", "content": prompt}],
        max_new_tokens=max_new_tokens,
        do_sample=False,
        return_full_text=False,
    )
    return extract_generated_text(result[0])


def generate_batch(
    generator,
    prompts: list[str],
    max_new_tokens: int,
    batch_size: int,
) -> list[str]:
    if batch_size <= 1:
        return [
            generate(generator, prompt, max_new_tokens=max_new_tokens)
            for prompt in prompts
        ]

    chats = [[{"role": "user", "content": prompt}] for prompt in prompts]
    results = generator(
        chats,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        return_full_text=False,
        batch_size=batch_size,
    )
    return [
        extract_generated_text(result[0] if isinstance(result, list) else result)
        for result in results
    ]


def extract_generated_text(result: Any) -> str:
    generated = result["generated_text"] if isinstance(result, dict) else result
    if isinstance(generated, list):
        return str(generated[-1].get("content", ""))
    return str(generated)


def is_empty_variant(variant: Any) -> bool:
    if variant is None:
        return True
    if isinstance(variant, str):
        return not variant.strip()
    if isinstance(variant, dict):
        text_values = [
            value
            for key, value in variant.items()
            if key not in {"context", "notes"} and isinstance(value, str)
        ]
        return bool(text_values) and all(not value.strip() for value in text_values)
    return False


def case_variants(case: dict) -> dict[str, Any]:
    if "variants" in case:
        return case["variants"]

    variants = {}
    for key in ["standard", "dialect"]:
        if key in case:
            variants[key] = case[key]
    return variants


def gold_value(case: dict) -> Any:
    return case.get("label") or case.get("reference") or case.get("answers")


def build_probe_items(case: dict, skip_empty_variants: bool) -> list[ProbeItem]:
    items = []
    for variant_name, variant in case_variants(case).items():
        if skip_empty_variants and is_empty_variant(variant):
            continue
        prompt = build_prompt(case, variant_name, variant)
        items.append(ProbeItem(case=case, variant_name=variant_name, prompt=prompt))
    return items


def output_row(item: ProbeItem, model_name: str, model_id: str, raw: str) -> dict:
    case = item.case
    parsed = parse_prediction(case["task"], raw)
    return {
        "id": case.get("id"),
        "task": case["task"],
        "model_name": model_name,
        "model_id": model_id,
        "variant": item.variant_name,
        "dialect_group": item.variant_name if item.variant_name != "standard" else "standard",
        "gold": gold_value(case),
        "prediction": parsed,
        "raw_output": raw.strip(),
    }


def chunks(items: list[ProbeItem], size: int):
    for start in range(0, len(items), size):
        yield items[start : start + size]


def find_model(models_path: Path, model_name: str) -> dict:
    config = yaml.safe_load(models_path.read_text(encoding="utf-8"))
    candidates = config.get("zero_shot_llms", []) + [
        model
        for task_models in config.get("pretrained_or_finetuned_task_models", {}).values()
        for model in task_models
    ]
    model_spec = next((model for model in candidates if model["name"] == model_name), None)
    if model_spec is None:
        known = ", ".join(model["name"] for model in candidates)
        raise SystemExit(f"Unknown model '{model_name}'. Known models: {known}")
    if not model_spec.get("model_id"):
        raise SystemExit(f"Model '{model_name}' has no model_id yet. Fill configs/models.yaml.")
    return model_spec


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run zero-shot LLM probes on standard and dialect variants."
    )
    parser.add_argument("--config", type=Path, default=Path("configs/probe.yaml"))
    parser.add_argument("--models", type=Path)
    parser.add_argument("--cases", type=Path)
    parser.add_argument("--model-name")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--max-new-tokens", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--include-empty-variants", action="store_true")
    args = parser.parse_args()

    run_config = load_run_config(args.config)
    models_path = args.models or Path(run_config.get("models", "configs/models.yaml"))
    cases_path = args.cases or Path(run_config.get("cases", "examples/probe_cases_template.jsonl"))
    model_name = args.model_name or run_config.get("model_name")
    output_path = args.output or Path(run_config.get("output", "outputs/model_probe.jsonl"))
    max_new_tokens = args.max_new_tokens or int(run_config.get("max_new_tokens", 64))
    batch_size = (
        args.batch_size
        if args.batch_size is not None
        else int(run_config.get("batch_size", 1))
    )
    skip_empty_variants = not (
        args.include_empty_variants or bool(run_config.get("include_empty_variants", False))
    )

    if not model_name:
        raise SystemExit("Missing model_name. Set it in configs/probe.yaml or pass --model-name.")
    if batch_size < 1:
        raise SystemExit("batch_size must be at least 1.")

    model_spec = find_model(models_path, model_name)
    generator = load_text_generator(model_spec["model_id"])
    cases = load_jsonl(cases_path)
    probe_items = [
        item
        for case in cases
        for item in build_probe_items(case, skip_empty_variants=skip_empty_variants)
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for batch in chunks(probe_items, batch_size):
            raw_outputs = generate_batch(
                generator,
                [item.prompt for item in batch],
                max_new_tokens=max_new_tokens,
                batch_size=batch_size,
            )
            for item, raw in zip(batch, raw_outputs, strict=True):
                row = output_row(item, model_spec["name"], model_spec["model_id"], raw)
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
