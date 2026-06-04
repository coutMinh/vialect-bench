from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from .prompting import build_prompt, parse_prediction


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


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
        prompt,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        return_full_text=False,
    )
    return result[0]["generated_text"]


def evaluate_case(generator, model_name: str, model_id: str, case: dict, max_new_tokens: int) -> list[dict]:
    outputs = []
    for variant_key in ["standard", "dialect"]:
        if variant_key not in case:
            continue
        prompt = build_prompt(case, variant_key)
        raw = generate(generator, prompt, max_new_tokens=max_new_tokens)
        parsed = parse_prediction(case["task"], raw)
        outputs.append(
            {
                "id": case.get("id"),
                "task": case["task"],
                "model_name": model_name,
                "model_id": model_id,
                "variant": variant_key,
                "dialect_group": case.get("dialect_group"),
                "label": case.get("label") or case.get("reference"),
                "prediction": parsed,
                "raw_output": raw.strip(),
            }
        )
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run zero-shot LLM probes on standard/dialect testcase pairs."
    )
    parser.add_argument("--models", type=Path, default=Path("configs/models.yaml"))
    parser.add_argument("--cases", type=Path, default=Path("examples/dialect_cases.jsonl"))
    parser.add_argument("--model-name", required=True, help="Name from configs/models.yaml.")
    parser.add_argument("--output", type=Path, default=Path("outputs/model_probe.jsonl"))
    parser.add_argument("--max-new-tokens", type=int, default=64)
    args = parser.parse_args()

    config = yaml.safe_load(args.models.read_text(encoding="utf-8"))
    candidates = config.get("zero_shot_llms", []) + [
        model
        for task_models in config.get("pretrained_or_finetuned_task_models", {}).values()
        for model in task_models
    ]
    model_spec = next((model for model in candidates if model["name"] == args.model_name), None)
    if model_spec is None:
        known = ", ".join(model["name"] for model in candidates)
        raise SystemExit(f"Unknown model '{args.model_name}'. Known models: {known}")
    if not model_spec.get("model_id"):
        raise SystemExit(f"Model '{args.model_name}' has no model_id yet. Fill configs/models.yaml.")

    generator = load_text_generator(model_spec["model_id"])
    cases = load_jsonl(args.cases)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for case in cases:
            for row in evaluate_case(
                generator,
                model_spec["name"],
                model_spec["model_id"],
                case,
                max_new_tokens=args.max_new_tokens,
            ):
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
