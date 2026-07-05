from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .prompting import PROMPT_STRATEGIES, build_prompt, parse_prediction


DATASET_PART_CASES = {
    "finalized": Path("data/Finalized/probe_dialects.jsonl"),
    "unreviewed": Path("data/Unreviewed/probe_dialects.jsonl"),
    "legacy": Path("data/probe_dialects.jsonl"),
}

CLASSIFICATION_CANDIDATES = {
    "mcqa": ["A", "B", "C", "D"],
    "nli": ["entailment", "neutral", "contradiction"],
    "sentiment": [
        "Anger",
        "Disgust",
        "Enjoyment",
        "Fear",
        "Sadness",
        "Surprise",
        "Other",
    ],
}


@dataclass(frozen=True)
class ProbeItem:
    case: dict
    variant_name: str
    prompt_strategy: str
    prompt: str


@dataclass
class LocalModelRunner:
    model: Any
    tokenizer: Any
    device: Any


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


def ensure_jsonl_append_newline(path: Path) -> None:
    if not path.exists() or path.stat().st_size == 0:
        return
    with path.open("rb+") as handle:
        handle.seek(-1, 2)
        if handle.read(1) != b"\n":
            handle.write(b"\n")


def load_cases(path: Path) -> list[dict]:
    if path.suffix == ".jsonl":
        with path.open(encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]

    if path.suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise SystemExit(f"Expected a JSON list in {path}")
        return data

    raise SystemExit(f"Unsupported cases file extension: {path.suffix}")


def load_run_config(path: Path | None) -> dict:
    if path is None:
        return {}
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def load_text_generator(model_id: str) -> LocalModelRunner:
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise SystemExit(
            "Model probing requires optional dependencies: transformers and torch."
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    )
    model.eval()
    device = next(model.parameters()).device
    return LocalModelRunner(model=model, tokenizer=tokenizer, device=device)


def format_chat_prompt(runner: LocalModelRunner, prompt: str) -> str:
    messages = [{"role": "user", "content": prompt}]
    if getattr(runner.tokenizer, "chat_template", None):
        return runner.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    return prompt


def generate(runner: LocalModelRunner, prompt: str, max_new_tokens: int) -> str:
    import torch

    chat_prompt = format_chat_prompt(runner, prompt)
    inputs = runner.tokenizer(chat_prompt, return_tensors="pt").to(runner.device)
    input_length = inputs["input_ids"].shape[-1]
    with torch.inference_mode():
        output_ids = runner.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=runner.tokenizer.pad_token_id,
            eos_token_id=runner.tokenizer.eos_token_id,
        )
    generated_ids = output_ids[0, input_length:]
    return runner.tokenizer.decode(generated_ids, skip_special_tokens=True)


def generate_batch(
    runner: LocalModelRunner,
    prompts: list[str],
    max_new_tokens: int,
    batch_size: int,
) -> list[str]:
    return [generate(runner, prompt, max_new_tokens=max_new_tokens) for prompt in prompts]


def candidate_completion(task: str, label: str) -> str:
    if task in {"sentiment", "nli"}:
        return json.dumps({"label": label}, ensure_ascii=False, separators=(",", ":"))
    if task == "mcqa":
        return json.dumps({"answer": label}, ensure_ascii=False, separators=(",", ":"))
    raise ValueError(f"Task '{task}' has no fixed label candidates.")


def softmax(scores: dict[str, float]) -> dict[str, float]:
    if not scores:
        return {}
    max_score = max(scores.values())
    exp_scores = {
        label: math.exp(score - max_score) if math.isfinite(score) else 0.0
        for label, score in scores.items()
    }
    total = sum(exp_scores.values())
    if total <= 0:
        uniform = 1.0 / len(scores)
        return {label: uniform for label in scores}
    return {label: value / total for label, value in exp_scores.items()}


def score_completion(
    runner: LocalModelRunner,
    prompt: str,
    completion: str,
) -> dict[str, float | int]:
    import torch

    chat_prompt = format_chat_prompt(runner, prompt)
    prompt_ids = runner.tokenizer(
        chat_prompt,
        add_special_tokens=True,
        return_tensors="pt",
    )["input_ids"][0]
    completion_ids = runner.tokenizer(
        completion,
        add_special_tokens=False,
        return_tensors="pt",
    )["input_ids"][0]
    if completion_ids.numel() == 0:
        return {"sequence_logprob": float("-inf"), "avg_token_logprob": float("-inf"), "num_tokens": 0}

    input_ids = torch.cat([prompt_ids, completion_ids]).unsqueeze(0).to(runner.device)
    with torch.inference_mode():
        logits = runner.model(input_ids=input_ids).logits

    log_probs = torch.log_softmax(logits[:, :-1, :], dim=-1)
    labels = input_ids[:, 1:]
    token_log_probs = log_probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)

    start = prompt_ids.numel() - 1
    end = start + completion_ids.numel()
    completion_log_probs = token_log_probs[0, start:end]
    sequence_logprob = float(completion_log_probs.sum().item())
    avg_token_logprob = sequence_logprob / int(completion_ids.numel())
    return {
        "sequence_logprob": sequence_logprob,
        "avg_token_logprob": avg_token_logprob,
        "num_tokens": int(completion_ids.numel()),
    }


def score_label_distribution(
    runner: LocalModelRunner,
    item: ProbeItem,
) -> dict[str, Any] | None:
    task = item.case["task"]
    candidates = CLASSIFICATION_CANDIDATES.get(task)
    if not candidates:
        return None

    sequence_logprobs = {}
    label_logprobs = {}
    token_counts = {}
    for label in candidates:
        scored = score_completion(runner, item.prompt, candidate_completion(task, label))
        sequence_logprobs[label] = scored["sequence_logprob"]
        label_logprobs[label] = scored["avg_token_logprob"]
        token_counts[label] = scored["num_tokens"]

    label_probs = softmax(label_logprobs)
    predicted_label = max(label_probs, key=label_probs.get)
    confidence = label_probs[predicted_label]
    gold = gold_value(item.case)
    gold_label = str(gold) if isinstance(gold, str) else None

    return {
        "label_candidates": candidates,
        "label_sequence_logprobs": sequence_logprobs,
        "label_logprobs": label_logprobs,
        "label_token_counts": token_counts,
        "label_probs": label_probs,
        "confidence": confidence,
        "prob_prediction": predicted_label,
        "gold_prob": label_probs.get(gold_label) if gold_label else None,
    }


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


def build_probe_items(
    case: dict,
    skip_empty_variants: bool,
    prompt_strategy: str,
) -> list[ProbeItem]:
    items = []
    for variant_name, variant in case_variants(case).items():
        if skip_empty_variants and is_empty_variant(variant):
            continue
        prompt = build_prompt(case, variant_name, variant, prompt_strategy=prompt_strategy)
        items.append(
            ProbeItem(
                case=case,
                variant_name=variant_name,
                prompt_strategy=prompt_strategy,
                prompt=prompt,
            )
        )
    return items


def output_row(
    item: ProbeItem,
    model_name: str,
    model_id: str,
    raw: str,
    dataset_part: str,
    label_distribution: dict[str, Any] | None = None,
) -> dict:
    case = item.case
    parsed = parse_prediction(case["task"], raw)
    row = {
        "id": case.get("id"),
        "task": case["task"],
        "dataset_part": case.get("dataset_part") or dataset_part,
        "model_name": model_name,
        "model_id": model_id,
        "prompt_strategy": item.prompt_strategy,
        "variant": item.variant_name,
        "dialect_group": item.variant_name if item.variant_name != "standard" else "standard",
        "gold": gold_value(case),
        "prediction": parsed,
        "raw_output": raw.strip(),
    }
    if label_distribution is not None:
        row.update(label_distribution)
    return row


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


def resolve_cases_path(
    args_cases: Path | None,
    run_config: dict,
    dataset_part: str,
    dataset_part_from_cli: bool,
) -> Path:
    if args_cases is not None:
        return args_cases
    if dataset_part_from_cli and dataset_part in DATASET_PART_CASES:
        return DATASET_PART_CASES[dataset_part]
    if run_config.get("cases"):
        return Path(run_config["cases"])
    if dataset_part in DATASET_PART_CASES:
        return DATASET_PART_CASES[dataset_part]
    known = ", ".join(sorted(DATASET_PART_CASES))
    raise SystemExit(
        f"Unknown dataset_part '{dataset_part}' and no cases path was provided. "
        f"Known dataset parts: {known}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run zero-shot LLM probes on standard and dialect variants."
    )
    parser.add_argument("--config", type=Path, default=Path("configs/probe.yaml"))
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
    parser.add_argument("--batch-size", type=int)
    parser.add_argument(
        "--prompt-strategy",
        choices=sorted(PROMPT_STRATEGIES),
        help="Prompt setting to use: direct or normalized_direct.",
    )
    parser.add_argument("--include-empty-variants", action="store_true")
    parser.add_argument(
        "--no-collect-probabilities",
        action="store_true",
        help="Disable fixed-label probability scoring for MCQA, NLI, and sentiment.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Ignore existing output rows and replace the output JSONL.",
    )
    args = parser.parse_args()

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
    output_path = args.output or Path(run_config.get("output", "outputs/model_probe.jsonl"))
    max_new_tokens = args.max_new_tokens or int(run_config.get("max_new_tokens", 64))
    prompt_strategy = args.prompt_strategy or run_config.get("prompt_strategy", "direct")
    batch_size = (
        args.batch_size
        if args.batch_size is not None
        else int(run_config.get("batch_size", 1))
    )
    skip_empty_variants = not (
        args.include_empty_variants or bool(run_config.get("include_empty_variants", False))
    )
    collect_probabilities = (
        bool(run_config.get("collect_probabilities", True))
        and not args.no_collect_probabilities
    )

    if not model_name:
        raise SystemExit("Missing model_name. Set it in configs/probe.yaml or pass --model-name.")
    if prompt_strategy not in PROMPT_STRATEGIES:
        known = ", ".join(sorted(PROMPT_STRATEGIES))
        raise SystemExit(f"Unknown prompt_strategy '{prompt_strategy}'. Known: {known}")
    if batch_size < 1:
        raise SystemExit("batch_size must be at least 1.")

    model_spec = find_model(models_path, model_name)
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

    output_path.parent.mkdir(parents=True, exist_ok=True)
    completed = set() if args.overwrite else load_existing_keys(output_path)
    pending_items = [
        item
        for item in probe_items
        if (str(item.case.get("id")), str(item.variant_name)) not in completed
    ]

    print(
        f"Loaded {len(cases)} cases -> {len(probe_items)} probe items; "
        f"{len(completed)} existing rows; {len(pending_items)} pending for {model_name}.",
        flush=True,
    )
    if not pending_items:
        return

    generator = load_text_generator(model_spec["model_id"])
    if args.overwrite:
        output_mode = "w"
    else:
        ensure_jsonl_append_newline(output_path)
        output_mode = "a"

    with output_path.open(output_mode, encoding="utf-8") as handle:
        for batch_index, batch in enumerate(chunks(pending_items, batch_size), start=1):
            print(
                f"Batch {batch_index}: {batch[0].case.get('id')} {batch[0].variant_name}",
                flush=True,
            )
            raw_outputs = generate_batch(
                generator,
                [item.prompt for item in batch],
                max_new_tokens=max_new_tokens,
                batch_size=batch_size,
            )
            for item, raw in zip(batch, raw_outputs, strict=True):
                label_distribution = (
                    score_label_distribution(generator, item)
                    if collect_probabilities
                    else None
                )
                row = output_row(
                    item,
                    model_spec["name"],
                    model_spec["model_id"],
                    raw,
                    dataset_part=dataset_part,
                    label_distribution=label_distribution,
                )
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                handle.flush()


if __name__ == "__main__":
    main()
