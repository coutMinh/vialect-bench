from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .probe_models import (
    CLASSIFICATION_CANDIDATES,
    DATASET_PART_CASES,
    ProbeItem,
    build_probe_items,
    candidate_completion,
    chunks,
    find_model,
    gold_value,
    load_cases,
    load_run_config,
    resolve_cases_path,
)
from .prompting import PROMPT_STRATEGIES, parse_prediction


@dataclass
class Gemma3Runner:
    model: Any
    processor: Any
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
            if not str(row.get("raw_output") or "").strip() and not str(
                row.get("prediction") or ""
            ).strip():
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


def safe_number(value: float | int | None) -> float | int | None:
    if value is None:
        return None
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def sanitize_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: sanitize_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [sanitize_json(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def load_gemma3_runner(model_id: str) -> Gemma3Runner:
    try:
        import torch
        from transformers import AutoProcessor, Gemma3ForConditionalGeneration
    except ImportError as exc:
        raise SystemExit(
            "Gemma 3 probing requires torch and transformers with Gemma3 support. "
            "Install/upgrade with: pip install -U 'transformers>=4.50.0' torch"
        ) from exc

    processor = AutoProcessor.from_pretrained(model_id)
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is None:
        raise SystemExit("Gemma 3 AutoProcessor did not expose a tokenizer.")
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = Gemma3ForConditionalGeneration.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype=dtype,
    )
    model.eval()
    device = next(model.parameters()).device
    return Gemma3Runner(
        model=model,
        processor=processor,
        tokenizer=tokenizer,
        device=device,
    )


def gemma3_messages(prompt: str) -> list[dict[str, Any]]:
    return [{"role": "user", "content": [{"type": "text", "text": prompt}]}]


def encode_gemma3_prompt(runner: Gemma3Runner, prompt: str):
    inputs = runner.processor.apply_chat_template(
        gemma3_messages(prompt),
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )
    return inputs.to(runner.device)


def gemma3_chat_text(runner: Gemma3Runner, prompt: str) -> str:
    return runner.processor.apply_chat_template(
        gemma3_messages(prompt),
        add_generation_prompt=True,
        tokenize=False,
    )


def generate(runner: Gemma3Runner, prompt: str, max_new_tokens: int) -> str:
    import torch

    inputs = encode_gemma3_prompt(runner, prompt)
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
    return runner.processor.decode(generated_ids, skip_special_tokens=True).strip()


def generate_batch(
    runner: Gemma3Runner,
    prompts: list[str],
    max_new_tokens: int,
    batch_size: int,
) -> list[str]:
    return [generate(runner, prompt, max_new_tokens=max_new_tokens) for prompt in prompts]


def score_completion(
    runner: Gemma3Runner,
    prompt: str,
    completion: str,
) -> dict[str, float | int | None]:
    import torch

    chat_prompt = gemma3_chat_text(runner, prompt)
    prompt_ids = runner.tokenizer(
        chat_prompt,
        add_special_tokens=False,
        return_tensors="pt",
    )["input_ids"][0]
    completion_ids = runner.tokenizer(
        completion,
        add_special_tokens=False,
        return_tensors="pt",
    )["input_ids"][0]
    if completion_ids.numel() == 0:
        return {"sequence_logprob": None, "avg_token_logprob": None, "num_tokens": 0}

    input_ids = torch.cat([prompt_ids, completion_ids]).unsqueeze(0).to(runner.device)
    attention_mask = torch.ones_like(input_ids, device=runner.device)
    with torch.inference_mode():
        logits = runner.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        ).logits.float()

    if not torch.isfinite(logits).all():
        return {
            "sequence_logprob": None,
            "avg_token_logprob": None,
            "num_tokens": int(completion_ids.numel()),
        }

    log_probs = torch.log_softmax(logits[:, :-1, :], dim=-1)
    labels = input_ids[:, 1:]
    token_log_probs = log_probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)

    start = prompt_ids.numel() - 1
    end = start + completion_ids.numel()
    completion_log_probs = token_log_probs[0, start:end]
    if not torch.isfinite(completion_log_probs).all():
        return {
            "sequence_logprob": None,
            "avg_token_logprob": None,
            "num_tokens": int(completion_ids.numel()),
        }

    sequence_logprob = float(completion_log_probs.sum().item())
    avg_token_logprob = sequence_logprob / int(completion_ids.numel())
    return {
        "sequence_logprob": safe_number(sequence_logprob),
        "avg_token_logprob": safe_number(avg_token_logprob),
        "num_tokens": int(completion_ids.numel()),
    }


def softmax(scores: dict[str, float | None]) -> dict[str, float]:
    finite_scores = {
        label: score
        for label, score in scores.items()
        if score is not None and math.isfinite(float(score))
    }
    if not finite_scores:
        uniform = 1.0 / len(scores) if scores else 0.0
        return {label: uniform for label in scores}

    max_score = max(finite_scores.values())
    exp_scores = {
        label: math.exp(float(score) - float(max_score))
        if score is not None and math.isfinite(float(score))
        else 0.0
        for label, score in scores.items()
    }
    total = sum(exp_scores.values())
    if total <= 0:
        uniform = 1.0 / len(scores)
        return {label: uniform for label in scores}
    return {label: value / total for label, value in exp_scores.items()}


def score_label_distribution(
    runner: Gemma3Runner,
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


def output_row(
    item: ProbeItem,
    model_name: str,
    model_id: str,
    raw: str,
    dataset_part: str,
    label_distribution: dict[str, Any] | None = None,
) -> dict[str, Any]:
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
        "dialect_group": item.variant_name
        if item.variant_name != "standard"
        else "standard",
        "gold": gold_value(case),
        "prediction": parsed,
        "raw_output": raw.strip(),
    }
    if label_distribution is not None:
        row.update(label_distribution)
    return sanitize_json(row)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Gemma 3 probes with the Gemma3ForConditionalGeneration API."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/direct_prompting/gemma_3_4b_it.yaml"),
    )
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
    parser.add_argument(
        "--limit",
        type=int,
        help="Run only the first N pending probe items for a smoke test.",
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
    model_name = args.model_name or run_config.get("model_name", "Gemma 3 4B IT")
    output_path = args.output or Path(
        run_config.get("output", "outputs/direct_prompting/gemma_3_4b_it.jsonl")
    )
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
    if args.limit is not None:
        pending_items = pending_items[: args.limit]

    print(
        f"Loaded {len(cases)} cases -> {len(probe_items)} probe items; "
        f"{len(completed)} existing valid rows; {len(pending_items)} pending for {model_name}.",
        flush=True,
    )
    if not pending_items:
        return

    runner = load_gemma3_runner(model_spec["model_id"])
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
                runner,
                [item.prompt for item in batch],
                max_new_tokens=max_new_tokens,
                batch_size=batch_size,
            )
            for item, raw in zip(batch, raw_outputs, strict=True):
                label_distribution = (
                    score_label_distribution(runner, item)
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
                handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                handle.flush()


if __name__ == "__main__":
    main()
