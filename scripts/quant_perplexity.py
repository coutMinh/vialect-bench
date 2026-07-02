from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm


def sentence_nll_and_ppl(
    text: Any,
    tokenizer,
    model,
    device: str,
    max_length: int = 512,
) -> dict[str, float | int]:
    text = str(text).strip()

    if not text:
        return {
            "n_tokens": 0,
            "nll": np.nan,
            "ppl": np.nan,
        }

    enc = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
    )

    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    import torch

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=input_ids,
        )

    nll = float(outputs.loss.item())
    ppl = float(np.exp(nll)) if np.isfinite(nll) else np.nan

    return {
        "n_tokens": int(attention_mask.sum().item()),
        "nll": nll,
        "ppl": ppl,
    }


def bootstrap_ci(
    values: pd.Series | np.ndarray,
    n_boot: int = 5000,
    ci: int = 95,
    seed: int = 42,
) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]

    if len(values) == 0:
        return np.nan, np.nan

    rng = np.random.default_rng(seed)
    means = []

    for _ in range(n_boot):
        sample = rng.choice(values, size=len(values), replace=True)
        means.append(np.mean(sample))

    low = np.percentile(means, (100 - ci) / 2)
    high = np.percentile(means, 100 - (100 - ci) / 2)

    return float(low), float(high)


def summarize(
    df: pd.DataFrame,
    group_col: str | None = None,
    n_boot: int = 5000,
) -> pd.DataFrame:
    rows = []

    if group_col and group_col in df.columns:
        groups = list(df.groupby(group_col, dropna=False))
    else:
        groups = [("ALL", df)]

    for group_name, group_df in groups:
        delta_ppl = group_df["delta_ppl"].dropna()
        ratio_ppl = group_df["ratio_ppl"].replace([np.inf, -np.inf], np.nan).dropna()
        delta_nll = group_df["delta_nll"].dropna()

        ci_low, ci_high = bootstrap_ci(delta_nll, n_boot=n_boot)

        rows.append(
            {
                "group": group_name,
                "n": len(group_df),
                "orig_ppl_mean": group_df["orig_ppl"].mean(),
                "orig_ppl_median": group_df["orig_ppl"].median(),
                "para_ppl_mean": group_df["para_ppl"].mean(),
                "para_ppl_median": group_df["para_ppl"].median(),
                "delta_ppl_mean": delta_ppl.mean(),
                "delta_ppl_median": delta_ppl.median(),
                "ratio_ppl_mean": ratio_ppl.mean(),
                "ratio_ppl_median": ratio_ppl.median(),
                "delta_nll_mean": delta_nll.mean(),
                "delta_nll_median": delta_nll.median(),
                "delta_nll_bootstrap_ci_low": ci_low,
                "delta_nll_bootstrap_ci_high": ci_high,
                "orig_len_mean": group_df["orig_len"].mean(),
                "para_len_mean": group_df["para_len"].mean(),
                "length_ratio_mean": group_df["length_ratio"].mean(),
            }
        )

    return pd.DataFrame(rows)


def read_input(path: Path) -> pd.DataFrame:
    if path.suffix == ".csv":
        return pd.read_csv(path)
    if path.suffix == ".jsonl":
        return pd.read_json(path, lines=True)
    if path.suffix == ".json":
        return pd.read_json(path)
    raise ValueError("Input must be .csv, .json, or .jsonl")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute dataset-level paired perplexity gaps between standard and "
            "dialect texts using a fixed reference causal LM as a scorer."
        )
    )
    parser.add_argument("--input", required=True, help="Input CSV/JSON/JSONL file")
    parser.add_argument("--output_dir", default="outputs/perplexity_analysis")
    parser.add_argument(
        "--model",
        "--reference_model",
        dest="model",
        default="Qwen/Qwen2.5-0.5B",
        help="Reference causal LM used only as a data-scoring instrument.",
    )
    parser.add_argument("--orig_col", default="original_text")
    parser.add_argument("--para_col", default="dialect_text")
    parser.add_argument("--group_col", default="target_dialect")
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--n_boot", type=int, default=5000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = read_input(input_path)

    for col in [args.orig_col, args.para_col]:
        if col not in df.columns:
            raise ValueError(f"Missing column: {col}")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        trust_remote_code=True,
    ).to(device)
    model.eval()

    results = []

    for _, row in tqdm(df.iterrows(), total=len(df)):
        orig_text = row[args.orig_col]
        para_text = row[args.para_col]

        orig_score = sentence_nll_and_ppl(
            orig_text,
            tokenizer,
            model,
            device,
            max_length=args.max_length,
        )
        para_score = sentence_nll_and_ppl(
            para_text,
            tokenizer,
            model,
            device,
            max_length=args.max_length,
        )

        orig_len = len(str(orig_text).split())
        para_len = len(str(para_text).split())

        item = row.to_dict()
        item.update(
            {
                "orig_n_tokens": orig_score["n_tokens"],
                "orig_nll": orig_score["nll"],
                "orig_ppl": orig_score["ppl"],
                "para_n_tokens": para_score["n_tokens"],
                "para_nll": para_score["nll"],
                "para_ppl": para_score["ppl"],
                "delta_nll": para_score["nll"] - orig_score["nll"],
                "delta_ppl": para_score["ppl"] - orig_score["ppl"],
                "ratio_ppl": (
                    para_score["ppl"] / orig_score["ppl"]
                    if orig_score["ppl"] > 0
                    else np.nan
                ),
                "orig_len": orig_len,
                "para_len": para_len,
                "length_ratio": para_len / orig_len if orig_len > 0 else np.nan,
            }
        )

        results.append(item)

    result_df = pd.DataFrame(results)

    result_path = output_dir / "perplexity_pairwise_results.csv"
    result_df.to_csv(result_path, index=False)

    summary_df = summarize(result_df, args.group_col, n_boot=args.n_boot)
    summary_path = output_dir / "perplexity_summary_by_group.csv"
    summary_df.to_csv(summary_path, index=False)

    overall_summary = summarize(result_df, n_boot=args.n_boot)
    overall_path = output_dir / "perplexity_summary_overall.csv"
    overall_summary.to_csv(overall_path, index=False)

    metadata = {
        "input": str(input_path),
        "reference_model": args.model,
        "orig_col": args.orig_col,
        "para_col": args.para_col,
        "group_col": args.group_col,
        "max_length": args.max_length,
        "n_boot": args.n_boot,
        "n_samples": len(result_df),
    }

    with (output_dir / "perplexity_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)

    print("Saved:")
    print(result_path)
    print(summary_path)
    print(overall_path)


if __name__ == "__main__":
    main()
