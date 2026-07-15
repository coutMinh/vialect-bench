#!/usr/bin/env python3
"""
lexical_swap_nll.py
===================
Measures the NLL increase caused by each individual standard→dialect word substitution
by performing a targeted single-swap experiment.

For every 'replace' opcode in the alignment table, we:
  1. Take the baseline_text (standard sentence) for that sample.
  2. Replace ONLY the standard_span with the dialect_span (all other words unchanged).
  3. Compute NLL(swapped sentence) − NLL(original baseline) using the same fixed
     reference model as the intrinsic analysis (Qwen/Qwen2.5-0.5B).

This gives true per-substitution NLL attribution, not sentence-level correlation.

OUTPUTS (under --output_dir)
------------------------------
swap_nll_per_occurrence.csv    — one row per (sample_id, dialect, span_swap)
swap_nll_by_pair.csv           — mean/std Δ NLL per (standard_span, dialect_span, category)
swap_nll_by_category.csv       — mean/std Δ NLL aggregated per category
swap_nll_by_dialect.csv        — mean/std Δ NLL aggregated per dialect

Run from vialect_github/vialect-bench/
"""

from __future__ import annotations
import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE = Path(".")
ALIGN_CSV  = BASE / "outputs/lexical_substitution_analysis/word_word_alignment/raw_alignment_diff.csv"
PPL_CSV    = BASE / "outputs/finalized_6_complete_six_intrinsic_analysis/perplexity_pairwise_results.csv"
OUT_DIR    = BASE / "outputs/lexical_substitution_analysis/swap_nll_analysis"

DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B"
MAX_LENGTH    = 512

# Only these 9 categories are kept in the swap NLL experiment.
# Noise, Orthographic variant, Multi-word idiom, Out-of-scope, Unknown,
# and compound labels (containing ";") are all excluded.
KEEP_CATEGORIES = {
    "Interrogatives",
    "Pronouns",
    "Demonstratives/Deixis",
    "Negation/Function words",
    "Connectives/Aspect markers",
    "Predicate vocabulary",
    "Kinship/Honorific terms",
    "Discourse particles",
    "Idiomatic expressions",
}


# ---------------------------------------------------------------------------
# NLL computation
# ---------------------------------------------------------------------------

def compute_nll(text: str, tokenizer, model, device, max_length: int) -> float:
    """Mean per-token NLL for a single text string."""
    enc = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
    )
    input_ids = enc["input_ids"].to(device)
    if input_ids.shape[1] < 2:
        return float("nan")
    with torch.no_grad():
        out = model(input_ids=input_ids, labels=input_ids)
    return float(out.loss.item())  # mean per-token NLL


def make_swapped_text(baseline: str, standard_span: str, dialect_span: str) -> str | None:
    """
    Replace the FIRST occurrence of standard_span with dialect_span in baseline,
    using a word-boundary-aware match (case-insensitive).
    Returns None if standard_span is not found.
    """
    # Try exact substring first
    idx = baseline.lower().find(standard_span.lower())
    if idx == -1:
        return None
    # Preserve original casing of the rest
    return baseline[:idx] + dialect_span + baseline[idx + len(standard_span):]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Word-level NLL attribution via targeted single-swap experiment."
    )
    p.add_argument("--model",      default=DEFAULT_MODEL)
    p.add_argument("--output_dir", default=str(OUT_DIR))
    p.add_argument("--max_length", type=int, default=MAX_LENGTH)
    p.add_argument("--min_n",      type=int, default=2,
                   help="Minimum occurrences to include a pair in pair-level summary.")
    p.add_argument("--batch_size", type=int, default=1,
                   help="Sentences to process per batch (1 = sequential, safe for any GPU).")
    p.add_argument("--device",     default=None,
                   help="cuda / mps / cpu. Auto-detected if not set.")
    return p.parse_args()


def main():
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # --- Device ---
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    # --- Load model ---
    print(f"Loading {args.model} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float32, trust_remote_code=False
    ).to(device)
    model.eval()
    print("Model loaded.")

    # --- Load alignment table ---
    align = pd.read_csv(ALIGN_CSV)
    # Keep only: replace opcodes, single-label categories, in the 9 report categories
    align = align[align["opcode"] == "replace"].copy()
    align = align[~align["category"].str.contains(";", na=False)]
    align = align[align["category"].isin(KEEP_CATEGORIES)]
    align = align.dropna(subset=["standard_span", "dialect_span"])
    align = align[align["standard_span"].str.strip() != ""]
    align = align[align["dialect_span"].str.strip() != ""]
    print(f"Alignment rows to process: {len(align)}")

    # --- Load baseline texts ---
    ppl = pd.read_csv(PPL_CSV)[["sample_id", "target_dialect", "baseline_text", "orig_nll"]] \
            .drop_duplicates(subset=["sample_id", "target_dialect"])
    # Merge to get baseline_text per row
    align = align.merge(ppl, on=["sample_id", "target_dialect"], how="left")
    missing = align["baseline_text"].isna().sum()
    if missing:
        print(f"  WARNING: {missing} rows have no baseline_text — skipped.")
    align = align.dropna(subset=["baseline_text"])

    # --- Run single-swap NLL experiment ---
    records = []
    total = len(align)
    for i, (_, row) in enumerate(align.iterrows()):
        if i % 500 == 0:
            print(f"  Processing {i}/{total} ...")

        baseline  = str(row["baseline_text"])
        std_span  = str(row["standard_span"]).strip()
        dial_span = str(row["dialect_span"]).strip()

        swapped = make_swapped_text(baseline, std_span, dial_span)
        if swapped is None or swapped == baseline:
            # Span not found or no change after swap
            continue

        nll_baseline = compute_nll(baseline, tokenizer, model, device, args.max_length)
        nll_swapped  = compute_nll(swapped,  tokenizer, model, device, args.max_length)

        if np.isnan(nll_baseline) or np.isnan(nll_swapped):
            continue

        delta = nll_swapped - nll_baseline

        records.append({
            "sample_id":     row["sample_id"],
            "task":          row["task"],
            "target_dialect":row["target_dialect"],
            "standard_span": std_span,
            "dialect_span":  dial_span,
            "category":      row["category"],
            "nll_baseline":  nll_baseline,
            "nll_swapped":   nll_swapped,
            "delta_nll_swap":delta,
        })

    print(f"  Processed {len(records)} valid swaps.")
    per_occ = pd.DataFrame(records)
    per_occ.to_csv(out / "swap_nll_per_occurrence.csv", index=False)

    # --- Aggregate by (standard_span, dialect_span, category) ---
    pair_stats = (
        per_occ.groupby(["standard_span", "dialect_span", "category"])
        .agg(
            n_occurrences       =("delta_nll_swap", "count"),
            delta_nll_swap_mean =("delta_nll_swap", "mean"),
            delta_nll_swap_std  =("delta_nll_swap", "std"),
            delta_nll_swap_median=("delta_nll_swap", "median"),
            dialects            =("target_dialect", lambda x: ", ".join(sorted(set(x)))),
            tasks               =("task", lambda x: ", ".join(sorted(set(x)))),
        )
        .reset_index()
        .sort_values("delta_nll_swap_mean", ascending=False)
    )
    pair_stats.to_csv(out / "swap_nll_by_pair.csv", index=False)

    # Top pairs with n >= min_n
    pair_stats[pair_stats["n_occurrences"] >= args.min_n] \
        .head(50) \
        .to_csv(out / "swap_nll_top50_pairs.csv", index=False)

    # --- Aggregate by category ---
    cat_stats = (
        per_occ.groupby("category")
        .agg(
            n_occurrences       =("delta_nll_swap", "count"),
            delta_nll_swap_mean =("delta_nll_swap", "mean"),
            delta_nll_swap_std  =("delta_nll_swap", "std"),
            delta_nll_swap_median=("delta_nll_swap","median"),
        )
        .reset_index()
        .sort_values("delta_nll_swap_mean", ascending=False)
    )
    cat_stats.to_csv(out / "swap_nll_by_category.csv", index=False)

    # --- Aggregate by dialect ---
    dial_stats = (
        per_occ.groupby("target_dialect")
        .agg(
            n_occurrences       =("delta_nll_swap", "count"),
            delta_nll_swap_mean =("delta_nll_swap", "mean"),
            delta_nll_swap_std  =("delta_nll_swap", "std"),
        )
        .reset_index()
        .sort_values("delta_nll_swap_mean", ascending=False)
    )
    dial_stats.to_csv(out / "swap_nll_by_dialect.csv", index=False)

    # --- Print summary ---
    print("\n" + "="*60)
    print("SWAP NLL ATTRIBUTION — SUMMARY")
    print("="*60)
    print(f"\nTotal valid swaps processed: {len(per_occ)}")
    print(f"\nCategory ranking by mean Δ NLL (word-swap level):")
    print(f"{'Category':<35} {'mean Δ NLL':>10}  {'n':>6}")
    print("-"*55)
    for _, r in cat_stats.iterrows():
        print(f"{r['category']:<35} {r['delta_nll_swap_mean']:>10.4f}  {int(r['n_occurrences']):>6}")

    print(f"\nTop 10 substitution pairs by mean Δ NLL (n >= {args.min_n}):")
    top = pair_stats[pair_stats["n_occurrences"] >= args.min_n].head(10)
    for _, r in top.iterrows():
        print(f"  {r['standard_span']}→{r['dialect_span']:<12} [{r['category']}]"
              f"  Δ={r['delta_nll_swap_mean']:.4f}  n={int(r['n_occurrences'])}")

    print(f"\nAll outputs saved to: {out}")
    for p in sorted(out.iterdir()):
        if p.suffix == ".csv":
            print(f"  {p.name}  ({p.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
