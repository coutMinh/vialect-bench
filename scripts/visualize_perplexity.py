from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from quant_perplexity import sentence_nll_and_ppl, summarize


DEFAULT_DIALECTS = ["PNB", "PNN", "PNT1", "PNT2", "PNT3", "PNT4"]


def read_input(path: Path) -> pd.DataFrame:
    if path.suffix == ".csv":
        return pd.read_csv(path)
    if path.suffix == ".jsonl":
        return pd.read_json(path, lines=True)
    if path.suffix == ".json":
        return pd.read_json(path)
    raise ValueError("Input must be .csv, .json, or .jsonl")


def sample_group_from_id(sample_id: Any) -> str:
    parts = str(sample_id).split("_")
    if len(parts) >= 3 and parts[-1].isdigit():
        return "_".join(parts[:-1])
    return str(sample_id)


def resolve_orig_text(
    df: pd.DataFrame,
    task_col: str,
    orig_col: str,
    hyp_col: str,
) -> pd.Series:
    result = df[orig_col].copy()
    if task_col in df.columns and hyp_col in df.columns:
        nli_mask = df[task_col].astype(str).str.upper() == "NLI"
        result.loc[nli_mask] = df.loc[nli_mask, hyp_col]
    return result


def filter_complete_sample_groups(
    df: pd.DataFrame,
    sample_id_col: str,
    dialect_col: str,
    dialects: list[str],
    sample_group_mode: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    data = df.copy()
    data = data.drop_duplicates(subset=[sample_id_col, dialect_col], keep="first")
    if sample_group_mode == "exact":
        data["sample_group"] = data[sample_id_col].astype(str)
    elif sample_group_mode == "base_id":
        data["sample_group"] = data[sample_id_col].map(sample_group_from_id)
    else:
        raise ValueError("sample_group_mode must be 'exact' or 'base_id'")
    dialect_set = set(dialects)

    group_rows = []
    complete_groups = []
    for sample_group, group_df in data.groupby("sample_group", sort=True):
        present = sorted(set(group_df[dialect_col].dropna()))
        is_complete = dialect_set.issubset(present)
        group_rows.append(
            {
                "sample_group": sample_group,
                "n_rows": len(group_df),
                "n_dialects": len(present),
                "dialects_present": ",".join(present),
                "is_complete": is_complete,
            }
        )
        if is_complete:
            complete_groups.append(sample_group)

    filtered = data[data["sample_group"].isin(complete_groups)].copy()
    filtered = filtered[filtered[dialect_col].isin(dialects)].copy()
    return filtered, pd.DataFrame(group_rows)


def choose_device(torch_module) -> str:
    if torch_module.cuda.is_available():
        return "cuda"
    if torch_module.backends.mps.is_available():
        return "mps"
    return "cpu"


def score_pairs(
    df: pd.DataFrame,
    model_name: str,
    orig_col: str,
    para_col: str,
    max_length: int,
) -> pd.DataFrame:
    import torch

    device = choose_device(torch)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.float16 if device == "cuda" else torch.float32,
        trust_remote_code=True,
    ).to(device)
    model.eval()

    results = []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        orig_text = row[orig_col]
        para_text = row[para_col]
        orig_score = sentence_nll_and_ppl(
            orig_text, tokenizer, model, device, max_length=max_length
        )
        para_score = sentence_nll_and_ppl(
            para_text, tokenizer, model, device, max_length=max_length
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

    return pd.DataFrame(results)


def svg_text(x: float, y: float, text: Any, **attrs: Any) -> str:
    attr_text = " ".join(f'{k.replace("_", "-")}="{html.escape(str(v))}"' for k, v in attrs.items())
    return f'<text x="{x:.2f}" y="{y:.2f}" {attr_text}>{html.escape(str(text))}</text>'


def write_clustered_ppl_svg(summary_df: pd.DataFrame, path: Path) -> None:
    data = summary_df.copy()
    width, height = 980, 560
    margin_left, margin_right, margin_top, margin_bottom = 86, 34, 60, 96
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom
    max_value = float(data[["orig_ppl_mean", "para_ppl_mean"]].max().max())
    y_max = max_value * 1.18 if max_value > 0 else 1.0
    n = len(data)
    group_w = plot_w / max(n, 1)
    bar_w = group_w * 0.28
    colors = {"orig": "#4C78A8", "para": "#F58518"}

    def y_pos(value: float) -> float:
        return margin_top + plot_h - (float(value) / y_max) * plot_h

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        svg_text(width / 2, 30, "Original vs. Dialect Paraphrase Perplexity", text_anchor="middle", font_size=22, font_weight="700", fill="#1f2937"),
        svg_text(width / 2, 52, "Filtered to sample groups containing all six dialects", text_anchor="middle", font_size=13, fill="#6b7280"),
        f'<line x1="{margin_left}" y1="{margin_top + plot_h}" x2="{margin_left + plot_w}" y2="{margin_top + plot_h}" stroke="#374151" stroke-width="1"/>',
        f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{margin_top + plot_h}" stroke="#374151" stroke-width="1"/>',
    ]

    for i in range(6):
        tick = y_max * i / 5
        y = y_pos(tick)
        parts.append(f'<line x1="{margin_left}" y1="{y:.2f}" x2="{margin_left + plot_w}" y2="{y:.2f}" stroke="#e5e7eb"/>')
        parts.append(svg_text(margin_left - 10, y + 4, f"{tick:.0f}", text_anchor="end", font_size=12, fill="#4b5563"))

    for idx, row in data.reset_index(drop=True).iterrows():
        cx = margin_left + idx * group_w + group_w / 2
        orig_x = cx - bar_w - 3
        para_x = cx + 3
        for key, x, value in [
            ("orig", orig_x, row["orig_ppl_mean"]),
            ("para", para_x, row["para_ppl_mean"]),
        ]:
            y = y_pos(value)
            h = margin_top + plot_h - y
            parts.append(f'<rect x="{x:.2f}" y="{y:.2f}" width="{bar_w:.2f}" height="{h:.2f}" fill="{colors[key]}" rx="2"/>')
            parts.append(svg_text(x + bar_w / 2, y - 6, f"{value:.1f}", text_anchor="middle", font_size=11, fill="#111827"))
        parts.append(svg_text(cx, margin_top + plot_h + 28, row["group"], text_anchor="middle", font_size=13, font_weight="700", fill="#111827"))

    legend_x = margin_left + plot_w - 250
    legend_y = margin_top - 34
    parts.extend(
        [
            f'<rect x="{legend_x}" y="{legend_y}" width="14" height="14" fill="{colors["orig"]}"/>',
            svg_text(legend_x + 22, legend_y + 12, "Original PPL", font_size=13, fill="#111827"),
            f'<rect x="{legend_x + 124}" y="{legend_y}" width="14" height="14" fill="{colors["para"]}"/>',
            svg_text(legend_x + 146, legend_y + 12, "Dialect PPL", font_size=13, fill="#111827"),
            svg_text(24, margin_top + plot_h / 2, "Mean PPL", transform=f"rotate(-90 24 {margin_top + plot_h / 2})", text_anchor="middle", font_size=14, fill="#374151"),
        ]
    )
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def write_delta_nll_svg(summary_df: pd.DataFrame, path: Path) -> None:
    data = summary_df.copy()
    width, height = 980, 520
    margin_left, margin_right, margin_top, margin_bottom = 86, 34, 60, 86
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom
    min_value = min(0.0, float(data["delta_nll_mean"].min()))
    max_value = max(0.0, float(data["delta_nll_mean"].max()))
    pad = max((max_value - min_value) * 0.15, 0.1)
    y_min, y_max = min_value - pad, max_value + pad
    n = len(data)
    group_w = plot_w / max(n, 1)
    bar_w = group_w * 0.48

    def y_pos(value: float) -> float:
        return margin_top + (y_max - value) / (y_max - y_min) * plot_h

    zero_y = y_pos(0.0)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        svg_text(width / 2, 30, "Mean Delta NLL by Dialect", text_anchor="middle", font_size=22, font_weight="700", fill="#1f2937"),
        svg_text(width / 2, 52, "Positive values indicate stronger dialectal distribution shift", text_anchor="middle", font_size=13, fill="#6b7280"),
        f'<line x1="{margin_left}" y1="{zero_y:.2f}" x2="{margin_left + plot_w}" y2="{zero_y:.2f}" stroke="#374151" stroke-width="1"/>',
        f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{margin_top + plot_h}" stroke="#374151" stroke-width="1"/>',
    ]

    for i in range(6):
        tick = y_min + (y_max - y_min) * i / 5
        y = y_pos(tick)
        parts.append(f'<line x1="{margin_left}" y1="{y:.2f}" x2="{margin_left + plot_w}" y2="{y:.2f}" stroke="#e5e7eb"/>')
        parts.append(svg_text(margin_left - 10, y + 4, f"{tick:.2f}", text_anchor="end", font_size=12, fill="#4b5563"))

    for idx, row in data.reset_index(drop=True).iterrows():
        cx = margin_left + idx * group_w + group_w / 2
        value = float(row["delta_nll_mean"])
        y = y_pos(max(value, 0))
        h = abs(y_pos(value) - zero_y)
        if value < 0:
            y = zero_y
        color = "#54A24B" if value >= 0 else "#E45756"
        parts.append(f'<rect x="{cx - bar_w / 2:.2f}" y="{y:.2f}" width="{bar_w:.2f}" height="{h:.2f}" fill="{color}" rx="2"/>')
        label_y = y - 8 if value >= 0 else y + h + 18
        parts.append(svg_text(cx, label_y, f"{value:.3f}", text_anchor="middle", font_size=12, fill="#111827"))
        ci_low = row.get("delta_nll_bootstrap_ci_low")
        ci_high = row.get("delta_nll_bootstrap_ci_high")
        if pd.notna(ci_low) and pd.notna(ci_high):
            y_lo = y_pos(float(ci_low))
            y_hi = y_pos(float(ci_high))
            parts.append(f'<line x1="{cx:.2f}" y1="{y_lo:.2f}" x2="{cx:.2f}" y2="{y_hi:.2f}" stroke="#1f2937" stroke-width="1.5"/>')
            parts.append(f'<line x1="{cx - 6:.2f}" y1="{y_lo:.2f}" x2="{cx + 6:.2f}" y2="{y_lo:.2f}" stroke="#1f2937" stroke-width="1.5"/>')
            parts.append(f'<line x1="{cx - 6:.2f}" y1="{y_hi:.2f}" x2="{cx + 6:.2f}" y2="{y_hi:.2f}" stroke="#1f2937" stroke-width="1.5"/>')
        parts.append(svg_text(cx, margin_top + plot_h + 28, row["group"], text_anchor="middle", font_size=13, font_weight="700", fill="#111827"))

    parts.append(svg_text(24, margin_top + plot_h / 2, "Mean delta NLL", transform=f"rotate(-90 24 {margin_top + plot_h / 2})", text_anchor="middle", font_size=14, fill="#374151"))
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Score and visualize perplexity gaps for sample groups that contain "
            "all requested dialect groups."
        )
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--output_dir", default="outputs/perplexity_unreviewed_visualization")
    parser.add_argument("--model", "--reference_model", dest="model", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--orig_col", default="original_text")
    parser.add_argument("--para_col", default="dialect_text")
    parser.add_argument("--dialect_col", default="target_dialect")
    parser.add_argument("--sample_id_col", default="sample_id")
    parser.add_argument("--task_col", default="task")
    parser.add_argument("--hypothesis_col", default="hypothesis")
    parser.add_argument(
        "--sample_group_mode",
        choices=["exact", "base_id"],
        default="exact",
        help=(
            "Use exact sample_id, or strip a trailing underscore-number "
            "for broader base-id grouping."
        ),
    )
    parser.add_argument("--dialects", nargs="+", default=DEFAULT_DIALECTS)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--n_boot", type=int, default=5000)
    parser.add_argument("--reuse_scores", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = read_input(input_path)
    required_cols = [args.orig_col, args.para_col, args.dialect_col, args.sample_id_col]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing column: {col}")

    filtered_df, coverage_df = filter_complete_sample_groups(
        df,
        args.sample_id_col,
        args.dialect_col,
        args.dialects,
        args.sample_group_mode,
    )
    coverage_df.to_csv(output_dir / "sample_group_coverage.csv", index=False)
    filtered_df.to_csv(output_dir / "filtered_complete_sample_groups.csv", index=False)

    nli_orig_resolved = False
    if args.task_col in filtered_df.columns and args.hypothesis_col in filtered_df.columns:
        filtered_df["orig_text_eval"] = resolve_orig_text(
            filtered_df, args.task_col, args.orig_col, args.hypothesis_col
        )
        nli_count = int((filtered_df[args.task_col].astype(str).str.upper() == "NLI").sum())
        nli_orig_resolved = nli_count > 0
        scoring_orig_col = "orig_text_eval"
    else:
        scoring_orig_col = args.orig_col

    pairwise_path = output_dir / "perplexity_pairwise_filtered.csv"
    metadata = {
        "input": str(input_path),
        "reference_model": args.model,
        "orig_col": args.orig_col,
        "para_col": args.para_col,
        "dialect_col": args.dialect_col,
        "sample_id_col": args.sample_id_col,
        "task_col": args.task_col,
        "hypothesis_col": args.hypothesis_col,
        "nli_orig_text_resolution": (
            f"For NLI rows, hypothesis column '{args.hypothesis_col}' used as baseline "
            f"instead of '{args.orig_col}' (which is the premise)."
            if nli_orig_resolved
            else "No NLI rows present, or task/hypothesis columns missing; orig_col used as-is."
        ),
        "dialects": args.dialects,
        "sample_group_mode": args.sample_group_mode,
        "sample_group_rule": (
            "exact sample_id"
            if args.sample_group_mode == "exact"
            else "strip trailing underscore-number from sample_id, e.g. MCQA_0041_5 -> MCQA_0041"
        ),
        "n_input_rows": len(df),
        "n_complete_group_rows": len(filtered_df),
        "n_complete_sample_groups": int(filtered_df["sample_group"].nunique())
        if not filtered_df.empty
        else 0,
        "outputs": [
            "sample_group_coverage.csv",
            "filtered_complete_sample_groups.csv",
        ],
    }

    if filtered_df.empty:
        metadata["note"] = "No sample groups contain all requested dialect groups; scoring and plots were skipped."
        (output_dir / "metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print("No sample groups contain all requested dialect groups.")
    else:
        if args.reuse_scores and pairwise_path.exists():
            result_df = pd.read_csv(pairwise_path)
        else:
            result_df = score_pairs(
                filtered_df,
                args.model,
                scoring_orig_col,
                args.para_col,
                args.max_length,
            )
            result_df.to_csv(pairwise_path, index=False)

        summary_df = summarize(result_df, args.dialect_col, n_boot=args.n_boot)
        summary_path = output_dir / "perplexity_summary_by_dialect.csv"
        summary_df.to_csv(summary_path, index=False)

        write_clustered_ppl_svg(summary_df, output_dir / "perplexity_clustered_bar.svg")
        write_delta_nll_svg(summary_df, output_dir / "delta_nll_by_dialect.svg")
        metadata["outputs"].extend(
            [
                "perplexity_pairwise_filtered.csv",
                "perplexity_summary_by_dialect.csv",
                "perplexity_clustered_bar.svg",
                "delta_nll_by_dialect.svg",
            ]
        )
        (output_dir / "metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    print("Saved:")
    for item in metadata["outputs"]:
        print(output_dir / item)


if __name__ == "__main__":
    main()
