from __future__ import annotations

import argparse
import html
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm


DEFAULT_DIALECTS = ["PNB", "PNN", "PNT1", "PNT2", "PNT3", "PNT4"]


def read_input(path: Path) -> pd.DataFrame:
    if path.suffix == ".csv":
        return pd.read_csv(path)
    if path.suffix == ".jsonl":
        return pd.read_json(path, lines=True)
    if path.suffix == ".json":
        return pd.read_json(path)
    raise ValueError("Input must be .csv, .json, or .jsonl")


def word_len(text: Any) -> int:
    return len(str(text).strip().split())


def resolve_baseline_text(row: pd.Series, orig_col: str, hypothesis_col: str) -> str:
    if str(row.get("task", "")).upper() == "NLI" and hypothesis_col in row:
        return str(row.get(hypothesis_col, "")).strip()
    return str(row.get(orig_col, "")).strip()


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
    means = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        means[i] = rng.choice(values, size=len(values), replace=True).mean()

    low = np.percentile(means, (100 - ci) / 2)
    high = np.percentile(means, 100 - (100 - ci) / 2)
    return float(low), float(high)


def summarize_numeric(
    df: pd.DataFrame,
    group_cols: str | list[str] | None,
    metrics: list[str],
    n_boot: int = 5000,
) -> pd.DataFrame:
    if group_cols is None:
        groups = [("ALL", df)]
        group_col_names = ["group"]
    else:
        group_col_names = [group_cols] if isinstance(group_cols, str) else group_cols
        groups = list(df.groupby(group_col_names, dropna=False))

    rows: list[dict[str, Any]] = []
    for key, group_df in groups:
        if not isinstance(key, tuple):
            key = (key,)
        row: dict[str, Any] = {
            col: key[i] for i, col in enumerate(group_col_names)
        }
        row["n"] = len(group_df)
        for metric in metrics:
            values = group_df[metric].replace([np.inf, -np.inf], np.nan).dropna()
            row[f"{metric}_mean"] = float(values.mean()) if len(values) else np.nan
            row[f"{metric}_median"] = float(values.median()) if len(values) else np.nan
            row[f"{metric}_std"] = float(values.std(ddof=1)) if len(values) > 1 else np.nan
        if "delta_nll" in metrics:
            ci_low, ci_high = bootstrap_ci(group_df["delta_nll"], n_boot=n_boot)
            row["delta_nll_bootstrap_ci_low"] = ci_low
            row["delta_nll_bootstrap_ci_high"] = ci_high
        rows.append(row)

    return pd.DataFrame(rows)


def choose_device(torch_module) -> str:
    if torch_module.cuda.is_available():
        return "cuda"
    if torch_module.backends.mps.is_available():
        return "mps"
    return "cpu"


def sentence_nll_and_ppl(
    text: Any,
    tokenizer,
    model,
    device: str,
    max_length: int,
) -> dict[str, float | int]:
    text = str(text).strip()
    if not text:
        return {"n_tokens": 0, "nll": np.nan, "ppl": np.nan}

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
    return {
        "n_tokens": int(attention_mask.sum().item()),
        "nll": nll,
        "ppl": float(math.exp(nll)) if np.isfinite(nll) else np.nan,
    }


def score_unique_texts(
    texts: list[str],
    model_name: str,
    max_length: int,
    trust_remote_code: bool,
) -> dict[str, dict[str, float | int]]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = choose_device(torch)
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=trust_remote_code,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        trust_remote_code=trust_remote_code,
    ).to(device)
    model.eval()

    scores: dict[str, dict[str, float | int]] = {}
    for text in tqdm(texts, desc="Scoring unique texts"):
        scores[text] = sentence_nll_and_ppl(
            text,
            tokenizer,
            model,
            device,
            max_length=max_length,
        )
    return scores


def add_pairwise_scores(df: pd.DataFrame, score_cache: dict[str, dict[str, float | int]]) -> pd.DataFrame:
    rows = []
    for _, row in df.iterrows():
        baseline_text = str(row["baseline_text"])
        dialect_text = str(row["dialect_text_eval"])
        orig_score = score_cache[baseline_text]
        para_score = score_cache[dialect_text]

        orig_len = word_len(baseline_text)
        para_len = word_len(dialect_text)
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
        rows.append(item)
    return pd.DataFrame(rows)


def svg_text(x: float, y: float, text: Any, **attrs: Any) -> str:
    attr_text = " ".join(
        f'{key.replace("_", "-")}="{html.escape(str(value))}"'
        for key, value in attrs.items()
    )
    return f'<text x="{x:.2f}" y="{y:.2f}" {attr_text}>{html.escape(str(text))}</text>'


def write_grouped_bar_svg(
    df: pd.DataFrame,
    path: Path,
    group_col: str,
    value_cols: list[tuple[str, str, str]],
    title: str,
    subtitle: str,
    y_label: str,
    width: int = 980,
    height: int = 560,
) -> None:
    data = df.copy()
    margin_left, margin_right, margin_top, margin_bottom = 86, 34, 70, 96
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom
    max_value = float(data[[col for col, _, _ in value_cols]].max().max())
    min_value = min(0.0, float(data[[col for col, _, _ in value_cols]].min().min()))
    y_max = max_value * 1.16 if max_value > 0 else 1.0
    y_min = min_value * 1.16 if min_value < 0 else 0.0
    if y_max == y_min:
        y_max = y_min + 1.0

    n_groups = max(len(data), 1)
    group_w = plot_w / n_groups
    bar_w = group_w * 0.68 / max(len(value_cols), 1)

    def y_pos(value: float) -> float:
        return margin_top + (y_max - value) / (y_max - y_min) * plot_h

    zero_y = y_pos(0)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        svg_text(width / 2, 30, title, text_anchor="middle", font_size=22, font_weight="700", fill="#1f2937"),
        svg_text(width / 2, 54, subtitle, text_anchor="middle", font_size=13, fill="#6b7280"),
        f'<line x1="{margin_left}" y1="{zero_y:.2f}" x2="{margin_left + plot_w}" y2="{zero_y:.2f}" stroke="#374151" stroke-width="1"/>',
        f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{margin_top + plot_h}" stroke="#374151" stroke-width="1"/>',
    ]

    for i in range(6):
        tick = y_min + (y_max - y_min) * i / 5
        y = y_pos(tick)
        parts.append(f'<line x1="{margin_left}" y1="{y:.2f}" x2="{margin_left + plot_w}" y2="{y:.2f}" stroke="#e5e7eb"/>')
        parts.append(svg_text(margin_left - 10, y + 4, f"{tick:.2f}" if y_max <= 5 else f"{tick:.0f}", text_anchor="end", font_size=12, fill="#4b5563"))

    for idx, row in data.reset_index(drop=True).iterrows():
        cx = margin_left + idx * group_w + group_w / 2
        start_x = cx - (bar_w * len(value_cols)) / 2
        for j, (col, label, color) in enumerate(value_cols):
            value = float(row[col])
            x = start_x + j * bar_w
            y = y_pos(max(value, 0))
            h = abs(y_pos(value) - zero_y)
            if value < 0:
                y = zero_y
            parts.append(f'<rect x="{x:.2f}" y="{y:.2f}" width="{bar_w * 0.86:.2f}" height="{h:.2f}" fill="{color}" rx="2"/>')
            label_y = y - 6 if value >= 0 else y + h + 16
            parts.append(svg_text(x + bar_w * 0.43, label_y, f"{value:.2f}" if abs(value) < 10 else f"{value:.1f}", text_anchor="middle", font_size=10, fill="#111827"))
        parts.append(svg_text(cx, margin_top + plot_h + 30, row[group_col], text_anchor="middle", font_size=13, font_weight="700", fill="#111827"))

    legend_x = margin_left + plot_w - 320
    legend_y = margin_top - 36
    cursor_x = legend_x
    for _, label, color in value_cols:
        parts.append(f'<rect x="{cursor_x}" y="{legend_y}" width="14" height="14" fill="{color}"/>')
        parts.append(svg_text(cursor_x + 20, legend_y + 12, label, font_size=13, fill="#111827"))
        cursor_x += 118

    parts.append(svg_text(24, margin_top + plot_h / 2, y_label, transform=f"rotate(-90 24 {margin_top + plot_h / 2})", text_anchor="middle", font_size=14, fill="#374151"))
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def write_delta_nll_svg(summary_df: pd.DataFrame, path: Path, group_col: str) -> None:
    data = summary_df.copy()
    width, height = 980, 520
    margin_left, margin_right, margin_top, margin_bottom = 86, 34, 70, 86
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom
    min_value = min(0.0, float(data["delta_nll_mean"].min()))
    max_value = max(0.0, float(data["delta_nll_mean"].max()))
    pad = max((max_value - min_value) * 0.16, 0.08)
    y_min, y_max = min_value - pad, max_value + pad
    group_w = plot_w / max(len(data), 1)
    bar_w = group_w * 0.52

    def y_pos(value: float) -> float:
        return margin_top + (y_max - value) / (y_max - y_min) * plot_h

    zero_y = y_pos(0.0)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        svg_text(width / 2, 30, "Perplexity Gap: Delta NLL by Dialect", text_anchor="middle", font_size=22, font_weight="700", fill="#1f2937"),
        svg_text(width / 2, 54, "Positive values mean dialect paraphrases are more surprising to the reference LM", text_anchor="middle", font_size=13, fill="#6b7280"),
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
        color = "#E45756" if value >= 0 else "#54A24B"
        parts.append(f'<rect x="{cx - bar_w / 2:.2f}" y="{y:.2f}" width="{bar_w:.2f}" height="{h:.2f}" fill="{color}" rx="2"/>')
        label_y = y - 8 if value >= 0 else y + h + 18
        parts.append(svg_text(cx, label_y, f"{value:.3f}", text_anchor="middle", font_size=12, fill="#111827"))

        ci_low = row.get("delta_nll_bootstrap_ci_low")
        ci_high = row.get("delta_nll_bootstrap_ci_high")
        if pd.notna(ci_low) and pd.notna(ci_high):
            y_low = y_pos(float(ci_low))
            y_high = y_pos(float(ci_high))
            parts.append(f'<line x1="{cx:.2f}" y1="{y_low:.2f}" x2="{cx:.2f}" y2="{y_high:.2f}" stroke="#1f2937" stroke-width="1.5"/>')
            parts.append(f'<line x1="{cx - 6:.2f}" y1="{y_low:.2f}" x2="{cx + 6:.2f}" y2="{y_low:.2f}" stroke="#1f2937" stroke-width="1.5"/>')
            parts.append(f'<line x1="{cx - 6:.2f}" y1="{y_high:.2f}" x2="{cx + 6:.2f}" y2="{y_high:.2f}" stroke="#1f2937" stroke-width="1.5"/>')
        parts.append(svg_text(cx, margin_top + plot_h + 30, row[group_col], text_anchor="middle", font_size=13, font_weight="700", fill="#111827"))

    parts.append(svg_text(24, margin_top + plot_h / 2, "Mean delta NLL", transform=f"rotate(-90 24 {margin_top + plot_h / 2})", text_anchor="middle", font_size=14, fill="#374151"))
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def write_count_svg(count_df: pd.DataFrame, path: Path, group_col: str, title: str) -> None:
    data = count_df.copy()
    data["n"] = data["n"].astype(float)
    write_grouped_bar_svg(
        data,
        path,
        group_col=group_col,
        value_cols=[("n", "Count", "#4C78A8")],
        title=title,
        subtitle="Number of finalized rows in the dataset",
        y_label="Count",
    )


def format_float(value: Any, ndigits: int = 3) -> str:
    if pd.isna(value):
        return "NA"
    return f"{float(value):.{ndigits}f}"


def markdown_table(df: pd.DataFrame, columns: list[str], rename: dict[str, str] | None = None, ndigits: int = 3) -> str:
    rename = rename or {}
    lines = []
    headers = [rename.get(col, col) for col in columns]
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for _, row in df[columns].iterrows():
        cells = []
        for col in columns:
            value = row[col]
            if isinstance(value, float):
                cells.append(format_float(value, ndigits=ndigits))
            else:
                cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def write_report(
    output_dir: Path,
    metadata: dict[str, Any],
    dialect_counts: pd.DataFrame,
    task_counts: pd.DataFrame,
    overall_summary: pd.DataFrame | None,
    dialect_summary: pd.DataFrame | None,
    task_summary: pd.DataFrame | None,
) -> None:
    lines = [
        "# Báo cáo quantitative data analysis cho VialectBench finalized 5",
        "",
        "## Mục tiêu",
        "",
        "Phân tích này là intrinsic evaluation của dataset, không phải đánh giá năng lực của model. Reference causal language model chỉ được dùng như một scorer cố định để đo câu paraphrase phương ngữ lệch khỏi phân phối tiếng Việt chuẩn đến mức nào.",
        "",
        "## Cách chạy",
        "",
        "```bash",
        "python scripts/analyze_vialectbench_data.py \\",
        "  --input data/vialectbench_finalized_5.json \\",
        "  --output_dir outputs/finalized_5_intrinsic_analysis \\",
        "  --model Qwen/Qwen2.5-0.5B",
        "```",
        "",
        "Nếu chỉ muốn thống kê dataset và length ratio, chưa chạy perplexity:",
        "",
        "```bash",
        "python scripts/analyze_vialectbench_data.py \\",
        "  --input data/vialectbench_finalized_5.json \\",
        "  --skip_perplexity",
        "```",
        "",
        "## Các metric",
        "",
        "- `orig_len`, `para_len`: số token thô theo khoảng trắng của câu gốc và câu phương ngữ.",
        "- `length_ratio = para_len / orig_len`: kiểm tra paraphrase có bị dài/ngắn bất thường hay không. Giá trị gần 1 nghĩa là độ dài được giữ tương đối ổn định.",
        "- `NLL`: negative log-likelihood trung bình theo token từ reference LM. NLL càng cao nghĩa là chuỗi càng ít quen thuộc với LM.",
        "- `PPL = exp(NLL)`: perplexity, cách đọc trực quan hơn của NLL nhưng dễ bị scale lớn theo model/tokenizer.",
        "- `delta_nll = NLL_para - NLL_original`: metric chính. Dương nghĩa là câu dialect gây distribution shift so với câu chuẩn; âm nghĩa là câu dialect quen thuộc hơn với reference LM trong cặp đó.",
        "- `ratio_ppl = PPL_para / PPL_original`: tỉ lệ perplexity để đọc nhanh mức tăng/giảm, nhưng nên diễn giải phụ sau `delta_nll`.",
        "- `95% bootstrap CI`: khoảng tin cậy bootstrap cho trung bình `delta_nll`.",
        "",
        "Lưu ý quan trọng: với task NLI, baseline đúng là `hypothesis`, không phải `original_text`, vì `original_text` là premise còn `dialect_text` là hypothesis được viết lại.",
        "",
        "## Tổng quan dataset",
        "",
        f"- Tổng số dòng finalized: {metadata['n_rows']}",
        f"- Số sample_id duy nhất: {metadata['n_unique_samples']}",
        f"- Số sample_id có đủ 6 dialect: {metadata['n_complete_six_dialect_samples']}",
        f"- Chỉ giữ sample_id đủ 6 dialect khi chạy analysis: {metadata['complete_six_dialects_only']}",
        f"- Số cặp `(sample_id, target_dialect)` bị trùng: {metadata['n_duplicate_sample_dialect_pairs']}",
        f"- Dialect groups: {', '.join(metadata['dialects'])}",
        f"- Tasks: {', '.join(metadata['tasks'])}",
        "",
        "### Distribution theo dialect",
        "",
        markdown_table(dialect_counts, ["target_dialect", "n"], {"target_dialect": "Dialect", "n": "N"}),
        "",
        "### Distribution theo task",
        "",
        markdown_table(task_counts, ["task", "n"], {"task": "Task", "n": "N"}),
        "",
    ]

    if overall_summary is not None and dialect_summary is not None and task_summary is not None:
        overall = overall_summary.iloc[0]
        lines.extend(
            [
                "## Kết quả intrinsic evaluation",
                "",
                f"Trên toàn bộ dataset, `delta_nll_mean = {format_float(overall['delta_nll_mean'])}` với CI 95% [{format_float(overall['delta_nll_bootstrap_ci_low'])}, {format_float(overall['delta_nll_bootstrap_ci_high'])}], `ratio_ppl_mean = {format_float(overall['ratio_ppl_mean'])}`, và `length_ratio_mean = {format_float(overall['length_ratio_mean'])}`.",
                "",
                "Diễn giải: nếu `delta_nll_mean` dương, paraphrase phương ngữ nhìn chung khó dự đoán hơn với reference LM so với câu chuẩn cùng nội dung. Điều này không có nghĩa dữ liệu kém chất lượng; nó cho thấy dataset đưa vào tín hiệu dialectal variation có thể đo được.",
                "",
                "### Theo dialect",
                "",
                markdown_table(
                    dialect_summary,
                    [
                        "target_dialect",
                        "n",
                        "orig_ppl_mean",
                        "para_ppl_mean",
                        "delta_nll_mean",
                        "delta_nll_bootstrap_ci_low",
                        "delta_nll_bootstrap_ci_high",
                        "ratio_ppl_mean",
                        "length_ratio_mean",
                    ],
                    {
                        "target_dialect": "Dialect",
                        "n": "N",
                        "orig_ppl_mean": "Orig PPL",
                        "para_ppl_mean": "Dialect PPL",
                        "delta_nll_mean": "Delta NLL",
                        "delta_nll_bootstrap_ci_low": "CI low",
                        "delta_nll_bootstrap_ci_high": "CI high",
                        "ratio_ppl_mean": "PPL ratio",
                        "length_ratio_mean": "Length ratio",
                    },
                ),
                "",
                "### Theo task",
                "",
                markdown_table(
                    task_summary,
                    [
                        "task",
                        "n",
                        "orig_ppl_mean",
                        "para_ppl_mean",
                        "delta_nll_mean",
                        "delta_nll_bootstrap_ci_low",
                        "delta_nll_bootstrap_ci_high",
                        "ratio_ppl_mean",
                        "length_ratio_mean",
                    ],
                    {
                        "task": "Task",
                        "n": "N",
                        "orig_ppl_mean": "Orig PPL",
                        "para_ppl_mean": "Dialect PPL",
                        "delta_nll_mean": "Delta NLL",
                        "delta_nll_bootstrap_ci_low": "CI low",
                        "delta_nll_bootstrap_ci_high": "CI high",
                        "ratio_ppl_mean": "PPL ratio",
                        "length_ratio_mean": "Length ratio",
                    },
                ),
                "",
                "## Insight chính",
                "",
                "- `delta_nll` là bằng chứng định lượng rằng các paraphrase phương ngữ tạo distribution shift so với tiếng Việt chuẩn, trong khi vẫn giữ thiết kế paired theo cùng sample.",
                "- `length_ratio` giúp kiểm tra shift này không chỉ đến từ việc câu dialect dài/ngắn bất thường. Nếu length ratio gần 1 thì gap perplexity đáng tin hơn như tín hiệu dialectal surface variation.",
                "- So sánh theo dialect cho biết nhóm vùng nào tạo shift mạnh hơn/yếu hơn; so sánh theo task giúp tránh kết luận sai do MCQA, QA, NLI, SENT có độ dài và format rất khác nhau.",
            ]
        )
    else:
        lines.extend(
            [
                "## Kết quả intrinsic evaluation",
                "",
                "Lần chạy này dùng `--skip_perplexity`, nên chỉ có thống kê dataset và length ratio. Chạy lại không có flag này để tạo bảng perplexity/NLL.",
            ]
        )

    lines.extend(
        [
            "",
            "## Output files",
            "",
            "- `dataset_metadata.json`: metadata và cấu hình chạy.",
            "- `distribution_by_dialect.csv`, `distribution_by_task.csv`, `distribution_task_x_dialect.csv`: phân bố dữ liệu.",
            "- `length_summary_overall.csv`, `length_summary_by_dialect.csv`, `length_summary_by_task.csv`: length ratio.",
            "- `perplexity_pairwise_results.csv`: điểm NLL/PPL theo từng cặp câu.",
            "- `perplexity_summary_overall.csv`, `perplexity_summary_by_dialect.csv`, `perplexity_summary_by_task.csv`: bảng summary chính.",
            "- `fig_ppl_by_dialect.svg`, `fig_delta_nll_by_dialect.svg`, `fig_count_by_dialect.svg`, `fig_count_by_task.svg`: hình minh họa.",
        ]
    )

    (output_dir / "report_vi.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run intrinsic dataset analysis for VialectBench finalized data."
    )
    parser.add_argument("--input", default="data/vialectbench_finalized_5.json")
    parser.add_argument("--output_dir", default="outputs/finalized_5_intrinsic_analysis")
    parser.add_argument("--model", "--reference_model", dest="model", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--orig_col", default="original_text")
    parser.add_argument("--para_col", default="dialect_text")
    parser.add_argument("--dialect_col", default="target_dialect")
    parser.add_argument("--sample_id_col", default="sample_id")
    parser.add_argument("--hypothesis_col", default="hypothesis")
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--n_boot", type=int, default=5000)
    parser.add_argument("--skip_perplexity", action="store_true")
    parser.add_argument("--reuse_scores", action="store_true")
    parser.add_argument(
        "--complete_six_dialects_only",
        action="store_true",
        help="Analyze only sample_id groups that contain all six dialect groups.",
    )
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        help="Allow Hugging Face remote code execution for models that require it. Off by default.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = read_input(input_path)
    required_cols = [args.orig_col, args.para_col, args.dialect_col, args.sample_id_col, "task"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing column: {col}")

    df = df.copy()
    df["baseline_text"] = df.apply(
        lambda row: resolve_baseline_text(row, args.orig_col, args.hypothesis_col),
        axis=1,
    )
    df["dialect_text_eval"] = df[args.para_col].astype(str).str.strip()
    df["orig_len"] = df["baseline_text"].map(word_len)
    df["para_len"] = df["dialect_text_eval"].map(word_len)
    df["length_ratio"] = np.where(
        df["orig_len"] > 0,
        df["para_len"] / df["orig_len"],
        np.nan,
    )

    duplicate_pairs = df.duplicated(subset=[args.sample_id_col, args.dialect_col], keep=False)
    coverage = (
        df.groupby(args.sample_id_col)[args.dialect_col]
        .agg(lambda values: ",".join(sorted(set(values.dropna()))))
        .reset_index(name="dialects_present")
    )
    coverage["n_dialects"] = coverage["dialects_present"].map(lambda value: len(value.split(",")) if value else 0)
    coverage["has_all_six_dialects"] = coverage["n_dialects"] == len(DEFAULT_DIALECTS)
    coverage.to_csv(output_dir / "sample_dialect_coverage.csv", index=False)

    n_original_rows = len(df)
    n_original_unique_samples = df[args.sample_id_col].nunique()
    complete_sample_ids = set(
        coverage.loc[coverage["has_all_six_dialects"], args.sample_id_col].astype(str)
    )
    if args.complete_six_dialects_only:
        df = df[df[args.sample_id_col].astype(str).isin(complete_sample_ids)].copy()

    dialect_counts = (
        df[args.dialect_col]
        .value_counts()
        .rename_axis(args.dialect_col)
        .reset_index(name="n")
        .sort_values(args.dialect_col)
    )
    task_counts = (
        df["task"]
        .value_counts()
        .rename_axis("task")
        .reset_index(name="n")
        .sort_values("task")
    )
    task_x_dialect = (
        df.pivot_table(
            index="task",
            columns=args.dialect_col,
            values=args.sample_id_col,
            aggfunc="count",
            fill_value=0,
        )
        .reset_index()
    )

    dialect_counts.to_csv(output_dir / "distribution_by_dialect.csv", index=False)
    task_counts.to_csv(output_dir / "distribution_by_task.csv", index=False)
    task_x_dialect.to_csv(output_dir / "distribution_task_x_dialect.csv", index=False)

    length_metrics = ["orig_len", "para_len", "length_ratio"]
    length_overall = summarize_numeric(df, None, length_metrics, n_boot=args.n_boot)
    length_by_dialect = summarize_numeric(df, args.dialect_col, length_metrics, n_boot=args.n_boot)
    length_by_task = summarize_numeric(df, "task", length_metrics, n_boot=args.n_boot)
    length_task_x_dialect = summarize_numeric(df, ["task", args.dialect_col], length_metrics, n_boot=args.n_boot)

    length_overall.to_csv(output_dir / "length_summary_overall.csv", index=False)
    length_by_dialect.to_csv(output_dir / "length_summary_by_dialect.csv", index=False)
    length_by_task.to_csv(output_dir / "length_summary_by_task.csv", index=False)
    length_task_x_dialect.to_csv(output_dir / "length_summary_task_x_dialect.csv", index=False)

    write_count_svg(dialect_counts, output_dir / "fig_count_by_dialect.svg", args.dialect_col, "Dataset Distribution by Dialect")
    write_count_svg(task_counts, output_dir / "fig_count_by_task.svg", "task", "Dataset Distribution by Task")

    metadata: dict[str, Any] = {
        "input": str(input_path),
        "n_rows": int(len(df)),
        "n_rows_before_filtering": int(n_original_rows),
        "n_unique_samples": int(df[args.sample_id_col].nunique()),
        "n_unique_samples_before_filtering": int(n_original_unique_samples),
        "n_complete_six_dialect_samples": int(coverage["has_all_six_dialects"].sum()),
        "complete_six_dialects_only": bool(args.complete_six_dialects_only),
        "n_duplicate_sample_dialect_pairs": int(duplicate_pairs.sum()),
        "dialects": sorted(df[args.dialect_col].dropna().astype(str).unique().tolist()),
        "tasks": sorted(df["task"].dropna().astype(str).unique().tolist()),
        "baseline_rule": f"NLI uses '{args.hypothesis_col}' as baseline; other tasks use '{args.orig_col}'.",
        "para_col": args.para_col,
        "reference_model": args.model if not args.skip_perplexity else None,
        "trust_remote_code": bool(args.trust_remote_code),
        "max_length": args.max_length,
        "n_boot": args.n_boot,
        "skip_perplexity": bool(args.skip_perplexity),
    }

    overall_summary = None
    dialect_summary = None
    task_summary = None

    pairwise_path = output_dir / "perplexity_pairwise_results.csv"
    if not args.skip_perplexity:
        if args.reuse_scores and pairwise_path.exists():
            result_df = pd.read_csv(pairwise_path)
        else:
            unique_texts = sorted(
                set(df["baseline_text"].astype(str)).union(
                    set(df["dialect_text_eval"].astype(str))
                )
            )
            metadata["n_unique_texts_scored"] = len(unique_texts)
            score_cache = score_unique_texts(
                unique_texts,
                model_name=args.model,
                max_length=args.max_length,
                trust_remote_code=args.trust_remote_code,
            )
            result_df = add_pairwise_scores(df, score_cache)
            result_df.to_csv(pairwise_path, index=False)

        ppl_metrics = [
            "orig_ppl",
            "para_ppl",
            "delta_ppl",
            "ratio_ppl",
            "orig_nll",
            "para_nll",
            "delta_nll",
            "orig_len",
            "para_len",
            "length_ratio",
        ]
        overall_summary = summarize_numeric(result_df, None, ppl_metrics, n_boot=args.n_boot)
        dialect_summary = summarize_numeric(result_df, args.dialect_col, ppl_metrics, n_boot=args.n_boot)
        task_summary = summarize_numeric(result_df, "task", ppl_metrics, n_boot=args.n_boot)
        task_x_dialect_summary = summarize_numeric(result_df, ["task", args.dialect_col], ppl_metrics, n_boot=args.n_boot)

        overall_summary.to_csv(output_dir / "perplexity_summary_overall.csv", index=False)
        dialect_summary.to_csv(output_dir / "perplexity_summary_by_dialect.csv", index=False)
        task_summary.to_csv(output_dir / "perplexity_summary_by_task.csv", index=False)
        task_x_dialect_summary.to_csv(output_dir / "perplexity_summary_task_x_dialect.csv", index=False)

        write_grouped_bar_svg(
            dialect_summary,
            output_dir / "fig_ppl_by_dialect.svg",
            group_col=args.dialect_col,
            value_cols=[
                ("orig_ppl_mean", "Original PPL", "#4C78A8"),
                ("para_ppl_mean", "Dialect PPL", "#F58518"),
            ],
            title="Original vs Dialect Perplexity by Dialect",
            subtitle="Reference LM is used only as a fixed data scorer",
            y_label="Mean PPL",
        )
        write_delta_nll_svg(
            dialect_summary,
            output_dir / "fig_delta_nll_by_dialect.svg",
            group_col=args.dialect_col,
        )

    (output_dir / "dataset_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_report(
        output_dir,
        metadata,
        dialect_counts,
        task_counts,
        overall_summary,
        dialect_summary,
        task_summary,
    )

    print("Saved intrinsic analysis to:", output_dir)
    for path in sorted(output_dir.iterdir()):
        print(path)


if __name__ == "__main__":
    main()
