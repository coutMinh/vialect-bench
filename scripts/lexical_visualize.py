#!/usr/bin/env python3
"""
lexical_swap_nll_visualize.py
=============================
Generates figures for the term-level NLL attribution experiment
(outputs of lexical_swap_nll.py).

Figures saved to:
  outputs/lexical_substitution_analysis/figures/

Run from vialect_github/vialect-bench/
"""

from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE      = Path(".")
DATA_DIR  = BASE / "outputs/lexical_substitution_analysis/swap_nll_analysis"
FIG_DIR   = BASE / "outputs/lexical_substitution_analysis/figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "figure.dpi":        150,
    "font.family":       "DejaVu Sans",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.3,
    "axes.titlesize":    13,
    "axes.labelsize":    11,
    "xtick.labelsize":   9,
    "ytick.labelsize":   9,
})

CAT_PALETTE = {
    "Interrogatives":             "#e63946",
    "Connectives/Aspect markers": "#457b9d",
    "Pronouns":                   "#2a9d8f",
    "Demonstratives/Deixis":      "#e9c46a",
    "Negation/Function words":    "#f4a261",
    "Predicate vocabulary":       "#264653",
    "Discourse particles":        "#a8dadc",
    "Kinship/Honorific terms":    "#6d6875",
    "Idiomatic expressions":      "#b5838d",
}

DIAL_PALETTE = {
    "PNB":  "#90e0ef",
    "PNN":  "#48cae4",
    "PNT1": "#f4a261",
    "PNT2": "#e76f51",
    "PNT3": "#c1121f",
    "PNT4": "#d4a373",
}

def save(fig, name):
    p = FIG_DIR / name
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {p.name}")


# ---------------------------------------------------------------------------
# Fig A — Category ranking bar chart (mean ± SE)
# ---------------------------------------------------------------------------
def fig_category_ranking():
    df = pd.read_csv(DATA_DIR / "swap_nll_by_category.csv")
    df = df.sort_values("delta_nll_swap_mean", ascending=True)
    se = df["delta_nll_swap_std"] / np.sqrt(df["n_occurrences"])
    colors = [CAT_PALETTE.get(c, "#999") for c in df["category"]]

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.barh(df["category"], df["delta_nll_swap_mean"],
                   xerr=se, color=colors, error_kw={"elinewidth": 1.2, "capsize": 3},
                   height=0.6)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Mean Δ NLL per term swap (↑ = more confusing)")
    ax.set_title("Term-level NLL attribution by substitution category")
    for bar, v in zip(bars, df["delta_nll_swap_mean"]):
        ax.text(v + 0.005, bar.get_y() + bar.get_height() / 2,
                f"{v:.3f}", va="center", fontsize=8)
    fig.tight_layout()
    save(fig, "figA_category_term_nll_ranking.png")


# ---------------------------------------------------------------------------
# Fig B — Dialect ranking bar chart
# ---------------------------------------------------------------------------
def fig_dialect_ranking():
    df = pd.read_csv(DATA_DIR / "swap_nll_by_dialect.csv")
    df = df.sort_values("delta_nll_swap_mean", ascending=True)
    se = df["delta_nll_swap_std"] / np.sqrt(df["n_occurrences"])
    colors = [DIAL_PALETTE.get(d, "#999") for d in df["target_dialect"]]

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.barh(df["target_dialect"], df["delta_nll_swap_mean"],
                   xerr=se, color=colors, error_kw={"elinewidth": 1.2, "capsize": 3},
                   height=0.5)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Mean Δ NLL per term swap")
    ax.set_title("Term-level NLL attribution by dialect")
    for bar, v in zip(bars, df["delta_nll_swap_mean"]):
        offset = 0.005 if v >= 0 else -0.03
        ax.text(v + offset, bar.get_y() + bar.get_height() / 2,
                f"{v:.3f}", va="center", fontsize=8)
    fig.tight_layout()
    save(fig, "figB_dialect_term_nll_ranking.png")


# ---------------------------------------------------------------------------
# Helper — generic heatmap renderer
# ---------------------------------------------------------------------------
def _draw_heatmap(pivot: pd.DataFrame, title: str, xlabel: str, ylabel: str,
                  fig_name: str, figsize=(8, 5)):
    vmin = np.nanmin(pivot.values)
    vmax = np.nanmax(pivot.values)
    # Use diverging palette if values span negative
    cmap = "RdYlGn_r" if vmin < 0 else "YlOrRd"

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(pivot.values, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=30, ha="right")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    plt.colorbar(im, ax=ax, label="Mean Δ NLL (term-level)")
    mid = (vmin + vmax) / 2
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            v = pivot.values[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.3f}", ha="center", va="center",
                        fontsize=7.5,
                        color="white" if v > mid else "black")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.tight_layout()
    save(fig, fig_name)


# ---------------------------------------------------------------------------
# Fig C — Category × dialect heatmap
# ---------------------------------------------------------------------------
def fig_category_dialect_heatmap():
    occ = pd.read_csv(DATA_DIR / "swap_nll_per_occurrence.csv")
    pivot = (
        occ.groupby(["category", "target_dialect"])["delta_nll_swap"]
        .mean()
        .unstack("target_dialect")
        .reindex(columns=["PNB", "PNN", "PNT1", "PNT2", "PNT3", "PNT4"])
    )
    pivot = pivot.loc[pivot.mean(axis=1).sort_values(ascending=False).index]
    _draw_heatmap(pivot,
                  title="Term-level Δ NLL: category × dialect",
                  xlabel="Dialect", ylabel="",
                  fig_name="figC_category_x_dialect_term_nll_heatmap.png",
                  figsize=(8, 5))


# ---------------------------------------------------------------------------
# Fig G — Category × task heatmap
# ---------------------------------------------------------------------------
def fig_category_task_heatmap():
    occ = pd.read_csv(DATA_DIR / "swap_nll_per_occurrence.csv")
    task_order = ["MCQA", "NLI", "QA", "SENT"]
    pivot = (
        occ.groupby(["category", "task"])["delta_nll_swap"]
        .mean()
        .unstack("task")
    )
    # Keep only known tasks, sort rows by overall mean
    cols = [t for t in task_order if t in pivot.columns]
    pivot = pivot[cols]
    pivot = pivot.loc[pivot.mean(axis=1).sort_values(ascending=False).index]
    _draw_heatmap(pivot,
                  title="Term-level Δ NLL: category × task",
                  xlabel="Task", ylabel="",
                  fig_name="figG_category_x_task_term_nll_heatmap.png",
                  figsize=(7, 5))


# ---------------------------------------------------------------------------
# Fig H — Task × dialect heatmap
# ---------------------------------------------------------------------------
def fig_task_dialect_heatmap():
    occ = pd.read_csv(DATA_DIR / "swap_nll_per_occurrence.csv")
    task_order = ["MCQA", "NLI", "QA", "SENT"]
    pivot = (
        occ.groupby(["task", "target_dialect"])["delta_nll_swap"]
        .mean()
        .unstack("target_dialect")
        .reindex(columns=["PNB", "PNN", "PNT1", "PNT2", "PNT3", "PNT4"])
    )
    rows = [t for t in task_order if t in pivot.index]
    pivot = pivot.loc[rows]
    _draw_heatmap(pivot,
                  title="Term-level Δ NLL: task × dialect",
                  xlabel="Dialect", ylabel="Task",
                  fig_name="figH_task_x_dialect_term_nll_heatmap.png",
                  figsize=(7, 4))


# ---------------------------------------------------------------------------
# Fig D — Top substitution pairs bar chart (n >= 5)
# ---------------------------------------------------------------------------
def fig_top_pairs(min_n: int = 5, top_k: int = 25):
    df = pd.read_csv(DATA_DIR / "swap_nll_by_pair.csv")
    df = df[df["n_occurrences"] >= min_n].sort_values(
        "delta_nll_swap_mean", ascending=False
    ).head(top_k)
    labels = [f"{r.standard_span}→{r.dialect_span}" for _, r in df.iterrows()]
    colors = [CAT_PALETTE.get(c, "#999") for c in df["category"]]
    se = df["delta_nll_swap_std"] / np.sqrt(df["n_occurrences"])

    fig, ax = plt.subplots(figsize=(8, max(5, top_k * 0.32)))
    bars = ax.barh(labels[::-1], df["delta_nll_swap_mean"].values[::-1],
                   xerr=se.values[::-1], color=colors[::-1],
                   error_kw={"elinewidth": 1, "capsize": 2}, height=0.7)
    ax.set_xlabel("Mean Δ NLL per term swap")
    ax.set_title(f"Top {top_k} substitution pairs by term-level Δ NLL (n ≥ {min_n})")

    # Legend for categories present
    from matplotlib.patches import Patch
    cats_present = df["category"].unique()
    legend_items = [Patch(facecolor=CAT_PALETTE.get(c, "#999"), label=c)
                    for c in cats_present]
    ax.legend(handles=legend_items, fontsize=7, loc="lower right")
    fig.tight_layout()
    save(fig, f"figD_top{top_k}_pairs_term_nll_min{min_n}.png")


# ---------------------------------------------------------------------------
# Fig E — Violin / box plot: per-category distribution of swap Δ NLL
# ---------------------------------------------------------------------------
def fig_category_distribution():
    occ = pd.read_csv(DATA_DIR / "swap_nll_per_occurrence.csv")
    cat_order = (
        occ.groupby("category")["delta_nll_swap"].mean()
        .sort_values(ascending=False).index.tolist()
    )
    data_by_cat = [occ[occ["category"] == c]["delta_nll_swap"].values
                   for c in cat_order]
    colors = [CAT_PALETTE.get(c, "#999") for c in cat_order]

    fig, ax = plt.subplots(figsize=(10, 5))
    parts = ax.violinplot(data_by_cat, positions=range(len(cat_order)),
                          showmedians=True, showextrema=False)
    for pc, col in zip(parts["bodies"], colors):
        pc.set_facecolor(col)
        pc.set_alpha(0.75)
    parts["cmedians"].set_color("black")
    ax.set_xticks(range(len(cat_order)))
    ax.set_xticklabels(cat_order, rotation=30, ha="right")
    ax.set_ylabel("Δ NLL per term swap")
    ax.set_title("Distribution of term-level Δ NLL by substitution category")
    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    fig.tight_layout()
    save(fig, "figE_category_term_nll_distribution.png")


# ---------------------------------------------------------------------------
# Fig F — Scatter: n_occurrences vs mean Δ NLL per pair (bubble chart)
# ---------------------------------------------------------------------------
def fig_pair_scatter(min_n: int = 3):
    df = pd.read_csv(DATA_DIR / "swap_nll_by_pair.csv")
    df = df[df["n_occurrences"] >= min_n]
    colors = [CAT_PALETTE.get(c, "#999") for c in df["category"]]

    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(
        df["n_occurrences"], df["delta_nll_swap_mean"],
        c=colors, s=60, alpha=0.7, edgecolors="none"
    )
    # Label the top-10 points by Δ NLL
    top10 = df.nlargest(10, "delta_nll_swap_mean")
    for _, r in top10.iterrows():
        ax.annotate(
            f"{r.standard_span}→{r.dialect_span}",
            (r.n_occurrences, r.delta_nll_swap_mean),
            fontsize=7, xytext=(5, 2), textcoords="offset points"
        )
    ax.set_xlabel("Number of occurrences in dataset")
    ax.set_ylabel("Mean Δ NLL per term swap")
    ax.set_title("Substitution pairs: frequency vs term-level NLL cost")
    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")

    from matplotlib.patches import Patch
    legend_items = [Patch(facecolor=CAT_PALETTE.get(c, "#999"), label=c)
                    for c in CAT_PALETTE if c in df["category"].values]
    ax.legend(handles=legend_items, fontsize=7, loc="upper right")
    fig.tight_layout()
    save(fig, "figF_pair_frequency_vs_term_nll.png")


# ---------------------------------------------------------------------------
# Fig F2 — All 351 pairs labeled, two-panel (rare n=1-4 | frequent n>=5)
# Left panel: x-jittered within each count to spread 206 n=1 pairs
# Right panel: log x-scale
# ---------------------------------------------------------------------------
def fig_pair_scatter_all():
    from adjustText import adjust_text

    df = pd.read_csv(DATA_DIR / "swap_nll_by_pair.csv")
    rare = df[df.n_occurrences <= 4].copy()
    freq = df[df.n_occurrences >= 5].copy()

    # Jitter x within each n-group for the rare panel
    rng = np.random.default_rng(42)
    def jitter_x(n_val, count, spread=0.28):
        base  = np.linspace(-spread, spread, count)
        noise = rng.uniform(-0.04, 0.04, count)
        return n_val + base + noise

    rare = rare.sort_values(["n_occurrences", "delta_nll_swap_mean"]).copy()
    rare["x_jitter"] = rare["n_occurrences"].astype(float)
    for n_val, grp in rare.groupby("n_occurrences"):
        rare.loc[grp.index, "x_jitter"] = jitter_x(n_val, len(grp))
    freq = freq.copy()
    freq["x_jitter"] = freq["n_occurrences"].astype(float)

    def draw_panel(ax, sub, title, fs_base=5.5, size_scale=6.0,
                   xscale="linear", xtick_vals=None, xlim=None):
        colors = [CAT_PALETTE.get(c, "#999") for c in sub["category"]]
        sizes  = 30 + sub["n_occurrences"] * size_scale
        ax.scatter(sub["x_jitter"], sub["delta_nll_swap_mean"],
                   c=colors, s=sizes, alpha=0.75,
                   edgecolors="white", linewidths=0.5, zorder=3)

        ymin = sub["delta_nll_swap_mean"].min()
        ymax = sub["delta_nll_swap_mean"].max()
        ypad = (ymax - ymin) * 0.10
        ax.set_ylim(ymin - ypad - 0.1, ymax + ypad + 0.35)
        if xlim:
            ax.set_xlim(*xlim)

        texts = []
        for _, r in sub.iterrows():
            t = ax.text(r.x_jitter, r.delta_nll_swap_mean,
                        f"{r.standard_span}→{r.dialect_span}",
                        fontsize=fs_base, color="#1a1a1a",
                        va="center", zorder=4, clip_on=True)
            texts.append(t)

        adjust_text(texts, ax=ax,
            x=sub["x_jitter"].values,
            y=sub["delta_nll_swap_mean"].values,
            arrowprops=dict(arrowstyle="-", color="#cccccc", lw=0.35),
            expand=(1.1, 1.35),
            force_text=(0.45, 0.65),
            force_points=(0.15, 0.25),
            lim=500,
            only_move={"points": "y", "text": "xy"},
        )
        # re-enforce limits after adjustText
        ax.set_ylim(ymin - ypad - 0.1, ymax + ypad + 0.35)
        if xlim:
            ax.set_xlim(*xlim)

        if xscale == "log":
            ax.set_xscale("log")
            ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
            ax.set_xticks(xtick_vals or [5, 10, 20, 50, 100, 200])
        else:
            ax.set_xticks(xtick_vals or sorted(sub["n_occurrences"].unique()))
            ax.tick_params(axis="x", labelsize=8.5)

        ax.axhline(0, color="gray", lw=0.8, linestyle="--", alpha=0.35, zorder=1)
        ax.set_title(title, fontsize=11, pad=6)
        ax.set_ylabel("Mean Δ NLL per term swap", fontsize=10)
        ax.set_xlabel("Number of occurrences", fontsize=10)

    fig, ax2 = plt.subplots(1, 1, figsize=(14, 10))

    draw_panel(ax2, freq,
               f"Frequent substitution pairs  (n ≥ 5,  {len(freq)} pairs)",
               fs_base=6.5, size_scale=5.5, xscale="log",
               xtick_vals=[5, 10, 20, 50, 100, 200])

    from matplotlib.patches import Patch
    cat_items  = [Patch(facecolor=CAT_PALETTE.get(c, "#999"), label=c)
                  for c in CAT_PALETTE if c in df["category"].values]
    size_items = [plt.scatter([], [], s=30 + n * 5.5, c="#aaaaaa", alpha=0.7,
                              edgecolors="white", label=f"n={n}")
                  for n in [1, 5, 20, 50, 100, 200]]
    ax2.legend(handles=cat_items + size_items, fontsize=8, loc="upper right",
               title="Category / n", title_fontsize=9, framealpha=0.92)

    ax2.set_title(
        "Substitution pairs (n ≥ 5): frequency vs term-level NLL cost\n"
        "(bubble size = n occurrences, color = category)",
        fontsize=13, pad=10
    )
    fig.subplots_adjust(left=0.07, right=0.97, top=0.92, bottom=0.08)
    fig.savefig(FIG_DIR / "figF2_all_pairs_frequency_vs_term_nll.png", dpi=120)
    plt.close(fig)
    print("  Saved figF2_all_pairs_frequency_vs_term_nll.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("Generating term-level NLL attribution figures ...")
    fig_category_ranking()
    fig_dialect_ranking()
    fig_category_dialect_heatmap()
    fig_category_task_heatmap()
    fig_task_dialect_heatmap()
    fig_top_pairs(min_n=5, top_k=25)
    fig_category_distribution()
    fig_pair_scatter(min_n=3)
    fig_pair_scatter_all()
    print(f"\nAll figures saved to: {FIG_DIR}")
    for f in sorted(FIG_DIR.iterdir()):
        if f.suffix == ".png":
            print(f"  {f.name}")


if __name__ == "__main__":
    main()
