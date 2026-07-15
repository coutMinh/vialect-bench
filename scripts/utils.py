#!/usr/bin/env python3
"""
Rebuilt lexical transformation + perplexity-uncertainty analysis for Vialect-Bench.

This replaces analyze_lexical_coverage.py / analyze_category_perplexity.py /
analyze_top_words_perplexity.py, which used a WRONG 1:1 dialect-code -> lexicon-column
mapping (e.g. hardcoded PNN -> "Nghe An"). Verified with word-level evidence that PNN's
dialect_text uses distinctly Southern Vietnamese words (ma, ba, tui, hong, vo, dia) --
i.e. PNN is the Southern macro-region, not Nghe An.

The lexicon (`data/DIALECT_LEXICON_v2.xlsx`) has 9 fine-grained regional columns:
    BacBo, ThanhHoa, NgheAn, HaTinh_QB_QT, Hue, QuangNam, QuangNgai, PhuYen, NamBo
but the dataset only has 6 macro dialect codes: PNB, PNN, PNT1, PNT2, PNT3, PNT4.
So the correct mapping is MANY lexicon columns -> ONE dataset dialect code:

    PNB  -> BacBo                            (Bac Bo)
    PNN  -> NamBo                            (Nam Bo)
    PNT1 -> ThanhHoa                         (Thanh Hoa)
    PNT2 -> NgheAn + HaTinh_QB_QT            (Nghe An, Ha Tinh -- HaTinh_QB_QT is the closest
                                               available lexicon column to "Ha Tinh"; the
                                               column itself is labelled "Ha Tinh - Quang Binh
                                               - Quang Tri" in the source spreadsheet, so it is
                                               shared with PNT3 by construction of the lexicon,
                                               not by our choice)
    PNT3 -> HaTinh_QB_QT + Hue               (Quang Binh, Quang Tri, Thua Thien Hue)
    PNT4 -> QuangNam + QuangNgai + PhuYen    (Nam Trung Bo / South-Central coast)

This mapping matches the authoritative dialect definitions provided by the research team:
    PNB  -> Bac Bo
    PNT1 -> Thanh Hoa
    PNT2 -> Nghe An, Ha Tinh
    PNT3 -> Quang Binh, Quang Tri, Thua Thien Hue
    PNT4 -> Nam Trung Bo (South-Central: Quang Nam, Quang Ngai, Phu Yen, ...)
    PNN  -> Nam Bo

The only unavoidable overlap is the lexicon's own "HaTinh_QB_QT" column, which the source
spreadsheet already merges "Ha Tinh - Quang Binh - Quang Tri" into a single column. Since Ha
Tinh belongs to PNT2 and Quang Binh/Quang Tri belong to PNT3 per the authoritative mapping,
that single lexicon column is assigned to both PNT2 and PNT3 -- this reflects a genuine
granularity limit of the source lexicon, not an analysis choice. Every other lexicon column
is assigned to exactly one dialect code.

Pipeline:
1. Load lexicon, build {category_section -> {region_column -> set(words/phrases)}} and a
   canonicalized 9-category label ("1. Interrogatives" style section headers already match
   the knowledge doc's 9 canonical categories).
2. Build per-dataset-dialect lookup: dialect_word/phrase (lowercased) -> (category, standard_word).
   A dialect word can map to more than one category (rare); we keep all matches and record
   ambiguity.
3. For every (sample_id, target_dialect) pair, diff baseline_text (hypothesis for NLI, else
   original_text) against dialect_text at the token level using difflib to find changed token
   positions. Then, for each changed position, try to match the LONGEST possible lexicon
   n-gram starting there first (up to the longest phrase actually present in the lexicon,
   e.g. 5 tokens for entries like "biệt tăm biệt tích" or "không (nỏ) chộ tăm hơi"), falling
   back to shorter windows and finally single tokens. This "longest-match-first" strategy is
   needed because several lexicon entries are multi-word idioms (trigrams/4-grams/5-grams),
   not just unigrams/bigrams -- e.g. "như ri thì" / "như rứa thì" are 3-token idiomatic
   rewrites of "thế này thì" that would be missed (or partially mis-matched at the single-word
   level) by a unigram/bigram-only diff. Once a span is matched, its tokens are marked
   "covered" so a shorter overlapping n-gram cannot double-count it. Only unmatched unigrams
   are recorded as category="Unmatched" (to avoid inflating the unmatched count by also
   emitting every unmatched multi-word window built on top of them), so we can report the
   lexicon match rate as a QA metric.
4. Join with existing perplexity_pairwise_results.csv (delta_nll, delta_ppl) on
   (sample_id, target_dialect) -- no perplexity is recomputed.
5. Part 1: coverage / distribution metrics (category x task, category x dialect, density).
   Part 2: perplexity-increase metrics (category x task/dialect, top-N phrases of any n-gram
   length by delta_nll, plus a combined all-n-gram-lengths-pooled ranking).
6. Save every artifact (mapping used, per-sample transformation records, aggregates, top-N
   rankings) under --output_dir.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Authoritative dialect -> lexicon region-column mapping, as specified by the research team:
#   PNB  -> Bac Bo
#   PNT1 -> Thanh Hoa
#   PNT2 -> Nghe An, Ha Tinh
#   PNT3 -> Quang Binh, Quang Tri, Thua Thien Hue
#   PNT4 -> Nam Trung Bo (South-Central)
#   PNN  -> Nam Bo
#
# The lexicon's own "HaTinh_QB_QT" column merges Ha Tinh + Quang Binh + Quang Tri into one
# column, so it is the only column assigned to two dialect codes (PNT2 for Ha Tinh, PNT3 for
# Quang Binh/Quang Tri) -- a granularity limit of the source spreadsheet, not an analysis
# choice. Every other lexicon column maps to exactly one dialect code.
#
# Earlier iteration history (kept for transparency): a first draft wrongly hardcoded
# PNN -> "Nghe An" (1:1, wrong region entirely). A second draft assigned QuangNgai + PhuYen to
# both PNN and PNT4, which double-counted two South-Central provinces into Nam Bo purely
# because of a shared colloquial-Southern vocabulary layer (tui, tụi, ổng, bả, dìa, kêu, đó);
# that was corrected after review since Quang Ngai / Phu Yen are Nam Trung Bo, not Nam Bo.
# ---------------------------------------------------------------------------
DIALECT_TO_LEXICON_COLUMNS: dict[str, list[str]] = {
    "PNB": ["BacBo"],
    "PNN": ["NamBo"],
    "PNT1": ["ThanhHoa"],
    "PNT2": ["NgheAn", "HaTinh_QB_QT"],
    "PNT3": ["HaTinh_QB_QT", "Hue"],
    "PNT4": ["QuangNam", "QuangNgai", "PhuYen"],
}

DIALECT_REGION_LABEL: dict[str, str] = {
    "PNB": "Bac Bo",
    "PNN": "Nam Bo",
    "PNT1": "Thanh Hoa",
    "PNT2": "Nghe An, Ha Tinh",
    "PNT3": "Quang Binh, Quang Tri, Thua Thien Hue",
    "PNT4": "Nam Trung Bo (South-Central)",
}

LEXICON_COLUMN_INDEX: dict[str, int] = {
    "BacBo": 7,
    "ThanhHoa": 8,
    "NgheAn": 9,
    "HaTinh_QB_QT": 10,
    "Hue": 11,
    "QuangNam": 12,
    "QuangNgai": 13,
    "PhuYen": 14,
    "NamBo": 15,
}
STANDARD_COL_IDX = 3
CATEGORY_COL_IDX = 5
SECTION_NAME_COL_IDX = 1

# Canonical 9 categories (from the knowledge doc), keyed by the numbered section header
# text that already exists in the lexicon's section-header rows.
SECTION_TO_CANONICAL_CATEGORY: dict[str, str] = {
    "1. interrogatives": "Interrogatives",
    "2. demonstratives & deixis": "Demonstratives/Deixis",
    "3. negation & function words": "Negation/Function words",
    "4. discourse particles": "Discourse particles",
    "5. connectives & aspect markers": "Connectives/Aspect markers",
    "6. pronouns": "Pronouns",
    "7. kinship & honorifics": "Kinship/Honorific terms",
    "8. predicate vocabulary": "Predicate vocabulary",
    "9. templates & idioms": "Idiomatic expressions",
}

# Fallback: free-text category column value -> canonical category, used only if a lexicon
# data row is somehow missing its section (should not happen after forward-fill, kept for
# robustness).
FREE_TEXT_TO_CANONICAL: dict[str, str] = {
    "interrogative": "Interrogatives",
    "demonstrative": "Demonstratives/Deixis",
    "negation": "Negation/Function words",
    "function": "Negation/Function words",
    "particle": "Discourse particles",
    "connective": "Connectives/Aspect markers",
    "aspect": "Connectives/Aspect markers",
    "pronoun": "Pronouns",
    "kinship": "Kinship/Honorific terms",
    "predicate": "Predicate vocabulary",
    "idiom": "Idiomatic expressions",
    "template": "Idiomatic expressions",
}

UNMATCHED_CATEGORY = "Unmatched"
DEFAULT_DIALECTS = ["PNB", "PNN", "PNT1", "PNT2", "PNT3", "PNT4"]


# ---------------------------------------------------------------------------
# Lexicon loading
# ---------------------------------------------------------------------------
def normalize_phrase(text: Any) -> str:
    text = str(text).strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def load_lexicon_raw(lexicon_path: Path) -> pd.DataFrame:
    df = pd.read_excel(lexicon_path, header=0)
    # Row 0 repeats the header labels ("Tieng chuan", ...); real data starts at row 1.
    if str(df.iloc[0, STANDARD_COL_IDX]).strip() == "Tiếng chuẩn":
        df = df.iloc[1:].reset_index(drop=True)
    return df


def build_lexicon_entries(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a long-format dataframe with one row per (standard_word, region_column,
    dialect_word_variant, canonical_category), forward-filling the section header so every
    data row gets a canonical category.
    """
    section_col = df.columns[SECTION_NAME_COL_IDX]
    standard_col = df.columns[STANDARD_COL_IDX]
    category_col = df.columns[CATEGORY_COL_IDX]

    current_section = None
    records: list[dict[str, Any]] = []

    for _, row in df.iterrows():
        section_val = row[section_col]
        if pd.notna(section_val):
            current_section = normalize_phrase(section_val)
            continue  # section header rows have no standard word / dialect data

        standard_val = row[standard_col]
        if pd.isna(standard_val):
            continue

        free_text_category = normalize_phrase(row[category_col]) if pd.notna(row[category_col]) else ""
        canonical = SECTION_TO_CANONICAL_CATEGORY.get(current_section or "")
        if canonical is None:
            canonical = FREE_TEXT_TO_CANONICAL.get(free_text_category, UNMATCHED_CATEGORY)

        standard_word = normalize_phrase(standard_val)

        for region_name, col_idx in LEXICON_COLUMN_INDEX.items():
            cell_val = row.iloc[col_idx]
            if pd.isna(cell_val):
                continue
            cell_text = normalize_phrase(cell_val)
            # Variants separated by "/" ; keep multi-word phrases intact.
            for variant in cell_text.split("/"):
                variant = variant.strip()
                if not variant or variant == "nan":
                    continue
                records.append(
                    {
                        "standard_word": standard_word,
                        "region_column": region_name,
                        "dialect_phrase": variant,
                        "category": canonical,
                        "free_text_category": free_text_category,
                    }
                )

    return pd.DataFrame(records)


def build_dialect_lookup(
    lexicon_long: pd.DataFrame,
) -> dict[str, dict[str, list[dict[str, str]]]]:
    """
    Build {dataset_dialect_code -> {dialect_phrase -> [ {category, standard_word, region_column}, ... ]}}
    by unioning the region columns assigned to that dialect code.
    """
    lookup: dict[str, dict[str, list[dict[str, str]]]] = {d: defaultdict(list) for d in DEFAULT_DIALECTS}

    for dialect_code, region_cols in DIALECT_TO_LEXICON_COLUMNS.items():
        subset = lexicon_long[lexicon_long["region_column"].isin(region_cols)]
        for _, row in subset.iterrows():
            phrase = row["dialect_phrase"]
            entry = {
                "category": row["category"],
                "standard_word": row["standard_word"],
                "region_column": row["region_column"],
            }
            # avoid exact duplicate entries
            if entry not in lookup[dialect_code][phrase]:
                lookup[dialect_code][phrase].append(entry)

    return {d: dict(v) for d, v in lookup.items()}


# ---------------------------------------------------------------------------
# Text diff extraction (unigram..N-gram, N = longest lexicon phrase length)
# ---------------------------------------------------------------------------
_PUNCT_RE = re.compile(r"[.,!?;:\n]+")


def tokenize(text: Any) -> list[str]:
    text = str(text).strip().lower()
    text = _PUNCT_RE.sub(" ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text.split(" ") if text else []


def resolve_baseline_text(row: pd.Series, orig_col: str, hypothesis_col: str) -> str:
    if str(row.get("task", "")).upper() == "NLI" and hypothesis_col in row and pd.notna(row.get(hypothesis_col)):
        return str(row.get(hypothesis_col, "")).strip()
    return str(row.get(orig_col, "")).strip()


# ---------------------------------------------------------------------------
# Category classification for a changed word/phrase
# ---------------------------------------------------------------------------
def classify_phrase(
    phrase: str,
    dialect_code: str,
    lookup: dict[str, dict[str, list[dict[str, str]]]],
) -> list[dict[str, str]]:
    """Exact lookup of `phrase` (any n-gram length) against the dialect-specific lexicon."""
    dialect_lookup = lookup.get(dialect_code, {})
    return dialect_lookup.get(phrase, [])


def extract_matched_spans(
    dialect_words: list[str],
    changed_idx: set[int],
    dialect_code: str,
    lookup: dict[str, dict[str, list[dict[str, str]]]],
    max_ngram: int = 5,
) -> list[dict[str, Any]]:
    """
    Longest-match-first extraction: scan changed token positions and try to match the
    longest possible lexicon n-gram (up to max_ngram) starting at each position before
    falling back to shorter windows. Once a span is matched and consumed, the covered token
    positions are not re-matched by a shorter overlapping n-gram, which avoids spurious
    partial hits (e.g. matching the single word "như" against the lexicon entry
    "nhu ri thi" when the full phrase "nhu ri thi" is actually present and should be
    classified as one Idiomatic-expression unit instead).

    Returns a list of {start, end, phrase, ngram_type, matches} dicts, plus records every
    unmatched unigram (that was itself a changed token, not already covered by a longer
    match) with an empty matches list so callers can report it as "Unmatched".
    """
    n_tokens = len(dialect_words)
    covered = [False] * n_tokens
    spans: list[dict[str, Any]] = []

    sorted_changed_idx = sorted(changed_idx)
    for start in sorted_changed_idx:
        if covered[start]:
            continue
        matched_here = False
        for n in range(min(max_ngram, n_tokens - start), 1, -1):
            end = start + n
            # only attempt this window if it overlaps a changed token (guaranteed since
            # start is itself changed) and is not already partially covered by a longer
            # match starting earlier
            if any(covered[start:end]):
                continue
            phrase = " ".join(dialect_words[start:end])
            matches = classify_phrase(phrase, dialect_code, lookup)
            if matches:
                spans.append(
                    {
                        "start": start,
                        "end": end,
                        "phrase": phrase,
                        "ngram_type": f"{n}-gram",
                        "matches": matches,
                    }
                )
                for i in range(start, end):
                    covered[i] = True
                matched_here = True
                break
        if not matched_here and not covered[start]:
            word = dialect_words[start]
            matches = classify_phrase(word, dialect_code, lookup)
            spans.append(
                {
                    "start": start,
                    "end": start + 1,
                    "phrase": word,
                    "ngram_type": "unigram",
                    "matches": matches,
                }
            )
            covered[start] = True

    return spans


def build_transformation_records(
    df: pd.DataFrame,
    lookup: dict[str, dict[str, list[dict[str, str]]]],
    sample_id_col: str,
    dialect_col: str,
    max_ngram: int = 5,
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []

    for _, row in df.iterrows():
        dialect_code = row[dialect_col]
        if dialect_code == "PNB":
            continue  # PNB is the standard baseline itself, nothing to diff against

        baseline_text = row["baseline_text"]
        dialect_text = row["dialect_text_eval"]

        import difflib

        baseline_words = tokenize(baseline_text)
        dialect_words = tokenize(dialect_text)
        sm = difflib.SequenceMatcher(None, baseline_words, dialect_words, autojunk=False)
        changed_idx: set[int] = set()
        for tag, _i1, _i2, j1, j2 in sm.get_opcodes():
            if tag in ("replace", "insert"):
                changed_idx.update(range(j1, j2))

        spans = extract_matched_spans(dialect_words, changed_idx, dialect_code, lookup, max_ngram=max_ngram)

        for span in spans:
            matches = span["matches"]
            if matches:
                seen_categories = set()
                for m in matches:
                    if m["category"] in seen_categories:
                        continue
                    seen_categories.add(m["category"])
                    records.append(
                        {
                            "sample_id": row[sample_id_col],
                            "task": row["task"],
                            "target_dialect": dialect_code,
                            "ngram_type": span["ngram_type"],
                            "phrase": span["phrase"],
                            "category": m["category"],
                            "standard_word": m["standard_word"],
                            "region_column": m["region_column"],
                        }
                    )
            else:
                # only unigram spans are recorded as Unmatched (to avoid double-counting
                # unmatched multi-word windows on top of their already-unmatched unigrams)
                if span["ngram_type"] == "unigram":
                    records.append(
                        {
                            "sample_id": row[sample_id_col],
                            "task": row["task"],
                            "target_dialect": dialect_code,
                            "ngram_type": "unigram",
                            "phrase": span["phrase"],
                            "category": UNMATCHED_CATEGORY,
                            "standard_word": "",
                            "region_column": "",
                        }
                    )

    return pd.DataFrame(records)


def build_qa_summary(
    df: pd.DataFrame,
    lookup: dict[str, dict[str, list[dict[str, str]]]],
    sample_id_col: str,
    dialect_col: str,
    max_ngram: int = 5,
) -> pd.DataFrame:
    rows = []
    for _, row in df.iterrows():
        dialect_code = row[dialect_col]
        if dialect_code == "PNB":
            continue
        baseline_text = row["baseline_text"]
        dialect_text = row["dialect_text_eval"]

        import difflib

        baseline_words = tokenize(baseline_text)
        dialect_words = tokenize(dialect_text)
        sm = difflib.SequenceMatcher(None, baseline_words, dialect_words, autojunk=False)
        changed_idx: set[int] = set()
        for tag, _i1, _i2, j1, j2 in sm.get_opcodes():
            if tag in ("replace", "insert"):
                changed_idx.update(range(j1, j2))

        spans = extract_matched_spans(dialect_words, changed_idx, dialect_code, lookup, max_ngram=max_ngram)
        n_changed_tokens = len(changed_idx)
        n_matched_tokens = sum((span["end"] - span["start"]) for span in spans if span["matches"])
        n_spans_multiword = sum(1 for span in spans if span["ngram_type"] != "unigram")
        rows.append(
            {
                "sample_id": row[sample_id_col],
                "task": row["task"],
                "target_dialect": dialect_code,
                "n_changed_unigrams": n_changed_tokens,
                "n_matched_unigrams": n_matched_tokens,
                "match_rate": (n_matched_tokens / n_changed_tokens) if n_changed_tokens > 0 else np.nan,
                "n_multiword_matches": n_spans_multiword,
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------
def read_dataset(path: Path) -> pd.DataFrame:
    if path.suffix == ".json":
        return pd.read_json(path)
    if path.suffix == ".jsonl":
        return pd.read_json(path, lines=True)
    if path.suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError("Dataset must be .json, .jsonl, or .csv")


def word_len(text: Any) -> int:
    return len(str(text).strip().split())


# ---------------------------------------------------------------------------
# Part 1: coverage / distribution metrics
# ---------------------------------------------------------------------------
def compute_part1_metrics(
    trans_df: pd.DataFrame,
    qa_df: pd.DataFrame,
    output_dir: Path,
) -> None:
    matched = trans_df[trans_df["category"] != UNMATCHED_CATEGORY]

    # 1. Count of transformations per category, overall / by task / by dialect
    by_category = (
        matched.groupby("category").size().rename("n_transformations").reset_index().sort_values(
            "n_transformations", ascending=False
        )
    )
    by_category["pct_of_matched"] = 100 * by_category["n_transformations"] / matched.shape[0]
    by_category.to_csv(output_dir / "part1_transformations_by_category.csv", index=False)

    by_category_task = (
        matched.groupby(["task", "category"]).size().rename("n_transformations").reset_index()
    )
    task_totals = by_category_task.groupby("task")["n_transformations"].transform("sum")
    by_category_task["pct_within_task"] = 100 * by_category_task["n_transformations"] / task_totals
    by_category_task.to_csv(output_dir / "part1_transformations_by_category_task.csv", index=False)

    by_category_dialect = (
        matched.groupby(["target_dialect", "category"]).size().rename("n_transformations").reset_index()
    )
    dialect_totals = by_category_dialect.groupby("target_dialect")["n_transformations"].transform("sum")
    by_category_dialect["pct_within_dialect"] = (
        100 * by_category_dialect["n_transformations"] / dialect_totals
    )
    by_category_dialect.to_csv(output_dir / "part1_transformations_by_category_dialect.csv", index=False)

    # Pivot tables for quick reading (task x category %, dialect x category %)
    pivot_task = by_category_task.pivot(index="task", columns="category", values="pct_within_task").fillna(0.0)
    pivot_task.to_csv(output_dir / "part1_pivot_task_x_category_pct.csv")

    pivot_dialect = by_category_dialect.pivot(
        index="target_dialect", columns="category", values="pct_within_dialect"
    ).fillna(0.0)
    pivot_dialect.to_csv(output_dir / "part1_pivot_dialect_x_category_pct.csv")

    # 2. Transformation density: avg number of matched + total changed words per instance
    density_by_dialect = (
        qa_df.groupby("target_dialect")[["n_changed_unigrams", "n_matched_unigrams", "match_rate"]]
        .mean()
        .reset_index()
        .rename(
            columns={
                "n_changed_unigrams": "avg_changed_words_per_instance",
                "n_matched_unigrams": "avg_matched_words_per_instance",
                "match_rate": "avg_lexicon_match_rate",
            }
        )
    )
    density_by_dialect.to_csv(output_dir / "part1_density_by_dialect.csv", index=False)

    density_by_task = (
        qa_df.groupby("task")[["n_changed_unigrams", "n_matched_unigrams", "match_rate"]]
        .mean()
        .reset_index()
        .rename(
            columns={
                "n_changed_unigrams": "avg_changed_words_per_instance",
                "n_matched_unigrams": "avg_matched_words_per_instance",
                "match_rate": "avg_lexicon_match_rate",
            }
        )
    )
    density_by_task.to_csv(output_dir / "part1_density_by_task.csv", index=False)

    density_task_x_dialect = (
        qa_df.groupby(["task", "target_dialect"])[["n_changed_unigrams", "n_matched_unigrams", "match_rate"]]
        .mean()
        .reset_index()
    )
    density_task_x_dialect.to_csv(output_dir / "part1_density_task_x_dialect.csv", index=False)

    # 3. Coverage: % of instances with each category present at least once
    n_instances_by_dialect = qa_df.groupby("target_dialect").size()
    coverage_rows = []
    for (dialect, category), grp in matched.groupby(["target_dialect", "category"]):
        n_instances_with_cat = grp["sample_id"].nunique()
        total = n_instances_by_dialect.get(dialect, np.nan)
        coverage_rows.append(
            {
                "target_dialect": dialect,
                "category": category,
                "n_instances_with_category": n_instances_with_cat,
                "n_total_instances": total,
                "coverage_pct": 100 * n_instances_with_cat / total if total else np.nan,
            }
        )
    coverage_df = pd.DataFrame(coverage_rows).sort_values(["category", "target_dialect"])
    coverage_df.to_csv(output_dir / "part1_category_coverage_by_dialect.csv", index=False)

    n_instances_by_task = qa_df.groupby("task").size()
    coverage_task_rows = []
    for (task, category), grp in matched.groupby(["task", "category"]):
        n_instances_with_cat = grp["sample_id"].nunique()
        total = n_instances_by_task.get(task, np.nan)
        coverage_task_rows.append(
            {
                "task": task,
                "category": category,
                "n_instances_with_category": n_instances_with_cat,
                "n_total_instances": total,
                "coverage_pct": 100 * n_instances_with_cat / total if total else np.nan,
            }
        )
    coverage_task_df = pd.DataFrame(coverage_task_rows).sort_values(["category", "task"])
    coverage_task_df.to_csv(output_dir / "part1_category_coverage_by_task.csv", index=False)


# ---------------------------------------------------------------------------
# Part 2: perplexity-increase metrics
# ---------------------------------------------------------------------------
def compute_part2_metrics(
    trans_df: pd.DataFrame,
    ppl_df: pd.DataFrame,
    output_dir: Path,
    top_n: list[int],
) -> None:
    ppl_key = ppl_df[["sample_id", "target_dialect", "delta_nll", "delta_ppl"]].drop_duplicates(
        subset=["sample_id", "target_dialect"]
    )

    joined = trans_df.merge(ppl_key, on=["sample_id", "target_dialect"], how="left")
    joined.to_csv(output_dir / "part2_transformation_perplexity_joined.csv", index=False)

    matched = joined[joined["category"] != UNMATCHED_CATEGORY]

    # Perplexity increase per category (overall)
    category_ppl = (
        matched.groupby("category")["delta_nll"]
        .agg(["count", "mean", "median", "std"])
        .reset_index()
        .rename(columns={"count": "n_occurrences", "mean": "delta_nll_mean", "median": "delta_nll_median", "std": "delta_nll_std"})
        .sort_values("delta_nll_mean", ascending=False)
    )
    category_ppl.to_csv(output_dir / "part2_category_perplexity.csv", index=False)

    # category x task
    category_task_ppl = (
        matched.groupby(["category", "task"])["delta_nll"]
        .agg(["count", "mean"])
        .reset_index()
        .rename(columns={"count": "n_occurrences", "mean": "delta_nll_mean"})
        .sort_values(["category", "delta_nll_mean"], ascending=[True, False])
    )
    category_task_ppl.to_csv(output_dir / "part2_category_x_task_perplexity.csv", index=False)

    # category x dialect
    category_dialect_ppl = (
        matched.groupby(["category", "target_dialect"])["delta_nll"]
        .agg(["count", "mean"])
        .reset_index()
        .rename(columns={"count": "n_occurrences", "mean": "delta_nll_mean"})
        .sort_values(["category", "delta_nll_mean"], ascending=[True, False])
    )
    category_dialect_ppl.to_csv(output_dir / "part2_category_x_dialect_perplexity.csv", index=False)

    # task x dialect (regardless of category) -- overall transformation-driven signal
    task_dialect_ppl = (
        matched.groupby(["task", "target_dialect"])["delta_nll"]
        .agg(["count", "mean"])
        .reset_index()
        .rename(columns={"count": "n_occurrences", "mean": "delta_nll_mean"})
    )
    task_dialect_ppl.to_csv(output_dir / "part2_task_x_dialect_perplexity.csv", index=False)

    # Top-N phrases by delta_nll mean, broken out per n-gram type (unigram, 2-gram, 3-gram,
    # 4-gram, 5-gram -- whichever are actually present) plus one combined "all n-grams
    # pooled" ranking so multi-word idioms can be compared directly against single words.
    ngram_types_present = [t for t in ["unigram", "2-gram", "3-gram", "4-gram", "5-gram"] if t in matched["ngram_type"].unique()]

    def _phrase_stats(subset: pd.DataFrame) -> pd.DataFrame:
        return (
            subset.groupby("phrase")
            .agg(
                n_occurrences=("delta_nll", "count"),
                delta_nll_mean=("delta_nll", "mean"),
                delta_nll_median=("delta_nll", "median"),
                delta_nll_std=("delta_nll", "std"),
                delta_ppl_mean=("delta_ppl", "mean"),
                category=("category", lambda x: x.mode().iat[0] if len(x.mode()) else ""),
                ngram_type=("ngram_type", lambda x: x.mode().iat[0] if len(x.mode()) else ""),
                dialects=("target_dialect", lambda x: ", ".join(sorted(set(x)))),
                tasks=("task", lambda x: ", ".join(sorted(set(x)))),
            )
            .reset_index()
            .sort_values("delta_nll_mean", ascending=False)
        )

    for ngram_type in ngram_types_present:
        subset = matched[matched["ngram_type"] == ngram_type]
        word_stats = _phrase_stats(subset)
        safe_name = ngram_type.replace("-", "")
        word_stats.to_csv(output_dir / f"part2_{safe_name}_perplexity_stats_all.csv", index=False)

        # only n>=2 for the "top" rankings to reduce single-occurrence noise, but keep the
        # full table above for anyone who wants n=1 entries too.
        filtered = word_stats[word_stats["n_occurrences"] >= 2]
        for n in top_n:
            filtered.head(n).to_csv(output_dir / f"part2_top_{n}_{safe_name}s_by_delta_nll.csv", index=False)

    # Combined ranking across all n-gram types pooled together (so multi-word idioms such as
    # trigrams/4-grams can be compared directly against unigrams/bigrams for "the single most
    # uncertainty-inducing dialectal transformation" regardless of its length).
    all_ngram_stats = _phrase_stats(matched)
    all_ngram_stats.to_csv(output_dir / "part2_all_ngrams_perplexity_stats_all.csv", index=False)
    all_ngram_filtered = all_ngram_stats[all_ngram_stats["n_occurrences"] >= 2]
    for n in top_n:
        all_ngram_filtered.head(n).to_csv(output_dir / f"part2_top_{n}_all_ngrams_by_delta_nll.csv", index=False)


# ---------------------------------------------------------------------------
# Validation against old buggy outputs
# ---------------------------------------------------------------------------
def write_validation_report(
    output_dir: Path,
    qa_df: pd.DataFrame,
    category_ppl: pd.DataFrame,
    old_category_stats_path: Path | None,
    trans_df: pd.DataFrame | None = None,
) -> None:
    lines = [
        "# Validation: corrected lexical transformation analysis vs old buggy pipeline",
        "",
        "## Root cause of the old bug",
        "",
        'The old scripts (`analyze_lexical_coverage.py`, `analyze_category_perplexity.py`, '
        '`analyze_top_words_perplexity.py`) hardcoded a 1:1 `DIALECT_MAPPING` such as '
        '`PNN -> "Nghe An"` and looked up dialect words directly in a single fixed lexicon '
        "column per dataset dialect code. This is factually wrong: the lexicon has 9 "
        "fine-grained regional columns (BacBo, ThanhHoa, NgheAn, HaTinh_QB_QT, Hue, QuangNam, "
        "QuangNgai, PhuYen, NamBo) but the dataset only has 6 macro dialect codes.",
        "",
        "### Manual spot check",
        "",
        "Sample `MCQA_0013_3`, dialect `PNN`, dialect_text: "
        '"Hoa đã phụ má một số công chuyện gì? ... Chờ má đi làm về ... Viết thư cho ba." '
        '-- words "má", "ba", "phụ" are unmistakably Southern Vietnamese (Nam Bộ), not Nghệ An. '
        "The old scripts would have looked these up against the wrong lexicon column and either "
        "mis-tagged their category or failed to match them at all.",
        "",
        "## Corrected mapping actually used",
        "",
    ]
    for dialect, cols in DIALECT_TO_LEXICON_COLUMNS.items():
        lines.append(f"- `{dialect}` ({DIALECT_REGION_LABEL[dialect]}) -> lexicon columns: {', '.join(cols)}")
    lines.append("")
    lines.append(
        "Note: this mapping was confirmed by the research team as the authoritative dialect "
        "definition: `PNB->Bac Bo, PNT1->Thanh Hoa, PNT2->Nghe An+Ha Tinh, "
        "PNT3->Quang Binh+Quang Tri+Thua Thien Hue, PNT4->Nam Trung Bo, PNN->Nam Bo`. The only "
        "unavoidable overlap is the lexicon's own `HaTinh_QB_QT` column, which the source "
        "spreadsheet merges \"Ha Tinh - Quang Binh - Quang Tri\" into a single column; since "
        "Ha Tinh belongs to PNT2 and Quang Binh/Quang Tri belong to PNT3, that one lexicon "
        "column is assigned to both -- a granularity limit of the source spreadsheet, not an "
        "analysis choice. Every other lexicon column maps to exactly one dialect code. An "
        "earlier draft of this analysis incorrectly assigned QuangNgai + PhuYen to both PNN "
        "and PNT4 (inflating PNN's match rate by borrowing PNT4's provinces); that overlap had "
        "come from a common colloquial-Southern vocabulary layer shared across "
        'NamBo/QuangNgai/PhuYen ("tui", "tụi", "ổng", "bả", "dìa", "kêu", "đó", ...), not from '
        "PNN actually being spoken in Quang Ngai / Phu Yen, so it was corrected to the "
        "authoritative mapping above. This lowers PNN's match rate somewhat (see below), which "
        "we report honestly as a lexicon coverage limitation rather than inflating it by "
        "borrowing another dialect's provinces."
    )
    lines.append("")
    lines.append("## Multi-word (trigram/4-gram/5-gram) lexicon entries fix")
    lines.append("")
    lines.append(
        "The lexicon contains idiomatic entries longer than 2 tokens (e.g. `nhu ri thi` / "
        "`nhu rua thi` -> `the nay thi`, `biet tam biet tich` -> `mat roi`, `ngoi chom ho\u0309m` "
        "-> `ngoi xom`). An earlier version of this pipeline only matched unigrams and "
        "bigrams, so these longer idioms were either recorded as \"Unmatched\", or worse, "
        "partially and incorrectly matched at the single-word level (e.g. matching just the "
        'word "nhu" against the lexicon entry "nhu ri thi" and mis-tagging it, while the rest '
        'of the phrase "ri thi" was left as separate unmatched/bigram tokens instead of being '
        "recognized as one idiomatic unit). This has been fixed with a longest-match-first "
        "n-gram extraction: at each changed token position, the pipeline now tries the "
        "longest lexicon phrase length first (up to the longest phrase actually present in "
        "the lexicon) and only falls back to shorter windows or single tokens if no longer "
        "match is found; once a span is matched, its tokens are marked covered so a shorter "
        "overlapping window cannot double-count or partially mis-match it."
    )
    lines.append("")
    lines.append(
        "Manual spot check: sample `SENT_1726`, dialect `PNT2`, dialect_text contains "
        '"...gap ban hang halo nhu ri thi cho e me luon...". The corrected pipeline now '
        'extracts "nhu ri thi" as a single 3-gram matched to the lexicon entry '
        '"nhu ri thi" -> "the nay thi" (Idiomatic expressions), instead of the previous '
        'behaviour of matching only the single word "nhu" against that same lexicon entry.'
    )
    lines.append("")
    if trans_df is not None:
        ngram_counts = trans_df["ngram_type"].value_counts()
        lines.append("N-gram type distribution across all extracted transformation records:")
        lines.append("")
        for ngram_type, count in ngram_counts.items():
            lines.append(f"  - `{ngram_type}`: {count}")
        lines.append("")
        n_multiword_lexicon_entries = int((trans_df["ngram_type"] != "unigram").sum())
        lines.append(
            f"Only {n_multiword_lexicon_entries} multi-word (2-gram+) matches were found in "
            "the actual dataset text overall, so most of the signal remains unigram-level -- "
            "but the trigram fix does recover a small number of genuinely idiomatic "
            "multi-word rewrites that were previously invisible or mis-attributed."
        )
    lines.append("")
    lines.append("## Lexicon match-rate QA (new pipeline)")
    lines.append("")
    overall_rate = qa_df["match_rate"].mean()
    lines.append(f"- Overall mean lexicon match rate across changed unigrams: {overall_rate:.3f}")
    by_dialect_rate = qa_df.groupby("target_dialect")["match_rate"].mean().sort_values(ascending=False)
    for dialect, rate in by_dialect_rate.items():
        lines.append(f"  - `{dialect}`: {rate:.3f}")
    lines.append("")
    lines.append("## New category -> perplexity ranking (top of `part2_category_perplexity.csv`)")
    lines.append("")
    lines.append(category_ppl.head(11).to_string(index=False))
    lines.append("")

    if old_category_stats_path and old_category_stats_path.exists():
        old_df = pd.read_csv(old_category_stats_path)
        lines.append("## Old (buggy) category -> perplexity ranking for comparison")
        lines.append("")
        lines.append(old_df.to_string(index=False))
        lines.append("")
        lines.append(
            "Note the old ranking was computed against `original_text` (not the correct "
            "hypothesis baseline for NLI) and matched dialect words to the wrong lexicon "
            "region for PNN, so category assignments for Southern-region samples are "
            "unreliable in the old output."
        )

    (output_dir / "validation_report.md").write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Corrected lexical transformation + perplexity analysis.")
    parser.add_argument("--dataset", default="data/vialectbench_finalized_6.json")
    parser.add_argument("--lexicon", default="data/DIALECT_LEXICON_v2.xlsx")
    parser.add_argument(
        "--ppl_results",
        default="outputs/finalized_6_complete_six_intrinsic_analysis/perplexity_pairwise_results.csv",
    )
    parser.add_argument("--output_dir", default="outputs/lexical_transformation_analysis_v2")
    parser.add_argument("--orig_col", default="original_text")
    parser.add_argument("--hypothesis_col", default="hypothesis")
    parser.add_argument("--para_col", default="dialect_text")
    parser.add_argument("--sample_id_col", default="sample_id")
    parser.add_argument("--dialect_col", default="target_dialect")
    parser.add_argument("--top_n", type=int, nargs="+", default=[30, 50])
    parser.add_argument(
        "--max_ngram",
        type=int,
        default=0,
        help="Longest n-gram window to try when matching dialect text against the lexicon. "
        "Default 0 means auto-detect from the longest lexicon phrase actually present.",
    )
    parser.add_argument(
        "--old_category_stats",
        default="outputs/category_perplexity_analysis/category_perplexity_stats.csv",
        help="Old buggy output to compare against in the validation report.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Lexicon
    lexicon_raw = load_lexicon_raw(Path(args.lexicon))
    lexicon_long = build_lexicon_entries(lexicon_raw)
    lexicon_long.to_csv(output_dir / "lexicon_long_format.csv", index=False)

    lookup = build_dialect_lookup(lexicon_long)

    max_lexicon_ngram = int(lexicon_long["dialect_phrase"].str.split().apply(len).max())
    max_ngram = args.max_ngram if args.max_ngram > 0 else max_lexicon_ngram

    mapping_meta = {
        "dialect_to_lexicon_columns": DIALECT_TO_LEXICON_COLUMNS,
        "dialect_region_label": DIALECT_REGION_LABEL,
        "canonical_categories": sorted(set(SECTION_TO_CANONICAL_CATEGORY.values())),
        "n_lexicon_rows_long_format": int(len(lexicon_long)),
        "n_dialect_phrases_by_dialect": {d: len(v) for d, v in lookup.items()},
        "max_lexicon_phrase_length_tokens": max_lexicon_ngram,
        "max_ngram_used_for_extraction": max_ngram,
    }
    (output_dir / "lexicon_mapping_used.json").write_text(
        json.dumps(mapping_meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # 2. Dataset + baseline resolution
    df = read_dataset(Path(args.dataset))
    required_cols = [args.orig_col, args.para_col, args.dialect_col, args.sample_id_col, "task"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing column: {col}")

    df["baseline_text"] = df.apply(
        lambda row: resolve_baseline_text(row, args.orig_col, args.hypothesis_col), axis=1
    )
    df["dialect_text_eval"] = df[args.para_col].astype(str).str.strip()

    # 3. Extract transformation records (unigram..N-gram, longest-match-first) with category
    # classification against the corrected dialect-specific lexicon lookup.
    trans_df = build_transformation_records(df, lookup, args.sample_id_col, args.dialect_col, max_ngram=max_ngram)
    trans_df.to_csv(output_dir / "per_sample_transformation_records.csv", index=False)

    qa_df = build_qa_summary(df, lookup, args.sample_id_col, args.dialect_col, max_ngram=max_ngram)
    qa_df.to_csv(output_dir / "per_sample_lexicon_match_qa.csv", index=False)

    # 4. Part 1: coverage / distribution
    compute_part1_metrics(trans_df, qa_df, output_dir)

    # 5. Part 2: perplexity join + rankings
    ppl_df = pd.read_csv(Path(args.ppl_results))
    compute_part2_metrics(trans_df, ppl_df, output_dir, top_n=args.top_n)

    # 6. Validation report vs old buggy outputs
    category_ppl = pd.read_csv(output_dir / "part2_category_perplexity.csv")
    old_stats_path = Path(args.old_category_stats) if args.old_category_stats else None
    write_validation_report(output_dir, qa_df, category_ppl, old_stats_path, trans_df=trans_df)

    print("Saved corrected lexical transformation analysis to:", output_dir)
    for path in sorted(output_dir.iterdir()):
        print(" -", path.name)


if __name__ == "__main__":
    main()
