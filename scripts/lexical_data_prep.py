#!/usr/bin/env python3
"""
lexical_data_prep.py
====================
Complete lexical data preparation pipeline. Runs four sequential steps:

  Step 1  alignment   — word-level diff + lexicon lookup → raw_alignment_diff.csv
  Step 2  label       — rule-based labeling of unknown spans → labeled_unknown_term_mappings.csv
  Step 3  review_prep — export per-dialect expert review sheets (manual_review/)
  Step 4  review_merge— merge completed review sheets back into raw_alignment_diff.csv
  Step 5  metrics     — distribution (Part 1) + perplexity-increase (Part 2) metrics

Run all steps:
    python scripts/lexical_data_prep.py

Skip alignment (re-use existing raw_alignment_diff.csv):
    python scripts/lexical_data_prep.py --skip_alignment

Run only specific steps:
    python scripts/lexical_data_prep.py --steps 3 4   # only review_prep + review_merge
    python scripts/lexical_data_prep.py --steps 5     # only metrics

All paths relative to vialect_github/vialect-bench/
"""

from __future__ import annotations

import argparse
import difflib
import re
import subprocess
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE     = Path(".")
DATA     = BASE / "data"
ALIGN    = BASE / "outputs/lexical_substitution_analysis/word_alignment"
OUT      = BASE / "outputs/lexical_substitution_analysis"
PPL_CSV  = BASE / "outputs/finalized_6_complete_six_intrinsic_analysis/perplexity_pairwise_results.csv"

ALIGN_CSV     = ALIGN / "raw_alignment_diff.csv"
LABELED_CSV   = ALIGN / "labeled_unknown_term_mappings.csv"
BACKUP_CSV    = ALIGN / "raw_alignment_diff_pre_review.csv"
REVIEW_DIR    = ALIGN / "manual_review"
SHEETS_DIR    = REVIEW_DIR / "review_sheets"
SWAP_DIR      = OUT / "swap_nll_analysis"

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------
UNKNOWN_LABEL   = "Unknown"
FILTER_CATS     = {"Noise/Non-linguistic", "Orthographic variant", UNKNOWN_LABEL}
KEEP_CATEGORIES = {
    "Interrogatives", "Pronouns", "Demonstratives/Deixis",
    "Negation/Function words", "Connectives/Aspect markers",
    "Predicate vocabulary", "Kinship/Honorific terms",
    "Discourse particles", "Idiomatic expressions",
}
VALID_CATEGORIES = KEEP_CATEGORIES | {
    "Noise/Non-linguistic", "Orthographic variant",
    "Multi-word idiom", "Out-of-scope",
}
DIALECT_ORDER = ["PNB", "PNN", "PNT1", "PNT2", "PNT3", "PNT4"]
TASK_ORDER    = ["MCQA", "NLI", "QA", "SENT"]

# ===========================================================================
# STEP 1 — ALIGNMENT
# ===========================================================================

def _import_utils():
    from utils import (
        DEFAULT_DIALECTS, DIALECT_TO_LEXICON_COLUMNS,
        build_dialect_lookup, build_lexicon_entries,
        load_lexicon_raw, read_dataset, resolve_baseline_text, tokenize,
    )
    return (DEFAULT_DIALECTS, DIALECT_TO_LEXICON_COLUMNS,
            build_dialect_lookup, build_lexicon_entries,
            load_lexicon_raw, read_dataset, resolve_baseline_text, tokenize)


def _build_reverse_lookup(lexicon_long, dialect_to_columns, DEFAULT_DIALECTS):
    lookup = {d: defaultdict(list) for d in DEFAULT_DIALECTS}
    for dialect_code, region_cols in dialect_to_columns.items():
        subset = lexicon_long[lexicon_long["region_column"].isin(region_cols)]
        for _, row in subset.iterrows():
            for std_variant in str(row["standard_word"]).split("/"):
                std_variant = std_variant.strip()
                if not std_variant:
                    continue
                entry = {
                    "category":      row["category"],
                    "dialect_phrase": row["dialect_phrase"],
                    "region_column":  row["region_column"],
                }
                if entry not in lookup[dialect_code][std_variant]:
                    lookup[dialect_code][std_variant].append(entry)
    return {d: dict(v) for d, v in lookup.items()}


def _classify_pair(standard_span, dialect_span, dialect_code,
                   dialect_lookup, standard_lookup):
    dialect_matches  = dialect_lookup.get(dialect_code, {}).get(dialect_span, [])
    if dialect_matches:
        cats = sorted({m["category"] for m in dialect_matches})
        return "dialect_phrase", dialect_matches, "; ".join(cats)
    standard_matches = standard_lookup.get(dialect_code, {}).get(standard_span, [])
    if standard_matches:
        cats = sorted({m["category"] for m in standard_matches})
        return "standard_word", standard_matches, "; ".join(cats)
    return "none", [], UNKNOWN_LABEL


def _build_alignment_records(df, dialect_lookup, standard_lookup,
                              sample_id_col, dialect_col, tokenize_fn):
    records = []
    for _, row in df.iterrows():
        dialect_code  = row[dialect_col]
        baseline_text = row["baseline_text"]
        dialect_text  = row["dialect_text_eval"]
        if dialect_code == "PNB" and baseline_text.strip() == dialect_text.strip():
            continue
        baseline_words = tokenize_fn(baseline_text)
        dialect_words  = tokenize_fn(dialect_text)
        sm = difflib.SequenceMatcher(None, baseline_words, dialect_words, autojunk=False)
        for idx, (tag, i1, i2, j1, j2) in enumerate(sm.get_opcodes()):
            if tag == "equal":
                continue
            std_span  = " ".join(baseline_words[i1:i2])
            dial_span = " ".join(dialect_words[j1:j2])
            direction, matches, cat = _classify_pair(
                std_span, dial_span, dialect_code, dialect_lookup, standard_lookup)
            records.append({
                "sample_id":             row[sample_id_col],
                "task":                  row["task"],
                "target_dialect":        dialect_code,
                "opcode":                tag,
                "opcode_seq":            idx,
                "standard_span":         std_span,
                "dialect_span":          dial_span,
                "standard_span_len":     i2 - i1,
                "dialect_span_len":      j2 - j1,
                "is_known":              direction != "none",
                "match_direction":       direction,
                "category":              cat,
                "lexicon_standard_word": "; ".join(sorted({m.get("standard_word","") for m in matches})),
                "lexicon_region_column": "; ".join(sorted({m["region_column"] for m in matches})),
            })
    return pd.DataFrame(records)


def step1_alignment(args):
    print("\n[Step 1] Word-level alignment ...")
    (DEFAULT_DIALECTS, DIALECT_TO_LEXICON_COLUMNS,
     build_dialect_lookup, build_lexicon_entries,
     load_lexicon_raw, read_dataset, resolve_baseline_text, tokenize_fn) = _import_utils()

    lexicon_raw  = load_lexicon_raw(Path(args.lexicon))
    lexicon_long = build_lexicon_entries(lexicon_raw)
    dialect_lookup  = build_dialect_lookup(lexicon_long)
    standard_lookup = _build_reverse_lookup(
        lexicon_long, DIALECT_TO_LEXICON_COLUMNS, DEFAULT_DIALECTS)

    df = read_dataset(Path(args.dataset))
    df["baseline_text"]     = df.apply(
        lambda r: resolve_baseline_text(r, args.orig_col, args.hypothesis_col), axis=1)
    df["dialect_text_eval"] = df[args.para_col].astype(str).str.strip()

    raw_df = _build_alignment_records(
        df, dialect_lookup, standard_lookup,
        args.sample_id_col, args.dialect_col, tokenize_fn)

    ALIGN.mkdir(parents=True, exist_ok=True)
    raw_df.to_csv(ALIGN_CSV, index=False)
    n, k = len(raw_df), int(raw_df["is_known"].sum())
    print(f"  raw_alignment_diff.csv: {n} records  |  lexicon-matched: {k} ({k/n:.1%})")
    return raw_df

# ===========================================================================
# STEP 2 — RULE-BASED LABELING
# ===========================================================================

_KINSHIP_EXACT = {
    ("bố","ba"),("cha","ba"),("mẹ","má"),("bố mẹ","ba má"),("cha mẹ","ba má"),
    ("bố mình","ba tui"),("bố mẹ","bọ mạ"),("cha mẹ","ba mạ"),("cha","bố"),
}
_INTERROG_EXACT = {
    ("gì","chi"),("sao","răng"),("thế nào","răng"),("đâu","mô"),("nào","mô"),
    ("như thế nào","a răng"),("như thế nào","à răng"),("như thế nào","ơ răng"),
    ("như thế nào","ra răng"),("như thế nào","ren"),("như thế nào","ra ren"),
    ("thế nào","ren"),("tại sao","vì răng"),("làm gì","mần chi"),
    ("làm sao","mần răng"),("khi nào","khi mô"),("bao giờ","hồi mô"),
    ("gì","cấy chi"),("gì","cái chi"),("nào","chi"),("thế nào","ri"),
    ("thế này","ri"),("làm gì","mần cấy chi"),("làm gì","mắc chi"),
}
_CONNECTIVE_EXACT = {
    ("vào","vô"),("đến","tới"),("và","với"),("vì","tại"),("do","tại"),
    ("bởi vì","do cấy"),("để","đặng"),("mãi","hoài"),("và","rồi"),
    ("vào","hồi"),("khi","lúc"),("khi","chừng"),("thật","thiệt"),
    ("thực","thật"),("lúc","hồi"),("hoặc","hay"),("nên","niên"),
    ("sau này","mai mốt"),("rằng","là"),
}
_NEGATION_EXACT = {
    ("không","nỏ"),("không","chả"),("không","hông"),("chẳng","nọ"),
    ("tôi không","tui hông"),("không về","hông dìa"),
    ("không làm gì","nỏ mần chi"),("chưa","nỏ"),("đâu","nỏ"),("chẳng","nỏ"),
}
_PARTICLE_EXACT = {
    ("nhé","nhen"),("nhé","nghe"),("nhá","nghe"),("vâng","dạ"),("vâng","nghe"),
    ("ư","hả"),("à","hả"),("hả","à"),("ạ","dạ"),("nhỉ","hỉ"),
}
_PRONOUN_EXACT = {
    ("tôi","tui"),("tao","tau"),("mày","mi"),("nó","hắn"),("nó","hấn"),
    ("mình","tau"),("chúng ta","tụi mình"),("chúng tôi","bọn tau"),
    ("chúng","tụi"),("bọn","tụi"),("họ","bọn hấn"),("ta","mình"),
    ("cô ấy","cổ"),("ông","ổng"),("bà","bả"),("cô","ả"),
    ("cậu","thằng"),("cậu","cu"),("mình","mềng"),("mình","miềng"),
}
_ORTHO_PAIRS = [
    ("đã","đạ"),("đã","đả"),("nữa","nửa"),("trâu","tru"),("giữa","trửa"),
    ("nước","nác"),("lúa","ló"),("nũa","nữa"),("phục","phúc"),("dàn","dần"),
    ("luyện","liệng"),("tuyết","tuyệt"),("chứ","chớ"),("cũng","cụng"),
    ("của","cụa"),("giữa","giửa"),("lầm","nhầm"),("người","ngài"),
]
_NOISE_EXACT = {"bb","cmm","chg","kkkkk","lol","wtf","or"}
_NOISE_RE = re.compile(
    r'^(k+h*e*h*e*|h+e+h+|hehe|hihi|lol|cmm|wtf|bb|chg|kkk+|[a-z]{1,2}$)',
    re.IGNORECASE)

_PRONOUN_TOK_STD  = {"tôi","tao","mày","mình","ta","cậu","họ","chúng","bọn","nó"}
_PRONOUN_TOK_DIAL = {"tui","tau","mi","hắn","hấn","mềng","miềng","ả","tụi","choa"}
_KINSHIP_TOK_STD  = {"mẹ","bố","cha","ba","má","anh","chị","ông","bà","vợ","chồng"}
_KINSHIP_TOK_DIAL = {"má","ba","mạ","bọ","mệ","eng","dông","cấy"}
_INTERROG_TOK_STD  = {"gì","đâu","sao","nào","ai","bao"}
_INTERROG_TOK_DIAL = {"chi","mô","răng","ren","ri","rứa"}
_DEMO_TOK_STD  = {"này","đó","ấy","kia","thế","vậy","đây","đấy","các","những","mọi"}
_DEMO_TOK_DIAL = {"ni","nớ","rứa","tê","mấy","cấy"}
_NEG_TOK_STD   = {"không","chẳng","chưa"}
_NEG_TOK_DIAL  = {"nỏ","chả","hông","hổng","khung"}
_PART_TOK_STD  = {"nhé","nhỉ","ạ","ư","à","hả","vâng","dạ","thôi"}
_PART_TOK_DIAL = {"nhen","hả","nghe","dạ","ha","hỉ","hè","chớ","liền"}
_CONN_TOK_STD  = {"và","vì","vào","đến","đã","rồi","để","tại","do","bởi","khi","thật"}
_CONN_TOK_DIAL = {"với","vô","tới","đặng","tại","hồi","rồi","thiệt","đạ","hoài","lúc"}


def _tok(s: str):
    return set(re.sub(r'[^\w\s]', ' ', str(s).lower()).split())


def _classify_unknown(std: str, dial: str) -> str:
    s, d = str(std).strip().lower(), str(dial).strip().lower()

    for a, b in _KINSHIP_EXACT:
        if (s == a and d == b) or (s == b and d == a):
            return "Kinship/Honorific terms"
    for a, b in _INTERROG_EXACT:
        if (s == a and d == b) or (s == b and d == a):
            return "Interrogatives"
    for a, b in _CONNECTIVE_EXACT:
        if (s == a and d == b) or (s == b and d == a):
            return "Connectives/Aspect markers"
    for a, b in _NEGATION_EXACT:
        if (s == a and d == b) or (s == b and d == a):
            return "Negation/Function words"
    for a, b in _PARTICLE_EXACT:
        if (s == a and d == b) or (s == b and d == a):
            return "Discourse particles"
    for a, b in _PRONOUN_EXACT:
        if (s == a and d == b) or (s == b and d == a):
            return "Pronouns"
    for a, b in _ORTHO_PAIRS:
        if (s == a and d == b) or (s == b and d == a):
            return "Orthographic variant"

    if s in _NOISE_EXACT or d in _NOISE_EXACT:
        return "Noise/Non-linguistic"
    if _NOISE_RE.match(s) or _NOISE_RE.match(d):
        return "Noise/Non-linguistic"
    if re.match(r'^["\'\s,;.!?\-]+$', s) or re.match(r'^["\'\s,;.!?\-]+$', d):
        return "Noise/Non-linguistic"
    if re.match(r'^\d+$', s) or re.match(r'^\d+$', d):
        return "Noise/Non-linguistic"

    ts, td = _tok(s), _tok(d)

    if ts & _PRONOUN_TOK_STD or td & _PRONOUN_TOK_DIAL:
        return "Pronouns"
    if (ts & _KINSHIP_TOK_STD and len(s.split()) <= 3) or \
       (td & _KINSHIP_TOK_DIAL and len(d.split()) <= 2):
        return "Kinship/Honorific terms"
    if (ts & _INTERROG_TOK_STD or td & _INTERROG_TOK_DIAL) and \
       s not in _CONN_TOK_STD and d not in _CONN_TOK_DIAL:
        return "Interrogatives"
    if (ts & _DEMO_TOK_STD or td & _DEMO_TOK_DIAL) and \
       s not in _CONN_TOK_STD and d not in _CONN_TOK_DIAL:
        return "Demonstratives/Deixis"
    if ts & _NEG_TOK_STD or td & _NEG_TOK_DIAL:
        return "Negation/Function words"
    if ts & _PART_TOK_STD or td & _PART_TOK_DIAL:
        return "Discourse particles"
    if (ts & _CONN_TOK_STD or td & _CONN_TOK_DIAL) and \
       not (ts & _PRONOUN_TOK_STD or td & _PRONOUN_TOK_DIAL) and \
       not (ts & _KINSHIP_TOK_STD or td & _KINSHIP_TOK_DIAL):
        return "Connectives/Aspect markers"
    if len(s.split()) >= 3 and len(d.split()) >= 3:
        return "Idiomatic expressions"
    return "Predicate vocabulary"


def step2_label(raw_df: pd.DataFrame):
    print("\n[Step 2] Rule-based labeling of unknown spans ...")
    unknown = raw_df[~raw_df["is_known"]].copy()
    agg = (
        unknown.groupby(["target_dialect", "opcode", "standard_span", "dialect_span"])
        .agg(
            n_occurrences      =("sample_id", "count"),
            n_distinct_samples =("sample_id", "nunique"),
            tasks              =("task", lambda s: "; ".join(sorted(set(s)))),
            example_sample_ids =("sample_id", lambda s: "; ".join(list(dict.fromkeys(s))[:5])),
        )
        .reset_index()
        .sort_values(["target_dialect", "n_occurrences"], ascending=[True, False])
    )
    agg["category"] = [
        _classify_unknown(str(r.standard_span), str(r.dialect_span))
        for _, r in agg.iterrows()
    ]
    ALIGN.mkdir(parents=True, exist_ok=True)
    agg.to_csv(LABELED_CSV, index=False)
    print(f"  labeled_unknown_term_mappings.csv: {len(agg)} pairs")
    for cat, cnt in agg["category"].value_counts().items():
        print(f"    {cat:<35} {cnt}")
    return agg

# ===========================================================================
# STEP 3 — REVIEW PREP (export manual review sheets)
# ===========================================================================

CATEGORY_GUIDE = """LEXICAL CATEGORY GUIDE FOR UNKNOWN TERM REVIEW
================================================
Assign exactly ONE category to each (standard → dialect) pair.

1. Interrogatives       — gì→chi, thế nào→răng, nào→mô, sao→răng
2. Pronouns             — tôi→tau, mình→tau, chúng tôi→tụi tui
3. Demonstratives/Deixis — này→ni, kia→nớ, bây giờ→chừ
4. Negation/Function words — không→nỏ, không→hông, chưa→chửa
5. Connectives/Aspect markers — đã→đạ, và→với, vì→tại, vào→vô
6. Predicate vocabulary — làm→mần, thấy→chộ, nói→kêu
7. Kinship/Honorific terms — bố→ba, mẹ→má, ông→bọ
8. Discourse particles  — nhỉ→hè, nhé→nghe, à→hả
9. Idiomatic expressions — van xin→năn nỉ, hão huyền→viển vông
10. Noise/Non-linguistic — typos, OCR artifacts, numbers only
11. Orthographic variant — same word, different spelling/tone: nũa→nữa
12. Multi-word idiom     — full phrase→phrase (use sparingly)
13. Out-of-scope         — proper nouns, numbers, entities

Fill in the CATEGORY column. Add NOTES if uncertain.
"""

_DIALECT_INFO = {
    "PNT1": "Thanh Hoa (North-Central I)",
    "PNT2": "Nghe An–Ha Tinh (North-Central II)",
    "PNT3": "Quang Binh–Hue (North-Central III)",
    "PNT4": "Da Nang–Binh Thuan (South-Central)",
    "PNN":  "Southern Vietnamese (HCMC, Mekong Delta)",
    "PNB":  "Northern Vietnamese (standard-adjacent)",
}


def step3_review_prep(raw_df: pd.DataFrame):
    print("\n[Step 3] Exporting manual review sheets ...")
    SHEETS_DIR.mkdir(parents=True, exist_ok=True)
    (REVIEW_DIR / "category_guide.txt").write_text(CATEGORY_GUIDE, encoding="utf-8")

    rep = raw_df[(raw_df["opcode"] == "replace") & (raw_df["category"] == "Unknown")].copy()
    if len(rep) == 0:
        print("  No Unknown rows — nothing to export.")
        return

    pair_dial_count = (
        rep.groupby(["standard_span", "dialect_span"])["target_dialect"]
        .nunique().reset_index(name="n_dialects"))
    shared_pairs = pair_dial_count[pair_dial_count["n_dialects"] >= 2][
        ["standard_span", "dialect_span"]]

    for dialect in _DIALECT_INFO:
        sub = rep[rep["target_dialect"] == dialect]
        unique = (
            sub.groupby(["standard_span", "dialect_span"])
            .agg(n_occurrences=("sample_id", "count"),
                 sample_ids=("sample_id", lambda x: "; ".join(sorted(set(x))[:5])),
                 tasks=("task", lambda x: ", ".join(sorted(set(x)))))
            .reset_index().sort_values("n_occurrences", ascending=False))
        unique["dialect"]  = dialect
        unique["CATEGORY"] = ""
        unique["NOTES"]    = ""
        unique["is_shared"] = unique.apply(
            lambda r: ((shared_pairs["standard_span"] == r["standard_span"]) &
                       (shared_pairs["dialect_span"]  == r["dialect_span"])).any(), axis=1)
        cols = ["standard_span","dialect_span","n_occurrences","is_shared",
                "tasks","sample_ids","dialect","CATEGORY","NOTES"]
        path = SHEETS_DIR / f"unknown_review_{dialect}.csv"
        unique[cols].to_csv(path, index=False, encoding="utf-8-sig")
        print(f"  {path.name}: {len(unique)} pairs")

    # Shared sheet
    shared_full = (
        rep[rep[["standard_span","dialect_span"]].apply(tuple, axis=1)
            .isin(shared_pairs.apply(tuple, axis=1))]
        .groupby(["standard_span","dialect_span"])
        .agg(n_occurrences_total=("sample_id","count"),
             dialects=("target_dialect", lambda x: ", ".join(sorted(set(x)))),
             tasks=("task", lambda x: ", ".join(sorted(set(x)))),
             sample_ids=("sample_id", lambda x: "; ".join(sorted(set(x))[:5])))
        .reset_index().sort_values("n_occurrences_total", ascending=False))
    shared_full["CATEGORY"] = ""
    shared_full["NOTES"]    = ""
    cols = ["standard_span","dialect_span","n_occurrences_total",
            "dialects","tasks","sample_ids","CATEGORY","NOTES"]
    shared_full[cols].to_csv(
        SHEETS_DIR / "unknown_review_SHARED.csv", index=False, encoding="utf-8-sig")
    print(f"  unknown_review_SHARED.csv: {len(shared_full)} cross-dialect pairs")
    print(f"  Category guide: {REVIEW_DIR}/category_guide.txt")
    print(f"  After review run: python scripts/lexical_data_prep.py --steps 4")


# ===========================================================================
# STEP 4 — REVIEW MERGE
# ===========================================================================

def step4_review_merge(dry_run: bool = False):
    print("\n[Step 4] Merging expert review sheets ...")
    if not SHEETS_DIR.exists():
        print("  No review sheets found — skipping.")
        return

    frames = []
    for sheet in sorted(SHEETS_DIR.glob("unknown_review_*.csv")):
        df = pd.read_csv(sheet, dtype=str).fillna("")
        done = df[df["CATEGORY"].str.strip() != ""]
        if len(done) == 0:
            continue
        frames.append(done[["standard_span","dialect_span","CATEGORY"]])
        print(f"  {sheet.name}: {len(done)}/{len(df)} labeled")

    if not frames:
        print("  No completed sheets — skipping merge.")
        return

    labels = pd.concat(frames).drop_duplicates(subset=["standard_span","dialect_span"])
    invalid = labels[~labels["CATEGORY"].isin(VALID_CATEGORIES)]
    if len(invalid):
        print(f"  WARNING: {len(invalid)} invalid CATEGORY values — kept as-is.")

    align = pd.read_csv(ALIGN_CSV)
    before = (align["category"].isin({"Unknown","Multi-word idiom"})).sum()
    lookup = labels.set_index(["standard_span","dialect_span"])["CATEGORY"].to_dict()

    def resolve(row):
        if row["category"] not in ("Unknown", "Multi-word idiom"):
            return row["category"]
        key = (str(row["standard_span"]).strip(), str(row["dialect_span"]).strip())
        return lookup.get(key, row["category"])

    align["category"] = align.apply(resolve, axis=1)
    after = (align["category"].isin({"Unknown","Multi-word idiom"})).sum()
    print(f"  Resolved: {before - after} rows  |  Remaining: {after}")

    if not dry_run:
        if not BACKUP_CSV.exists():
            pd.read_csv(ALIGN_CSV).to_csv(BACKUP_CSV, index=False)
        align.to_csv(ALIGN_CSV, index=False)
        print(f"  Updated: {ALIGN_CSV}")


# ===========================================================================
# STEP 5 — METRICS (Part 1 distribution + Part 2 perplexity)
# ===========================================================================

def _build_unified(raw_df: pd.DataFrame, labeled_agg: pd.DataFrame) -> pd.DataFrame:
    known = raw_df[raw_df["is_known"]].copy()
    known["source"] = "lexicon_match"

    unk = raw_df[~raw_df["is_known"]].copy()
    for df in (unk, labeled_agg):
        for col in ("standard_span","dialect_span"):
            df[col + "_key"] = df[col].astype(str).str.strip().str.lower()
    label_map = labeled_agg.set_index(
        ["target_dialect","opcode","standard_span_key","dialect_span_key"]
    )["category"].to_dict()
    unk["category"] = unk.apply(
        lambda r: label_map.get(
            (r["target_dialect"], r["opcode"],
             r["standard_span_key"], r["dialect_span_key"]), UNKNOWN_LABEL), axis=1)
    unk["source"] = "alignment_labeled"

    cols = ["sample_id","task","target_dialect","opcode",
            "standard_span","dialect_span","category","source"]
    unified = pd.concat([known[cols], unk[cols]], ignore_index=True)

    before = len(unified)
    unified = unified[~unified["category"].isin(FILTER_CATS)].copy()
    n_removed = before - len(unified)
    print(f"  Filtered: {before} → {len(unified)} rows (removed {n_removed} Noise/Ortho/Unknown)")
    return unified


def step5_metrics(raw_df: pd.DataFrame, labeled_agg: pd.DataFrame):
    print("\n[Step 5] Computing distribution metrics ...")
    SWAP_DIR.mkdir(parents=True, exist_ok=True)

    # Read active rows from the full records file (kept_in_analysis=True)
    records_path = OUT / "dialectal_substitution_records.csv"
    if records_path.exists():
        full = pd.read_csv(records_path)
        if "kept_in_analysis" in full.columns:
            unified = full[full["kept_in_analysis"] == True].copy()
            unified = unified[~unified["category"].str.contains(";", na=False)]
            print(f"  Reading from dialectal_substitution_records.csv: {len(unified)} active rows")
        else:
            print("  WARNING: kept_in_analysis column missing — rebuilding from raw diff")
            unified = _build_unified(raw_df, labeled_agg)
    else:
        print("  dialectal_substitution_records.csv not found — rebuilding from raw diff")
        unified = _build_unified(raw_df, labeled_agg)

    # --- Part 1: substitution distribution ---
    by_cat = (unified.groupby("category").size()
              .rename("n_transformations").reset_index()
              .sort_values("n_transformations", ascending=False))
    by_cat["pct_of_total"] = 100 * by_cat["n_transformations"] / len(unified)
    by_cat.to_csv(SWAP_DIR / "dist_by_category.csv", index=False)

    by_task_cat = unified.groupby(["task","category"]).size().rename("n").reset_index()
    by_task_cat["pct_within_task"] = (
        100 * by_task_cat["n"] / by_task_cat.groupby("task")["n"].transform("sum"))
    by_task_cat.to_csv(SWAP_DIR / "dist_by_task_category.csv", index=False)

    by_dial_cat = unified.groupby(["target_dialect","category"]).size().rename("n").reset_index()
    by_dial_cat["pct_within_dialect"] = (
        100 * by_dial_cat["n"] / by_dial_cat.groupby("target_dialect")["n"].transform("sum"))
    by_dial_cat.to_csv(SWAP_DIR / "dist_by_dialect_category.csv", index=False)

    density = (unified.groupby(["target_dialect","task"])
               .agg(n_spans=("sample_id","count"), n_samples=("sample_id","nunique"))
               .reset_index())
    density["avg_spans_per_sample"] = density["n_spans"] / density["n_samples"]
    density.to_csv(SWAP_DIR / "dist_density_task_x_dialect.csv", index=False)

    print("  Distribution stats saved to swap_nll_analysis/")
    print("  NOTE: perplexity metrics are computed by lexical_swap_nll.py (term-level, from scratch)")


# ===========================================================================
# CLI
# ===========================================================================

def parse_args():
    p = argparse.ArgumentParser(description="Lexical data preparation pipeline.")
    p.add_argument("--dataset",        default="data/vialectbench_finalized_6.json")
    p.add_argument("--lexicon",        default="data/DIALECT_LEXICON_v2.xlsx")
    p.add_argument("--orig_col",       default="original_text")
    p.add_argument("--hypothesis_col", default="hypothesis")
    p.add_argument("--para_col",       default="dialect_text")
    p.add_argument("--sample_id_col",  default="sample_id")
    p.add_argument("--dialect_col",    default="target_dialect")
    p.add_argument("--skip_alignment", action="store_true",
                   help="Re-use existing raw_alignment_diff.csv.")
    p.add_argument("--steps", nargs="+", type=int, default=[1,2,3,4,5],
                   help="Steps to run (1=align 2=label 3=review_prep 4=review_merge 5=metrics).")
    p.add_argument("--dry_run", action="store_true",
                   help="For step 4: show changes without writing.")
    return p.parse_args()


def main():
    args = parse_args()
    steps = set(args.steps)

    raw_df      = None
    labeled_agg = None

    if args.skip_alignment or 1 not in steps:
        if ALIGN_CSV.exists():
            raw_df = pd.read_csv(ALIGN_CSV)
            print(f"Loaded existing raw_alignment_diff.csv ({len(raw_df)} rows)")
    if 1 in steps and not args.skip_alignment:
        raw_df = step1_alignment(args)

    if 2 in steps:
        if raw_df is None:
            raw_df = pd.read_csv(ALIGN_CSV)
        labeled_agg = step2_label(raw_df)

    if 3 in steps:
        if raw_df is None:
            raw_df = pd.read_csv(ALIGN_CSV)
        step3_review_prep(raw_df)

    if 4 in steps:
        step4_review_merge(dry_run=args.dry_run)
        raw_df = pd.read_csv(ALIGN_CSV)  # reload after merge

    if 5 in steps:
        if raw_df is None:
            raw_df = pd.read_csv(ALIGN_CSV)
        if labeled_agg is None:
            labeled_agg = pd.read_csv(LABELED_CSV)
        step5_metrics(raw_df, labeled_agg)

    print(f"\n✓ Done. Outputs: {OUT}")


if __name__ == "__main__":
    main()
