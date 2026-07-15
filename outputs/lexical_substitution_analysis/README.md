# Lexical Substitution Analysis

## Purpose

This analysis identifies **which word-level substitutions cause model confusion** under
Vietnamese dialectal variation, and **why** certain dialects and tasks are harder than others.

It operates at the **term level** — measuring how much each individual standard→dialect word
swap increases a language model's uncertainty — rather than at the sentence level. This makes
it complementary to both the intrinsic analysis and the direct prompting experiments. It also
serves as a **data quality check**, verifying that annotators changed only dialectal vocabulary
and preserved sentence structure and task labels throughout.

---

## How This Differs from the Other Two Analyses

| | Intrinsic analysis | Direct prompting | This analysis |
|---|---|---|---|
| **Unit** | Full sentence | Full sentence | Individual word swap |
| **Measurement** | Δ NLL on entire dialect sentence vs standard | Task accuracy drop across models | Δ NLL from swapping one term at a time |
| **What it answers** | How much does the dialect sentence shift the model's distribution overall? | Do models get the right answer on dialect inputs? | Which specific word substitution types cause the most distributional shift? |
| **Perplexity scope** | Sentence-level (all dialect words changed at once) | Not perplexity-based | **Term-level** (isolated single-word substitution, everything else held standard) |
| **Reference model** | Qwen/Qwen2.5-0.5B | 10 models (GPT-4o, Llama, Qwen, etc.) | Qwen/Qwen2.5-0.5B |

The term-level Δ NLL is computed by replacing exactly one word in the standard sentence with
its dialectal equivalent, running the model, and measuring `NLL(swapped) − NLL(original)`.
No new model training is performed — the reference model is the same fixed scorer used in the
intrinsic analysis. The main artifact is **`dialectal_substitution_records.csv`** — 5,614
aligned word spans with linguistic category labels and a `kept_in_analysis` flag (4,480 rows
pass the filter into the NLL experiment).

---

## Key Results

### Substitution category distribution (4,480 classified substitutions)

| Category | n | % of total |
|---|---|---|
| Predicate vocabulary | 961 | 21.5% |
| Interrogatives | 948 | 21.2% |
| Negation/Function words | 621 | 13.9% |
| Pronouns | 543 | 12.1% |
| Connectives/Aspect markers | 538 | 12.0% |
| Demonstratives/Deixis | 489 | 10.9% |
| Kinship/Honorific terms | 239 | 5.3% |
| Discourse particles | 85 | 1.9% |
| Idiomatic expressions | 56 | 1.2% |

### Term-level Δ NLL by category

| Category | Mean Δ NLL per swap | n swaps |
|---|---|---|
| **Interrogatives** | **0.479** | 936 |
| Pronouns | 0.326 | 532 |
| Demonstratives/Deixis | 0.304 | 478 |
| Connectives/Aspect markers | 0.253 | 538 |
| Negation/Function words | 0.252 | 617 |
| Kinship/Honorific terms | 0.185 | 234 |
| Discourse particles | 0.153 | 85 |
| **Predicate vocabulary** | **0.122** | 954 |
| Idiomatic expressions | 0.108 | 55 |

**Interrogatives cause the most confusion per swap (0.479)** despite being the second most
frequent category. The paper's motivating example (gì→chi misread as chi phí) is directly
supported: interrogative swaps are both common and highly disruptive.

**Predicate vocabulary is the most frequent substitution type (20.3%) but causes the least
confusion per swap (0.122)** — the model can handle content verb synonyms in context.

### Term-level vs sentence-level Δ NLL by dialect

| Dialect | Sent. Δ NLL | Term Δ NLL | Spans/sentence |
|---|---|---|---|
| PNT3 | **0.925** | 0.337 | **3.16** |
| PNT2 | 0.795 | **0.378** | 2.41 |
| PNT4 | 0.495 | 0.277 | 2.50 |
| PNT1 | 0.468 | 0.338 | 1.71 |
| PNN | 0.316 | 0.160 | 3.05 |
| PNB | −0.104 | −0.034 | 1.48 |

The two analyses decompose dialect difficulty into two factors — **per-swap confusion cost**
and **substitution density** (spans per sentence). PNT3 is hardest at sentence level (0.925)
because it has the highest density (3.34 swaps/sentence), not because each individual swap is
the most confusing. PNT2 ranks first at the term level (0.378/swap) with fewer swaps per
sentence. PNN illustrates the opposite case: high density (3.10) but low per-swap cost (0.160),
meaning its vocabulary is more familiar to the model. PNB is near-neutral at both levels,
consistent with its near-standard linguistic profile.

### Most confusing substitution pairs (n ≥ 5)

| Standard → Dialect | Category | Δ NLL | n |
|---|---|---|---|
| như thế nào → ơ răng | Interrogatives | 0.946 | 5 |
| như thế nào → ra ren | Interrogatives | 0.824 | 22 |
| tôi → tau | Pronouns | 0.718 | 8 |
| thế nào → ren | Interrogatives | 0.709 | 17 |
| thế nào → răng | Interrogatives | 0.707 | 104 |
| như thế nào → ra răng | Interrogatives | 0.698 | 11 |
| làm gì → mần cấy chi | Interrogatives | 0.684 | 5 |
| như thế nào → a răng | Interrogatives | 0.675 | 30 |

Interrogative substitutions dominate the high-confusion pairs, confirming the category-level
finding.

---

## Folder Structure

```
lexical_substitution_analysis/
├── dialectal_substitution_records.csv   ← full labeled substitution inventory (5,614 rows,
│                                           kept_in_analysis column marks the 4,480 used)
├── README.md
├── word_alignment/
│   ├── raw_alignment_diff.csv           — word-level diff for all 2,400 rewrites (5,614 rows)
│   ├── labeled_unknown_term_mappings.csv — rule-labeled unknown spans
│   ├── raw_alignment_diff_pre_review.csv — backup before expert review merge
│   └── manual_review/
│       ├── category_guide.txt           — 13-category labeling guide
│       └── review_sheets/               — per-dialect expert label CSVs (completed)
├── swap_nll_analysis/
│   ├── dist_by_category.csv            — substitution counts per category
│   ├── dist_by_task_category.csv       — substitution counts per task × category
│   ├── dist_by_dialect_category.csv    — substitution counts per dialect × category
│   ├── dist_density_task_x_dialect.csv — avg substitution spans per sentence
│   ├── swap_nll_per_occurrence.csv      — per-swap term-level Δ NLL (4,429 rows)
│   ├── swap_nll_by_pair.csv             — mean Δ NLL per (standard, dialect) pair
│   ├── swap_nll_by_category.csv         — mean Δ NLL per category
│   ├── swap_nll_by_dialect.csv          — mean Δ NLL per dialect
│   └── swap_nll_top50_pairs.csv         — top 50 pairs by mean Δ NLL (n ≥ 2)
└── figures/
    ├── figA — category ranking by term-level Δ NLL
    ├── figB — dialect ranking by term-level Δ NLL
    ├── figC — category × dialect heatmap
    ├── figD — top 25 substitution pairs (n ≥ 5)
    ├── figE — distribution violin plot per category
    ├── figF — frequency vs Δ NLL scatter (n ≥ 3)
    ├── figF2 — frequency vs Δ NLL scatter (n ≥ 5, all labeled)
    ├── figG — category × task heatmap
    └── figH — task × dialect heatmap
```

---

## Replication

Run from `vialect_github/vialect-bench/`:

```bash
# Step 1 — Intrinsic NLL cache (skip if already exists)
python scripts/analyze_vialectbench_data.py \
  --input data/vialectbench_finalized_6.json \
  --output_dir outputs/finalized_6_complete_six_intrinsic_analysis \
  --model Qwen/Qwen2.5-0.5B --complete_six_dialects_only

# Steps 1–5: alignment → labeling → review prep → review merge → distribution metrics
python scripts/lexical_data_prep.py

# Term-level NLL experiment (requires GPU / MPS, ~30 min on Apple M-series)
python scripts/lexical_swap_nll.py

# Figures
python scripts/lexical_visualize.py
```

Re-run only metrics and figures after updating review sheets:
```bash
python scripts/lexical_data_prep.py --steps 4 5
python scripts/lexical_swap_nll.py
python scripts/lexical_visualize.py
```
