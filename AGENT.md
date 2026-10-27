# Coding Agent Guide: VialectBench Rebuttal

This branch, `rebuttal/cleanup`, contains the finalized benchmark and existing
experimental results. Most rebuttal analyses can start from cached predictions
and NLL scores. Read this guide before selecting inputs or running a pipeline.
The [Vietnamese rebuttal handoff](docs/rebuttal.md) maps reviewer questions to
available evidence and records what was removed during cleanup.

Run commands from the **repository root**, using Python 3.10 or newer:

```bash
python -m pip install -r requirements-analysis.txt
python scripts/check_rebuttal_bundle.py
```

The checker uses only the Python standard library. It checks snapshot hashes and
coverage: **400 source cases, 2,400 dialect rewrites, and 10 models with 2,800
prediction rows each**. It does not evaluate semantic preservation or model accuracy.
The analysis dependencies above support reading caches and plotting without a
GPU, model download, or API key.

```text
data/
  Finalized/probe_dialects.jsonl           runnable benchmark, grouped by case
  vialectbench_finalized_6.json            annotation rows, one per case/dialect
  {sentiment,nli,qa,mcqa}_100.jsonl         selected source samples and provenance
  DIALECT_LEXICON_v2.xlsx                  dialect lexicon
  selection_summary.csv                   sampling summary
  qualification_10.jsonl                  qualification examples
outputs/
  direct_prompting/                       raw predictions for 10 models
  finalized_6_complete_six_intrinsic_analysis/  sentence-level PPL/NLL cache
  lexical_substitution_analysis/          alignment, review, swap NLL, figures
paper_artifacts/
  dataset_analysis/                       dataset tables and figures
  direct_prompting/                       downstream tables, figures, hard cases
notebooks/                                EDA, data audit, optional normalizer
scripts/                                  intrinsic and lexical analysis code
src/                                      sampling, conversion, prompting, inference
configs/                                  dataset and model/run configurations
docs/                                     rebuttal handoff and snapshot manifest
submission.pdf                            submitted manuscript
```

## 1. Dataset files and join keys

| File | Unit / current size | Use |
| --- | --- | --- |
| [data/Finalized/probe_dialects.jsonl](data/Finalized/probe_dialects.jsonl) | 400 JSONL case objects; 100 per task | Canonical task inputs for inference and qualitative error analysis |
| [data/vialectbench_finalized_6.json](data/vialectbench_finalized_6.json) | JSON array of 2,400 annotation rows | Annotation metadata, rewritten fields, intrinsic and lexical analysis |
| `data/{sentiment,nli,qa,mcqa}_100.jsonl` | 100 source rows per file | Original text, labels, answers, source indices, lexical trigger features |
| [data/DIALECT_LEXICON_v2.xlsx](data/DIALECT_LEXICON_v2.xlsx) | Excel lexicon | Sampling guidance and lexical categorization |
| [data/selection_summary.csv](data/selection_summary.csv) | Sampling summary | Selection provenance; not the complete original source pools |
| [data/qualification_10.jsonl](data/qualification_10.jsonl) | Qualification examples | Annotation workflow reference |

Task names differ between file families:

| Task | Probe / prediction `task` | Annotation / intrinsic / lexical `task` |
| --- | --- | --- |
| Emotion recognition | `sentiment` | `SENT` |
| Natural language inference | `nli` | `NLI` |
| Extractive question answering | `qa` | `QA` |
| Multiple-choice QA | `mcqa` | `MCQA` |

Probe `id`, prediction `id`, and annotation/cache `sample_id` identify the same
source case, e.g. `MCQA_0002_4`. Treat them as strings. Each probe has a `variants`
object with keys `standard`, `PNB`, `PNN`, `PNT1`, `PNT2`, `PNT3`, `PNT4`.
Annotation rows have `target_dialect` and no additional standard row.

| Task | `variants.standard` | A dialect variant | Gold in probe |
| --- | --- | --- | --- |
| Sentiment | `text` | `text` | `label` |
| NLI | `premise`, `hypothesis` | `hypothesis` | `label` |
| QA | `context`, `question` | `question` | `answers` |
| MCQA | `context`, `question`, `options` | `question`, `options` | `label` (`A`–`D`) |

For error analysis, recover shared premises/contexts from the **standard variant**.
Their absence from a dialect object is intentional. Preserve MCQA option order.
Use `(sample_id, target_dialect)` for annotation/cache joins; lexical tables can
contain multiple spans for this key, so such joins are generally many-to-one.
To map back to source samples, use the stored source metadata and conversion
logic in [src/prepare_annotated_probe_cases.py](src/prepare_annotated_probe_cases.py).
MCQA source identity includes both source index and question index; row position
in a 100-sample file is not the sample ID.

The broad region mapping used in EDA is `PNB → Northern`, `PNN → Southern`,
`PNT1–PNT4 → Central`. These six groups are not individual province labels.

## 2. Existing downstream predictions

[outputs/direct_prompting/](outputs/direct_prompting/) contains these JSONL files:

```text
qwen_2_5_0_5b_instruct.jsonl    qwen_2_5_3b_instruct.jsonl
qwen_2_5_7b_instruct.jsonl      llama_3_1_8b_instruct.jsonl
mistral_7b_instruct_v0_3.jsonl  gemma_2_9b_it.jsonl
gemma_3_4b_it.jsonl            vistral_7b_chat.jsonl
seallm_7b_chat.jsonl           gpt_4o.jsonl
```

Each file has **2,800 rows = 400 cases × 7 variants**, including 400 standard rows
and 2,400 dialect rows. The entire folder contains 28,000 rows. One prediction is
identified by `(model file, task, id, variant)`.

| Fields | Meaning |
| --- | --- |
| `id`, `task`, `variant`, `dialect_group` | Case, task and input variant; EDA resolves variant with `dialect_group or variant` |
| `model_name`, `model_id`, `dataset_part`, `prompt_strategy` | Recorded run metadata; retain these when comparing runs |
| `gold` | String label for classification/MCQA; list of acceptable answers for QA |
| `prediction` | Parsed answer; use for scoring |
| `raw_output` | Model response before parsing; use for parser/error inspection |
| `label_candidates`, `label_logprobs`, `label_probs`, `gold_prob`, `confidence`, `prob_prediction` | Optional candidate-label scoring signals on classification tasks |

Probability fields are not universal. GPT-4o is configured with
`collect_probabilities: false` and has API metadata such as `api_usage` instead.
Do not fill missing probabilities with zero. Dataset `confidence` is annotation
metadata; prediction `confidence` is a model scoring signal with a different meaning.
The result JSONL files do not store all derived EDA metrics or full input text:
join to the probe for text, and calculate scores from `prediction` and `gold`.

### Minimal read-only pairing example

This creates 24,000 matched dialect rows without running models or writing files:

```python
import json
from pathlib import Path
import pandas as pd

rows = []
for path in sorted(Path("outputs/direct_prompting").glob("*.jsonl")):
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            row["model_slug"] = path.stem
            row["variant"] = row.get("dialect_group") or row["variant"]
            rows.append(row)
results = pd.DataFrame(rows)
keys = ["model_slug", "task", "id"]
standard = results.loc[results.variant.eq("standard"), keys + ["prediction"]]
standard = standard.rename(columns={"prediction": "standard_prediction"})
paired = results.loc[~results.variant.eq("standard")].merge(
    standard, on=keys, how="left", validate="many_to_one", indicator=True
)
assert paired["_merge"].eq("both").all()
assert len(paired) == 24000
print(paired.groupby("model_slug").size())
```

### Metric conventions already implemented

The first two code cells of [notebooks/EDA.ipynb](notebooks/EDA.ipynb) define
answer normalization, `exact_match`, `token_f1`, result loading and `pair_df`.
Reuse those definitions when extending existing results.

- Classification/MCQA performance is accuracy. QA performance is token-level F1,
  taking the best match over acceptable answers; QA binary correctness uses EM.
- `performance_drop = standard_score - dialect_score`; positive means degradation.
- A harmful flip is standard correct and dialect incorrect; a beneficial flip is
  the reverse. EDA averages these indicators over **all included paired rows**,
  not only standard-correct or standard-incorrect rows.
- Prediction flip compares normalized predictions, independently of correctness.
  Inter-variant disagreement checks whether dialect predictions differ within a case.
- Raw scores are on a 0–1 scale. Many exported paper tables format them as
  percentages/strings. Use raw predictions for new calculations and label units.
- Compute dialect-specific drops against standard scores for the **same cases**.
  The overall Standard column of an absolute-score table can have a different
  aggregation scope. Cross-task averages combine QA F1 with accuracy and need
  explicit qualification; QA EM can be recomputed from the existing cache.
- Each source ID contributes six dialect rows. Keep that grouping explicit in
  statistical analyses; 2,400 dialect rows are not 2,400 independent source cases.

## 3. Existing intrinsic PPL/NLL results

The directory [outputs/finalized_6_complete_six_intrinsic_analysis/](outputs/finalized_6_complete_six_intrinsic_analysis/)
contains sentence-level reference-LM scores, independent of downstream predictions.

| File / pattern inside this directory | Use |
| --- | --- |
| `dataset_metadata.json` | Input, model, filtering and scoring settings |
| `perplexity_pairwise_results.csv` | 2,400 paired rows with source text, dialect text and cached scores |
| `perplexity_summary_{overall,by_dialect,by_task,task_x_dialect}.csv` | Aggregates and bootstrap intervals |
| `length_summary_*.csv` | Length statistics |
| `distribution_*.csv`, `sample_dialect_coverage.csv` | Composition and completeness |
| `fig_*.svg`, `report_vi.md` | Existing visualizations and interpretation |

The fixed scorer is **`Qwen/Qwen2.5-0.5B`**, max length 512, with 5,000 bootstrap
samples in the recorded run. This is distinct from the instruction-tuned Qwen
models evaluated for downstream task performance.

Important pairwise columns are `sample_id`, `task`, `target_dialect`,
`baseline_text`, `dialect_text_eval`, `orig_nll`, `para_nll`, `orig_ppl`, `para_ppl`,
`delta_nll`, `ratio_ppl`, and `length_ratio`.
`delta_nll = para_nll - orig_nll`; `ratio_ppl = para_ppl / orig_ppl`.
The mean of within-pair PPL ratios differs from the ratio of mean PPL values.
The standard NLI baseline is the **hypothesis**, not the unchanged premise stored
in `original_text`. Inspect `baseline_text` for the exact field being scored.
Length uses whitespace-token counts; `*_n_tokens` refers to model-tokenizer counts.
These scores measure reference-model likelihood shifts and do not by themselves
establish annotation quality or causes of downstream errors.

## 4. Existing lexical analysis results

Start from [outputs/lexical_substitution_analysis/](outputs/lexical_substitution_analysis/).
Its [local README](outputs/lexical_substitution_analysis/README.md) describes the
experiment; the schemas and row counts below describe the current checked snapshot.

| File / directory inside this directory | Contents and use |
| --- | --- |
| `dialectal_substitution_records.csv` | 5,614 aligned spans with categories and `kept_in_analysis`; 4,480 rows are marked `True` |
| `word_alignment/raw_alignment_diff.csv` | 5,614 categorized alignment operations; includes `opcode`, spans and `opcode_seq` |
| `word_alignment/raw_alignment_diff_pre_review.csv` | Earlier backup; retain separately from reviewed alignment |
| `word_alignment/labeled_unknown_term_mappings.csv` | Aggregated labels for previously unknown spans |
| `word_alignment/manual_review/review_sheets/` | Completed per-dialect and shared review sheets; preserve manual work |
| `word_alignment/manual_review/category_guide.txt` | Labeling guide |
| `word_alignment/unified_transformation_records.csv` | 4,909 records, including compound categories; a separate intermediate scope |
| `swap_nll_analysis/dist_*.csv` | Category/task/dialect distributions and substitution density; current category counts sum to 4,480 |
| `swap_nll_analysis/swap_nll_per_occurrence.csv` | 4,429 scored swaps; fields include spans, category, `nll_baseline`, `nll_swapped`, `delta_nll_swap` |
| `swap_nll_analysis/swap_nll_by_{pair,category,dialect}.csv` | Aggregated term-level NLL effects |
| `swap_nll_analysis/swap_nll_top50_pairs.csv` | Ranked subset under the script's occurrence threshold |
| `figures/fig*.png` | Existing category/dialect rankings, heatmaps, distributions and pair plots |

These row counts refer to different filters/stages; do not substitute one for
another as a denominator. A span in `dialectal_substitution_records.csv` carries
sentence-level `delta_nll` joined from the intrinsic cache. The swap experiment
instead replaces a single standard span with its dialect counterpart, keeping
the rest of the baseline text fixed, and calculates
`delta_nll_swap = nll_swapped - nll_baseline` with the same reference LM.
The current implementation replaces the first case-insensitive matching occurrence.
The swap script reads reviewed alignment and applies its own nine-category,
single-label, `replace`-operation filter; it does not directly use `kept_in_analysis`.

## 5. Analysis code and generated paper artifacts

| Entry point | Reads | Produces / purpose |
| --- | --- | --- |
| [notebooks/EDA.ipynb](notebooks/EDA.ipynb) | Annotation v6, finalized probe, direct-prompting outputs; optional normalized outputs | Dataset composition, downstream paired metrics, confidence analyses, paper tables/figures, normalizer hard cases |
| [notebooks/probe_dialects_quality_audit.ipynb](notebooks/probe_dialects_quality_audit.ipynb) | Annotation v6, finalized probe, four source files | Read-only coverage, format and source-consistency reports; does not validate linguistic naturalness |
| [scripts/analyze_vialectbench_data.py](scripts/analyze_vialectbench_data.py) | Annotation v6; optional cached pairwise NLL | Intrinsic summaries, lengths, distributions, SVGs and report |
| [scripts/lexical_data_prep.py](scripts/lexical_data_prep.py) | Annotation v6, lexicon, intrinsic cache, alignment and review files | Steps 1 alignment, 2 labeling, 3 review preparation, 4 review merge, 5 distribution metrics |
| [scripts/utils.py](scripts/utils.py) | Lexicon/dataset utilities | Dependency of lexical prep; also contains a separate analysis CLI |
| [scripts/lexical_swap_nll.py](scripts/lexical_swap_nll.py) | Reviewed alignment and intrinsic baseline texts | Runs reference-LM inference for individual swaps; writes swap-NLL cache |
| [scripts/lexical_visualize.py](scripts/lexical_visualize.py) | Cached `swap_nll_analysis/*.csv` | Recreates lexical PNG figures without model inference |
| [scripts/check_rebuttal_bundle.py](scripts/check_rebuttal_bundle.py) | Manifest, data and cached results | Integrity and coverage checks, no model/API calls |
| [notebooks/ViDia2std_Normalizer.ipynb](notebooks/ViDia2std_Normalizer.ipynb) | External ViDia2Std data / model paths | Optional normalization research; separate environment and machine-specific paths need configuration |

Most `scripts/` paths are relative to the working directory. Keep running from
the repo root. EDA's model order/labels and coverage expectations explicitly list
the ten baseline models; update these definitions when intentionally adding a model.

### Where to find paper-ready results

[paper_artifacts/dataset_analysis/](paper_artifacts/dataset_analysis/) contains
dataset composition/coverage plus existing PPL figures.
[paper_artifacts/direct_prompting/](paper_artifacts/direct_prompting/) contains:

| Table filename / stem | Question answered |
| --- | --- |
| `table_direct_prompting_main` | Performance and flips by model/task |
| `table_direct_prompting_by_dialect` | Performance/drop/flips by model/dialect |
| `table_direct_prompting_coverage.csv` | Counts by model/task/variant |
| `paper_table_task_performance_matrix` | Standard/dialect/drop across tasks |
| `paper_table_task_performance_components` | Long-form standard/dialect/drop scores by model/task |
| `paper_table_task_dialect_performance_matrix` | Model scores by task and variant |
| `paper_table_all_model_dialect_performance` | Absolute standard and dialect scores |
| `paper_table_central_task_performance_radar` | Central dialect performance by task |
| `paper_table_confidence_robustness` | Confidence analysis by model/dialect |
| `paper_table_preserved_correct_confidence_drop` | Confidence erosion on still-correct cases |
| `normalizer_hard_cases.csv`, `normalizer_hard_cases.jsonl` | Selected cases/text for normalization follow-up |
| `paper_table_normalizer_rescue_probe` | Hard-case counts and optional normalized performance/rescue |

Table stems usually have CSV and TeX versions; figures have PNG and PDF versions
with corresponding descriptive names. These are derived exports; use underlying
inputs for new calculations. EDA regenerates its downstream artifacts and dataset
composition/coverage. It **does not regenerate** intrinsic scores or the separate
PPL figures already present in `paper_artifacts/dataset_analysis/`.

## 6. Commands and their side effects

The integrity checker above only reads files. The following commands execute
notebooks or recreate figures. EDA and lexical visualization **rewrite derived
outputs**, even when the executed notebook itself is saved under `.local/`:

```bash
python -m nbconvert --to notebook --execute notebooks/EDA.ipynb --output EDA.executed.ipynb --output-dir .local
python -m nbconvert --to notebook --execute notebooks/probe_dialects_quality_audit.ipynb --output audit.executed.ipynb --output-dir .local
python scripts/lexical_visualize.py
```

EDA rewrites its files under `paper_artifacts/`; lexical visualization rewrites
`outputs/lexical_substitution_analysis/figures/`. The audit reads data and only
saves the executed notebook through nbconvert. Review diffs after regeneration.

To recompute intrinsic summaries from existing cached scores:

```bash
python scripts/analyze_vialectbench_data.py --complete_six_dialects_only --reuse_scores
```

This rewrites summaries/metadata/figures in the intrinsic output directory.
`--reuse_scores` falls back to model scoring if `perplexity_pairwise_results.csv`
is absent in the selected output directory. Check that file exists first.
Running lexical prep with no `--steps` executes all five stages and can replace
alignment/review files. `--dry_run` only applies to review merging (step 4), not
the entire pipeline. Read the code before rerunning preparation or scoring.

For new analyses, write draft code to `scripts/` and scratch outputs to
`.local/rebuttal/<topic>/`. The latter is ignored and will not be shared with
teammates; publish reviewed results to a deliberate tracked location and adjust
`.gitignore` where needed. Existing `outputs/` and `paper_artifacts/` rules allow
specific directories, not arbitrary new top-level experiment folders.

### Inference entry points, if new predictions are required

[src/prompting.py](src/prompting.py) defines prompts/parsing;
[configs/models.yaml](configs/models.yaml) lists models;
[configs/direct_prompting/](configs/direct_prompting/) contains per-model run
configs. Saved result metadata describes the existing run; current config files
alone do not prove every historical inference setting.

```bash
python -m pip install -r requirements.txt
python -m src.probe_models --config configs/direct_prompting/qwen_2_5_3b_instruct.yaml --output .local/qwen_rebuttal.jsonl
python -m src.probe_gemma3 --config configs/direct_prompting/gemma_3_4b_it.yaml --output .local/gemma3_rebuttal.jsonl
python -m src.probe_openai --config configs/direct_prompting/gpt_4o.yaml --output .local/gpt4o_rebuttal.jsonl
```

Gemma 3 uses its dedicated runner. These commands can download models or call APIs;
they are unnecessary for reading/reanalyzing existing predictions. Export `HF_TOKEN`
or `OPENAI_API_KEY` when needed; runners do not load `.env` automatically.
To compare conversion output without replacing the shared benchmark:

```bash
python -m src.prepare_annotated_probe_cases --output .local/probe_rebuilt.jsonl
```

[src/select_samples.py](src/select_samples.py) and
[configs/datasets.yaml](configs/datasets.yaml) implement source selection.
Running selection again is a dataset-generation operation, not a prerequisite
for analyzing this frozen benchmark.

## 7. Current gaps and handoff conventions

- **Normalization:** no normalized inference outputs are included. EDA checks
  `outputs/normalized_direct_prompting/`, `outputs/normalized_direct/`, then
  `outputs/direct_prompting_normalized/`, selecting the first populated directory.
  Rescue-table dashes mean unavailable results, not measured zero performance.
- **Review ratings:** annotation v6 has review metadata but lacks the complete
  per-reviewer Yes/Partial/No ratings and adjudication history needed to reproduce
  paper Table 1, reviewer agreement or a fully passing subset.
- **Source description:** the manuscript names VIMQA for MCQA, while the checked
  source file/config records `uitnlp/vimmrc2.0`. Flag this for the authors rather
  than silently changing provenance.
- **Available evidence:** full original sampling pools, province-level dialect
  labels and open-ended generation experiments are not supplied as dedicated
  benchmark resources here. Do not infer these from existing aggregate scores.
- **Local archive:** `.local/rebuttal-cleanup/legacy/` contains removed projects
  only on the cleanup machine. It is ignored, absent from a fresh clone, and not
  a dependency of the current pipeline. Older tracked files remain in `main` history.
- **Shared baseline:** preserve cached predictions, manual review sheets and
  finalized data while developing new analyses. Keep notebook outputs cleared in
  source control; share derived artifacts and reproducible analysis code instead.

[docs/rebuttal_manifest.json](docs/rebuttal_manifest.json) records hashes for the
119 current data/cache/artifact files. Text hashing normalizes CRLF to LF; binary
files are hashed unchanged. The manifest was created after inference and provides
snapshot integrity, not proof of the original model/data revisions. After an
intentional result change has been reviewed, refresh it explicitly:

```bash
python scripts/check_rebuttal_bundle.py --update-manifest
```

Do not refresh hashes merely to suppress an unexplained mismatch. When handing
off new work, record the input files, cohort/filter, metric and denominator,
output path, and command needed to reproduce it.
