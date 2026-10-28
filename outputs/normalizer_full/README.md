# Full dialect-to-standard normalization experiment

This archive evaluates the full finalized VialectBench probe: 400 source cases
with six dialect variants each, or 2,400 unique dialect inputs. The frozen
benchmark and original direct-prompting results are from commit
d9314e80bf508b2f533063c2f8f2a1be14f57792.

The normalizer is the private tarudesu/mbart-large-50 checkpoint at revision
83c5b2a3b5078b54e4357c690f42b0bdaa167d08. It used the same generation
settings as the earlier rescue experiment: max_length=128, batch size 16,
greedy decoding (do_sample=False, num_beams=1), and Vietnamese source and
target language codes. normalized_case_dialects.jsonl stores both the source
and normalized text for all 2,400 variants. It changed 1,683 variants.
`normalized_segments.jsonl` archives all 3,294 distinct input text segments
and their normalizer outputs. `full_normalized_cases.jsonl` is the exact
400-case file passed to the downstream runners.

Only the dialect-bearing task fields were normalized: sentiment text, NLI
hypothesis, QA question, and MCQA question and four options. The standard
premise/context, task label or answer, and option order were retained.
For unchanged inputs and normalized inputs identical to their paired standard
form, the constructed task prompts were verified byte-for-byte identical to
the original direct-prompting prompts.

The 10 files in outputs/normalized_direct_prompting_full/ contain the raw
downstream model responses, parsed predictions, and run metadata. They are
generated with the repository's original src.probe_models,
src.probe_gemma3, and src.probe_openai runners and each model's original
configs/direct_prompting/*.yaml configuration. The only runner overrides
are the case-file and output paths and --prompt-strategy normalized_direct,
which uses the same prompt template as direct. No standard rows are rerun.
The full run's goal is 2,400 normalized predictions per model.
`remote_runner_hashes.json` records SHA-256 hashes for the 16 source and
configuration files used in Molab; all 16 match this repository's pinned files
after normalizing Windows line endings.
`text_change_summary.csv` describes text changes by task and dialect. Overall,
649 normalized inputs match their paired standard text exactly. The normalizer
also changed 82 inputs that already matched the standard text; this includes
both superficial edits and occasional meaning-changing errors. These counts
describe text equality, not linguistic correctness.
All 541 variants overlapping the earlier rescue run have byte-identical
normalized fields in both archives.
`field_match_summary.csv` adds question/option-level exact matches.
`prompt_equivalence_report.json` verifies identical direct prompts for 717
unchanged inputs and identical standard prompts for 649 exact-standard inputs.
`gpt_usage_summary.json` totals the GPT-4o API usage recorded in its raw rows.
`run_logs.tar.gz` preserves the queue status, per-model execution logs, and
the exact queue wrapper; it was checked for token-like secret strings before
archiving.

To recreate the inputs from the saved segment outputs without loading a model,
or to fill missing segments with the pinned checkpoint and `HF_TOKEN`, run:

    python scripts/prepare_normalizer_full.py

To rerun downstream inference using the original configurations, run:

    python scripts/run_normalizer_full_predictions.py

To analyze all completed model files, run:

    python scripts/analyze_normalizer_full.py

This writes model-level and task/dialect-level tables under
paper_artifacts/direct_prompting/ and an analysis_summary.json here. The
case_level_outcomes.csv here has one row per variant with text and correction/
regression counts across the ten models. Model and pooled gold-probability
tables cover the nine local models where probabilities are available.
`selected_case_studies.jsonl` expands five representative success/failure
cases across all ten models, including both raw responses and exact-match
outcomes; regenerate it with `python scripts/export_normalizer_case_studies.py`.
`python scripts/write_normalizer_full_manifest.py` validates raw-output
coverage and records file/configuration hashes in run_manifest.json. Text
hashes and byte counts normalize CRLF to LF so the manifest survives Git
checkouts on Windows and Linux; binary files are hashed unchanged. The
main score follows notebooks/EDA.ipynb: QA token F1 and exact-match accuracy
for the other three tasks. Error-correction and regression counts use exact
match for all four tasks, including QA. This full evaluation should be kept
separate from the selected hard-failure rescue results in
outputs/normalizer_rescue/.
