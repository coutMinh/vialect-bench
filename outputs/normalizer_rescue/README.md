# Normalizer rescue run archive

This experiment evaluates the hard failures selected by `notebooks/EDA.ipynb`:
the standard input is exactly correct, at least half of a case's dialect
variants are exactly wrong, and only those wrong dialect rows are rerun. The
direct-prompting snapshot at commit `d9314e80bf508b2f533063c2f8f2a1be14f57792`
contains 753 such model rows, covering 541 unique case/dialect variants across
138 source cases. This is a subset of the 2,400 dialect rewrites.

`normalized_case_dialects.jsonl` records the original and normalized dialect
fields for every unique variant. The private normalizer is `tarudesu/mbart-large-50` at revision
`83c5b2a3b5078b54e4357c690f42b0bdaa167d08`. Only dialect text was
normalized: sentiment text, NLI hypothesis, QA question, and MCQA question and
four options. Shared premises and contexts came unchanged from each standard
case. Inference used `max_length=128`, batch size 16, and changed 480 of the 541
variants.

The ten `outputs/normalized_direct_prompting/*.jsonl` files hold every model
result. Each row includes the untouched `raw_output` from the LLM, parsed
`prediction`, gold answer, model identity, task, case ID, variant, and prompt
strategy. The local-model rows also include label scores/probabilities where
available; GPT-4o rows include response IDs, token usage, latency, and retry
counts. API keys and Hugging Face tokens are not stored in this archive.

`hard_failure_rows.jsonl` records the exact model/variant selection and original
direct-prompting prediction. `unique_case_dialects.jsonl` records the 541
distinct selected variants. `model_cases/*.jsonl` contains the exact normalized
case objects sent to each model, including the unchanged shared premise or
context and gold answer. `worklist_summary.json` records selection counts.
`run_manifest.json` lists the archived files, row counts, sizes, and SHA-256
checksums so later work can verify that the inputs and outputs are unchanged.

The predictions were produced with the repository's `src.probe_models`,
`src.probe_gemma3`, and `src.probe_openai` runners, using
`--prompt-strategy normalized_direct`, `configs/probe.yaml`, and its 48-token
output limit. No standard rows or other dialect rows were rerun. The EDA notebook
loads that directory to calculate the table at
`paper_artifacts/direct_prompting/paper_table_normalizer_rescue_probe.csv` and
its matching figure. Per model, normalized performance averages QA token F1 or
classification/MCQA accuracy over that model's failed rows. Rescue rate is
exact-match correctness over the same rows, including QA.

To recreate the saved worklist and per-model case files and check that these
normalized inputs still match the finalized probe and direct-prompting snapshot,
run from the repository root:

```bash
python scripts/prepare_normalizer_rescue.py \
  --output-dir outputs/normalizer_rescue \
  --normalized-rows outputs/normalizer_rescue/normalized_case_dialects.jsonl
```

After installing `requirements-analysis.txt`, execute `notebooks/EDA.ipynb` to
regenerate the table and figure from the saved predictions:

```bash
python -m nbconvert --to notebook --execute notebooks/EDA.ipynb \
  --output EDA-executed.ipynb --output-dir .local/rebuttal/normalizer-rescue
```
