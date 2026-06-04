# Pilot Sampling And Model Probing

This code supports the first experimental task: choose 100 low-similarity
samples per task from one runnable source dataset, then probe recommended LLMs
on standard/dialect testcase pairs.

## Selected Pilot Datasets

| Task | Selected dataset | Reason |
| --- | --- | --- |
| Sentiment/emotion | `tridm/UIT-VSMEC` | Publicly loadable now; short social-style Vietnamese utterances are useful for dialect perturbation. |
| NLI | `uitnlp/ViANLI` | Public Vietnamese NLI with entailment, neutral, contradiction labels. |
| QA | `taidng/UIT-ViQuAD2.0` | Public extractive QA. For the pilot, keep context unchanged and dialect-paraphrase only the question. |

Two sentiment candidates are not selected for the runnable pilot yet:
`uitnlp/vietnamese_students_feedback` currently uses a legacy dataset script
that recent `datasets` rejects, and `hung20gg/NEU-ESC` is gated.

## Selection Definition

At this stage we do not define difficulty. The selector applies light quality
filters and then greedily selects samples whose task text is least similar to
already selected samples, using token-set Jaccard similarity.

Task strategies:

- Sentiment: remove very short sentences, then select by low similarity on
  `Sentence`.
- NLI: select by low similarity on `premise + hypothesis`.
- QA: keep only answerable questions with reasonable question and answer length,
  then select by low similarity on `question` only. The context is not used for
  similarity because long contexts dominate the comparison and should not be
  dialect-paraphrased in the pilot.

Run:

```bash
python -m src.vialect_bench.select_samples --sample-size 100 --seed 42
```

Smoke test:

```bash
python -m src.vialect_bench.select_samples --sample-size 5 --max-source-rows 200
```

Outputs are written to `data/pilot_samples/*.jsonl` plus
`data/pilot_samples/selection_summary.csv`. Each selected row keeps the original
Hugging Face columns directly, with only `_task`, `_source_dataset`, and
`_source_index` added for provenance.

## Model Probing

Zero-shot LLM candidates are configured in `configs/models.yaml`: Qwen, Gemma,
Llama, Mistral, Vistral, and SeaLLM. Task-specific pretrained or fine-tuned
models are listed separately and should be filled with exact checkpoints once
the ViDia2Std/sentiment model paths are confirmed.

Run one model on dialect cases:

```bash
python -m src.vialect_bench.probe_models --model-name "Qwen 2.5 7B Instruct"
```

The probe records raw outputs for truthfulness/debugging instead of hiding
model behavior behind only a score.
