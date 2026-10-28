# Q3: Full dialect-to-standard normalization baseline

## Rebuttal framing

The response should acknowledge the omitted mitigation baseline directly:
the submitted benchmark measured model behavior on its original dialect
inputs, and it did not show what preprocessing could recover. The full
controlled run below supplies that missing result. The original standard
score is a useful exact-restoration reference, not a mathematical upper
bound: a normalized paraphrase, model run variation, or a preprocessing
error can yield a score above or below it. Do not claim that model access
or any specific operational obstacle caused the omission unless the authors
can verify that history.

## Controlled experiment

The original direct-prompting experiment has 400 source cases (100 each for
sentiment, NLI, QA, and MCQA), six dialect variants per case, and ten models.
For this baseline, the private `tarudesu/mbart-large-50` normalizer at revision
`83c5b2a3b5078b54e4357c690f42b0bdaa167d08` rewrites only the dialect-bearing
fields: sentiment text, NLI hypothesis, QA question, and MCQA question/options.
Shared standard premises/contexts, gold labels/answers, option order, all
model-specific configurations, prompt wording, output parsing, and the original
standard-input scores are retained. The normalizer uses Vietnamese source/target
codes, maximum length 128, batch size 16, and greedy decoding. Downstream
inference uses the original model runners and their `configs/direct_prompting`
YAMLs, with `normalized_direct` as provenance metadata for the unchanged direct
prompt template. The exact 400-case input sent to the runners is archived in
`outputs/normalizer_full/full_normalized_cases.jsonl`.

The 2,400-source-variant coverage is essential: the earlier 753-row rescue
cohort was selected because the direct model was already wrong, so its rescue
rate cannot estimate an overall performance gain. This full baseline compares
every normalized dialect prediction with the matching original direct
prediction, using the same gold answer and 400 source cases. QA uses token F1
in the main score; the other three tasks use exact-match accuracy. QA exact
match is also exported separately. Paired uncertainty is bootstrapped over
source cases within each task (100 per task), keeping their six dialect
variants and ten model results together.

## Measured recovery

Across the ten models, the original direct-dialect score is **59.40%**, the
normalized-dialect score is **61.25%**, and the original standard-input score
is **62.22%**. Thus normalization gains **1.85 percentage points**, recovering
**65.5%** of the 2.82-point standard-reference gap. The paired, task-stratified
source-case bootstrap interval for the gain is **+1.19 to +2.52 points**
(2,000 replicates). These are averages of QA token F1 and the exact-match
accuracy of the other three tasks, following the original EDA convention.
The result comes from 24,000 paired model/dialect rows, not from the earlier
selected rescue cohort.

Nine of ten models improve. Mistral gains +4.0 points, Qwen 2.5 0.5B +3.2,
SeaLLM +2.6, Llama +2.2, Qwen 2.5 3B +2.0, Qwen 2.5 7B +1.8, Gemma 3 +1.7,
Gemma 2 +1.1, and Vistral +0.7. GPT-4o changes by **−0.8 points**; its
source-case interval is **−2.1 to +0.5 points**. On 717 inputs whose normalized
text was identical to the dialect input, GPT-4o had 25 exact-correctness flips
(13 corrections, 12 regressions); the other models had 0–7 each. This matched-
prompt control is especially important because the original GPT setting uses
the `gpt-4o` provider alias. Among the 1,683 actually changed inputs, GPT-4o's
mean score shift is −1.27 points, while its unchanged-input shift is +0.30.

The pooled gain is largest for **QA: +3.9 points token F1** (QA exact match
+1.6 points), followed by NLI +1.5, sentiment +1.1, and MCQA +0.9. By
dialect group, PNT3 gains +4.7 and PNT2 +3.4 points; PNB changes by −0.3.
The model outputs show **982 incorrect-to-correct** and **676 correct-to-
incorrect** exact-match transitions across all tasks, including QA. Those
transition counts are separate from the main mixed score. Among the nine
local models with label-probability readouts, mean gold-label probability
rises by 0.0102 for NLI, 0.0074 for MCQA, and 0.0065 for sentiment. These
probabilities are not a semantic-quality judgment or a calibrated measure of
certainty.

PNT3 has the largest mean score change. The before/after columns below each
cover the same 400 PNT3 inputs per model (100 per task); QA uses token F1 and
the other tasks use accuracy. A *hard failure* is a model/source-case pair
answered correctly in standard Vietnamese but incorrectly on at least three
of its six dialect variants. The rescue column restricts this cohort to rows
where PNT3 was originally wrong and reports exact-match correctness after
normalization. Thus its denominator varies by model; the pooled 106/175 rate
does not describe all PNT3 inputs. These rescue rates use the full run's
original model settings, not the earlier rescue-only run settings.

| Model | PNT3 before | PNT3 after | Hard-failure rescue |
| --- | ---: | ---: | ---: |
| Qwen 2.5 0.5B | 31.8% | 39.5% | 10/19 (52.6%) |
| Qwen 2.5 3B | 56.4% | 64.5% | 14/21 (66.7%) |
| Qwen 2.5 7B | 61.3% | 65.3% | 13/21 (61.9%) |
| Llama 3.1 8B | 57.6% | 63.3% | 7/15 (46.7%) |
| Mistral 7B | 43.4% | 52.1% | 22/30 (73.3%) |
| Gemma 2 9B | 67.3% | 71.1% | 11/18 (61.1%) |
| Gemma 3 4B | 62.1% | 65.6% | 9/12 (75.0%) |
| Vistral 7B | 47.9% | 49.5% | 3/9 (33.3%) |
| SeaLLM 7B | 56.9% | 62.0% | 13/21 (61.9%) |
| GPT-4o | 75.9% | 75.0% | 4/9 (44.4%) |
| **Overall** | **56.1%** | **60.8%** | **106/175 (60.6%)** |

The complete model table, task and dialect breakdowns, QA EM, and label-
probability comparisons are in `paper_artifacts/direct_prompting/` under the
`paper_table_normalizer_full*` names. The main table is also available as
LaTeX. Mistral and SeaLLM slightly exceed their original standard-input
scores after normalization, another reason to treat standard performance
as a reference rather than a strict upper bound.

## What the normalized text looks like

The checkpoint changed 1,683 of 2,400 variant-level inputs. Of the 457 source
inputs already identical to their paired standard field(s), it changed 82;
274 other inputs became identical to their paired standard, leaving 649 exact
standard matches after normalization. Exact text equality is an audit signal,
not a judgment of semantic quality. Of the 82 edits to already-standard inputs,
68 still differ after applying the benchmark's case, spacing, and basic
punctuation normalization; these are not automatically semantic errors, but
they require inspection. In
particular, MCQA question matches rise
from 131 to 262 of 600, and QA question matches from 137 to 255 of 600.
Sentiment text matches remain 82 of 600 even though 492 sentiment inputs were
changed, because the model often paraphrases rather than reconstructing the
paired standard sentence.

Examples to inspect alongside the raw model outputs:

| Case | Dialect input | Normalized input | What happened across ten models |
| --- | --- | --- | --- |
| `SENT_1239/PNT3` | `anh luôn nghĩ thế giới ni không có chi đáng sợ , cho đến khi anh không còn chộ em nữa` | `anh luôn nghĩ thế giới này không có gì đáng sợ , cho đến khi anh không còn thấy em nữa` | Exact standard reconstruction; five incorrect-to-correct flips, zero regressions. |
| `QA_22620/PNT1` | `Cuộc nổi loạn tại đất nước mô đã làm chua Cuba trở nên lo lắng nếu như Tây Ban Nha rời khỏi Cuba?` | `Cuộc nổi loạn tại đất nước nào đã làm chua Cuba trở nên lo lắng nếu như Tây Ban Nha rời khỏi Cuba?` | Resolves `mô` to `nào` but leaves a typo; five QA exact-match corrections, zero regressions. |
| `NLI_1465/PNT4` | `Jovani Quiles đã nhận tin nhắn hình ảnh về cái bóp, nhẫn qua mạng xã hội mà cổ gửi.` | `Ông ấy đã nhận tin nhắn hình ảnh về cái bóp, nhẫn qua mạng xã hội mà không gửi.` | Introduces an unsupported negation; three correct-to-incorrect flips. |
| `MCQA_0081_3/PNT2` | Option A: `Bịn rịn áo nâu.` | Option A: `Vớ vẩn áo nâu.` | Corrupts an already-standard answer option; two correct-to-incorrect flips. |
| `SENT_5315/PNT3` | 73 whitespace-token angry post beginning `tau két nhất mấy mạ ngoài bắc` | 31 whitespace-token output beginning `tao khờ nhất mấy mẹ ngoài bắc` | Changes an emotion word and omits much of the sentence; all ten models remain incorrect. |

Another PNT3 corruption occurs in `MCQA_0147_4`: the normalizer correctly
changes the question's `mô` to `nào` but changes option B from `Khổ 3.` to
`Khổ 2.` even though the paired standard option is `Khổ 3.`. Four of ten
models move from correct to incorrect; none move from incorrect to correct.

The QA and NLI examples show why gains and regressions must both be measured.
An isolated option or sentence is sometimes ambiguous to the normalizer;
preprocessing can remove a dialect cue while also altering a name, relation,
answer option, or emotionally decisive phrase. Conversely, a reasonable
standardized question can still receive a wrong or verbose downstream answer.
The full case-level outcome file records all such corrections and regressions,
with the original and normalized text for each variant.
`outputs/normalizer_full/selected_case_studies.jsonl` also records these five
cases for every model, including the raw response before and after normalization.
The original GPT configuration names the provider alias `gpt-4o`, so a
provider-side revision between the original and normalized runs cannot be
ruled out from the cached metadata alone. The 717 inputs left unchanged by
the normalizer give a matched-prompt control for observed run variation;
those controls should accompany any GPT-specific causal claim.

## Files and reproducibility

`outputs/normalizer_full/` contains all normalized segments, variant inputs,
exact runner cases, text-change counts, and a run manifest. Each model's full
JSONL in `outputs/normalized_direct_prompting_full/` includes unaltered
`raw_output`, parsed prediction, gold answer, model identity, task, case ID,
and dialect. The nine local models additionally expose label probabilities
where supported; GPT-4o rows contain API response metadata. No API or Hugging
Face token is stored. `scripts/prepare_normalizer_full.py` rebuilds the
normalized inputs from cached segments byte for byte; the private checkpoint
is needed only to regenerate missing segments. The runner/config files used
in Molab match the pinned repository sources by SHA-256 after line-ending
normalization.

## Suggested response to Q3

We thank the reviewer for identifying the missing mitigation baseline. The
submission measured robustness on the benchmark's unmodified dialect inputs
and deferred normalization, leaving the recoverable portion of the performance
gap unmeasured. We have now evaluated a pretrained dialect-to-standard mBART
normalizer on **all 2,400 dialect variants**, not just selected failures. We
reran the original ten models with the same task prompts and model-specific
settings, changing only the dialect-bearing input fields; standard contexts,
gold answers, and option order were preserved.

The average score rises from **59.40% to 61.25%** (**+1.85 points**, 95% paired
source-case bootstrap interval **+1.19 to +2.52**). The original standard-input
score is **62.22%**. Exact restoration to the paired standard wording would
therefore target a **2.82-point reference gap** under stable model behavior;
the observed normalization recovers **65.5%** of it. QA gains **+3.9 F1
points** (exact match **+1.6**); nine of ten models improve. GPT-4o changes
by **−0.8 points**, with an interval spanning zero.

We will add the full model/task/dialect table and qualitative cases to the
revision. The normalizer changes 1,683 inputs; 649 outputs exactly match their
paired standard text. For example, changing `mô` to `nào` helps several models
answer a QA case. It can also introduce an unsupported negation or corrupt an
MCQA answer option, producing regressions. We therefore report both recovery
and failures: normalization mitigates a substantial portion of the measured
gap, but does not reliably preserve every meaning or guarantee that the
downstream model answers correctly.
