# Technical Report: Dialectal-English Robustness Benchmarks and Design Lessons for a Vietnamese Dialect Benchmark

**Date:** 2026-05-20  
**Purpose:** summarize five dialectal-English robustness papers and extract benchmark-design lessons for a Vietnamese dialect-robustness project.  
**Evidence base:** five local PDFs supplied in the workspace, parsed with `document_parse`; public source URLs are listed at the end.

---

## 1. Executive Summary

The five papers show a clear progression in dialect robustness benchmarking:

1. **Multi-VALUE (2023)** introduced a scalable, linguistically grounded **rule-based perturbation framework** for English dialects. It converts Standard American English (SAE) into dialectal variants using documented morphosyntactic features from eWAVE, enabling synthetic stress tests and augmentation.
2. **DialectBench (2024)** broadened the scope beyond English by aggregating existing dialect/variety datasets across **281 varieties, 40 clusters, and 10 NLP tasks**, with explicit dialect-gap metrics.
3. **Evaluating Dialect Robustness via Conversation Understanding (2024/2025)** moved from isolated benchmark prompts to **dialogue understanding**, using US English vs Indian English conversations and Multi-VALUE-style synthetic transformations.
4. **ReDial / Assessing Dialect Fairness and Robustness in Reasoning Tasks (2025)** argued that rule-based perturbations are not enough: human-written AAVE reasoning prompts caused larger failures than synthetic transformations. It introduced a high-quality parallel **SE–AAVE reasoning benchmark**.
5. **EnDive (2025)** combined LLM-based dialect rewriting, eWAVE-guided few-shot prompts, BLEU filtering, automatic and native-speaker validation, and comparison against Multi-VALUE. It targets multiple English dialects and 12 reasoning/NLU tasks.

**Main implication for Vietnamese:** a strong Vietnamese dialect benchmark should not rely only on normalization or automatic perturbation. A defensible design should combine: (i) a linguistically explicit dialect-feature inventory, (ii) synthetic perturbation for scalable stress tests, (iii) human-authored or human-validated parallel dialect data for authenticity, (iv) task coverage beyond sentiment/classification into reasoning and conversation, and (v) dialect-gap metrics comparing dialectal inputs to meaning-equivalent standard Vietnamese inputs.

---

## 2. Cross-Paper Comparison

| Paper | Main benchmark/resource | Dialects/varieties | Data creation method | Main tasks | Key technical contribution |
|---|---:|---|---|---|---|
| **Multi-VALUE** | Multi-VALUE framework + synthetic/gold benchmarks | 50 English dialects; stress tests mainly AppE, ChcE, IndE, CollSgE, UAAVE | Rule-based morphosyntactic perturbations grounded in eWAVE; native-speaker validation; synthetic augmentation | CoQA QA, Spider semantic parsing, WMT19 MT | Scalable, controllable dialect perturbation rules; shows dialect stress tests predict gold dialect trends directionally |
| **DialectBench** | DIALECTBENCH | 281 varieties in 40 clusters | Aggregates existing datasets; standardizes variety metadata and cluster mappings | DEP, POS, NER, DId, SA, TC, NLI, MRC, EQA, MT | Large multilingual/dialectal evaluation suite; dialect-gap metrics across clusters/varieties |
| **Conversation Robustness** | M-MD3 | en-US, en-IN, en-MV, en-TR | Extends MD3 dialogue game data; masks target words; en-MV via Multi-VALUE, en-TR via GPT-4 normalization | Target Word Prediction, Target Word Selection | Tests dialect robustness in conversation understanding, not only single-turn prompts |
| **ReDial** | ReDial | Standard English and AAVE | Human AAVE rewrites by 13 annotators; naturalness and correctness checks; synthetic perturbation comparison | Algorithm, logic, math, integrated planning/reasoning | Shows human dialect data causes larger LLM failures than synthetic perturbations; reasoning is dialect brittle |
| **EnDive** | ENDIVE | AAVE, ChcE, CollSgE, IndE, JamE | GPT-4o few-shot rewriting guided by eWAVE examples; BLEU filtering; native-speaker validation; compared to Multi-VALUE | 12 tasks: BoolQ, MultiRC, WSC, SST-2, COPA, HumanEval, MBPP, GSM8K, SVAMP, FOLIO, LogicBench YN/MCQ | Hybrid LLM-based dialect translation pipeline with validation and broad task coverage |

---

## 3. Paper-by-Paper Technical Summary

### 3.1 Multi-VALUE: A Framework for Cross-Dialectal English NLP (2023, arXiv:2212.08011)

**Key contribution.** Multi-VALUE introduces a **controllable, interpretable, rule-based dialect transformation framework** for English. Its central insight is that dialect robustness can be evaluated by applying meaning-preserving **difference rules** to standard benchmark inputs. These rules inject dialectal morphosyntactic features while preserving task labels.

**Technique.**

- Uses the **Electronic World Atlas of Varieties of English (eWAVE)** as the linguistic basis.
- Implements **189 linguistic feature perturbations** across **50 English dialects**.
- Uses spaCy-based syntactic conditions to apply transformations only when grammatical contexts match.
- Organizes perturbations into categories such as pronouns, noun phrases, tense/aspect, mood, verb morphology, negation, agreement, relativization, complementation, adverbial subordination, adverbs/prepositions, discourse, and word order.
- Validates a subset of perturbation rules with native speakers via MTurk; the paper reports high rule accuracies for many validated rules.

**Benchmarking tasks and datasets.**

- **Conversational QA:** CoQA; evaluates BERT and RoBERTa on SAE and dialect-transformed CoQA.
- **Semantic parsing:** Spider; transforms only the natural-language query while leaving database schema and SQL unchanged.
- **Machine translation:** WMT19 English-to-Chinese/German/Gujarati/Russian using NLLB 615M and 1.3B.
- Dialects emphasized in experiments: **Appalachian English (AppE), Chicano English (ChcE), Indian English (IndE), Colloquial Singapore English (CollSgE), Urban African American English (UAAVE)**.

**Important findings.**

- Dialectal transformations produce significant drops across QA, semantic parsing, and MT.
- Multi-dialect augmentation improves cross-dialect robustness but may slightly reduce SAE performance, suggesting an interference tradeoff.
- Synthetic stress tests directionally match gold human-translated CoQA results, but synthetic feature density can overestimate some dialect drops.

**Limitations relevant to Vietnamese.**

- The approach depends on a rich dialectological feature inventory like eWAVE; comparable Vietnamese resources may be less standardized.
- It focuses mostly on morphosyntax, not lexical variation, pragmatics, spelling, or conversational norms.
- Synthetic perturbations can miss context-dependent dialect usage.

---

### 3.2 DIALECTBENCH: An NLP Benchmark for Dialects, Varieties, and Closely-Related Languages (2024)

**Key contribution.** DialectBench is a large-scale benchmark that unifies many existing dialect/variety datasets under one evaluation framework. It shifts the field from single-dialect case studies to broad comparative evaluation across varieties.

**Technique.**

- Aggregates datasets for **281 varieties** across **40 language clusters**.
- Uses cluster-variety mappings based on mutual intelligibility, phylogenetic similarity, geographic proximity, Glottocodes, and metadata.
- Defines standard and non-standard varieties inside clusters and computes dialectal performance gaps.
- Evaluates both conventional multilingual encoders and LLM-style prompting.

**Benchmarking tasks and datasets.**

DialectBench covers **10 text-level tasks**:

1. Dependency parsing — Universal Dependencies, TwitterAAE, Singlish.
2. POS tagging — Universal Dependencies, Singlish, Noisy Dialects.
3. NER — WikiANN, Norwegian NER.
4. Dialect identification — MADAR, DMT, Greek, DSL-TL, Swiss German datasets.
5. Sentiment analysis — TSAC, TUNIZI, DzSentiA, SaudiBank, MAC, ASTD, AJGT, OCLAR.
6. Topic classification — SIB-200.
7. Natural language inference — XNLI translate-test.
8. Multiple-choice reading comprehension — Belebele.
9. Extractive QA — SD-QA.
10. Machine translation — CODET, TIL-MT.

**Models and evaluation.**

- Uses **mBERT** and **XLM-R** for most tasks.
- Uses **NLLB 600M/1.3B** for MT.
- Uses **Mistral 7B** for in-context learning experiments.
- Defines dialect gap metrics as relative performance decreases against a representative or high-resource variety.

**Important findings.**

- Strong performance disparities appear both across language clusters and within clusters.
- Low-resource varieties often perform poorly, especially when absent from pretraining or lacking task-specific data.
- Benchmark construction requires careful metadata normalization; dialect names alone are insufficient.

**Limitations relevant to Vietnamese.**

- Aggregation is powerful but depends on available datasets; Vietnamese dialect datasets may be sparse.
- DialectBench is broad but not deeply tailored to a single language’s sociolinguistic structure.
- For a Vietnamese project, the main transferable idea is the **cluster/variety metadata scheme and gap metric**, not the dataset content itself.

---

### 3.3 Evaluating Dialect Robustness of Language Models via Conversation Understanding (2024/2025, arXiv:2405.05688)

**Key contribution.** This paper evaluates dialect robustness through **conversation understanding**, using human-human taboo-game dialogues rather than isolated benchmark sentences. It introduces **M-MD3**, a target-word-masked extension of MD3.

**Technique.**

- Starts from MD3, which contains taboo-style conversations between human participants.
- Selects **US English (en-US)** and **Indian English (en-IN)** conversations.
- Creates four subsets:
  - **en-US:** original US English conversations.
  - **en-IN:** original Indian English conversations.
  - **en-MV:** en-US transformed into Indian English using Multi-VALUE.
  - **en-TR:** en-IN normalized by GPT-4 to remove dialectal information.
- Masks the target word at the final turn where the guesser identifies it.

**Benchmarking tasks and datasets.**

- **Target Word Prediction (TWP):** given a masked conversation, generate the target word.
- **Target Word Selection (TWS):** given a masked conversation plus candidate target words, select the correct one.
- Dataset sizes after splitting:
  - en-US: train 62 / valid 41 / test 311.
  - en-IN: train 31 / valid 21 / test 160.
  - en-MV: train 49 / valid 33 / test 250.
  - en-TR: train 23 / valid 17 / test 131.

**Models and metrics.**

- Models: GPT-4, GPT-3.5 Turbo, Llama-3 70B Chat.
- Fine-tuning: GPT-3.5 and Llama-3; GPT-4 only pre-trained/API evaluation.
- Metrics: exact-match accuracy and Sentence-BERT cosine similarity.

**Important findings.**

- Models perform better on en-US than en-IN across configurations.
- Removing dialectal information from en-IN can improve pre-trained model performance, suggesting models are closer to normalized/US-like English distributions.
- Multi-VALUE-transformed en-MV reduces performance compared with en-US, showing that synthetic dialectal features can induce robustness failures.
- Error analysis shows failures are not only linguistic: shared cultural context, ambiguous descriptions, public-figure references, and incomplete/fragmented descriptions matter.

**Limitations relevant to Vietnamese.**

- It treats en-IN as homogeneous, which the authors acknowledge is unrealistic. A Vietnamese benchmark must avoid treating “Central Vietnamese” or “Southern Vietnamese” as a single uniform register without metadata.
- Dialogue tasks expose cultural/pragmatic gaps that sentence perturbation misses; this is especially relevant for Vietnamese regional vocabulary, kinship/pronoun systems, particles, and conversational style.

---

### 3.4 Assessing Dialect Fairness and Robustness of Large Language Models in Reasoning Tasks / ReDial (2025)

**Key contribution.** ReDial provides a human-annotated parallel benchmark for **reasoning under dialectal variation**, focusing on Standard English versus AAVE. Its strongest contribution is empirical: **human-written AAVE prompts produce larger and qualitatively different LLM failures than synthetic perturbations**.

**Technique.**

- Constructs **1,216 parallel Standard English–AAVE prompts**.
- Hires **13 AAVE annotators** to rewrite prompts naturally while preserving meaning.
- Performs AAVE-speaker naturalness checks and non-AAVE/LLM-assisted correctness checks; no item is rejected solely by LLM judgment.
- Compares human AAVE rewrites to synthetic AAVE-style perturbations based on Multi-VALUE-like rules.

**Benchmarking tasks and datasets.**

ReDial samples from seven reasoning datasets:

| Category | Source datasets | Size |
|---|---|---:|
| Algorithm | HumanEval, MBPP | 164 + 150 |
| Logic | LogicBench, FOLIO original+perturbed | 200 + 162 |
| Math | GSM8K, SVAMP | 150 + 150 |
| Integrated reasoning | AsyncHow | 240 |
| **Total** | — | **1,216** |

**Models and evaluation.**

- Models include GPT-o1, GPT-4o, GPT-4, GPT-3.5, Claude 3.5 Sonnet, Llama-3/3.1, Mistral/Mixtral, and Phi-3 variants.
- Uses direct prompting and zero-shot Chain-of-Thought where applicable.
- Converts all task outputs to pass/fail, including code evaluation via EvalPlus.

**Important findings.**

- Nearly all models show lower pass rates on AAVE than Standard English.
- Average direct-prompt pass rate across tasks drops from **0.597 SE to 0.529 AAVE**.
- Integrated reasoning has especially large relative drops.
- CoT and standardization prompts help somewhat but do not remove the gap and may increase inference cost.
- Typos/noise and synthetic AAVE perturbations do not fully reproduce the severity of human AAVE failures.
- Error analysis identifies morphosyntax, non-standard verb forms, omission of auxiliaries/articles, double negatives, informal quantity expressions, and phrase-level conversational norms as contributors.

**Limitations relevant to Vietnamese.**

- ReDial covers only AAVE and English reasoning tasks, but the methodology is highly relevant.
- The key lesson is to include **human dialect rewrites** for at least a core evaluation set, because rule-based transformations may underestimate real dialect brittleness.

---

### 3.5 EnDive: A Cross-Dialect Benchmark for Fairness and Performance in Large Language Models (2025)

**Key contribution.** EnDive builds a broad English dialect benchmark using **LLM-based few-shot dialect rewriting** guided by eWAVE examples, compares it to Multi-VALUE, filters near-identical rewrites, and validates translation quality with automatic and human evaluation.

**Technique.**

- Translates SAE tasks into five dialects:
  - African American Vernacular English (AAVE)
  - Chicano English (ChcE)
  - Colloquial Singapore English (CollSgE)
  - Indian English (IndE)
  - Jamaican English (JamE)
- Uses GPT-4o few-shot prompting with eWAVE examples.
- Compares against Multi-VALUE rule-based translations.
- Applies **sentence-level BLEU filtering**, removing rewrites with BLEU ≥ 0.70 against SAE to avoid near-identical examples.
- Evaluates translation quality with ROUGE diversity, BARTScore, GPT-4o fluency judgments, LLM pairwise preferences, and native-speaker Likert ratings.

**Benchmarking tasks and datasets.**

EnDive covers **12 tasks** across language understanding, algorithmic understanding, math, and logic:

| Area | Datasets and sample sizes |
|---|---|
| Language understanding | BoolQ 1,000; MultiRC 1,000; WSC 659; SST-2 1,000; COPA 500 |
| Algorithmic understanding | HumanEval 164; MBPP 374 |
| Math | GSM8K 1,000; SVAMP 700 |
| Logic | LogicBench 980 total; FOLIO 1,000 |

**Models and results.**

- Evaluates Gemini 2.5 Pro, o1, Claude 3.5 Sonnet, GPT-4o, DeepSeek-v3, GPT-4o-mini, and Llama-3-8B Instruct.
- Reports average CoT accuracy drops for dialectal inputs relative to SAE equivalents.
- Native-speaker evaluation over 120 sampled translations per dialect gives high scores on dialect faithfulness, fluency, formality, and information retention.

**Important findings.**

- EnDive outputs are usually more diverse, fluent, and preferred than Multi-VALUE outputs.
- Models consistently underperform on dialectal inputs compared with SAE, including strong reasoning models.
- Common failure mode: semantic polarity errors in yes/no QA caused by dialectal negation, aspect markers, and tense/aspect constructions.

**Limitations relevant to Vietnamese.**

- LLM-generated dialect data can inherit the biases and inaccuracies of the rewriting model.
- Automatic quality metrics are imperfect for dialects.
- Native-speaker validation is necessary but expensive.
- For Vietnamese, an EnDive-style pipeline should be treated as **candidate generation**, not final gold data.

---

## 4. Methodological Themes Across the Five Papers

### 4.1 Rule-based perturbation is scalable but incomplete

Multi-VALUE’s difference-rule approach is the most reusable engineering idea: define dialect features, write label-preserving transformations, control density, and apply them to arbitrary tasks. However, ReDial and EnDive show that rule-based transformations often miss:

- phrase-level constructions,
- discourse markers,
- pragmatic intent,
- culturally grounded references,
- natural co-occurrence of multiple features,
- orthographic and register variation,
- annotator/community norms.

**Design lesson:** use rules for stress testing and controlled ablations, but not as the only benchmark source.

### 4.2 Parallel meaning-preserving data is central

All five papers rely on some version of meaning equivalence:

- Multi-VALUE preserves labels by construction.
- ReDial preserves reasoning prompts across SE and AAVE.
- EnDive translates SAE to dialectal equivalents and filters/validates them.
- Conversation Robustness compares original and transformed/normalized dialogue sets.
- DialectBench compares varieties via standard metrics and representative varieties.

**Design lesson:** a Vietnamese benchmark should store `(standard Vietnamese input, dialectal input, label/answer, dialect metadata, transformation source, validation status)` for every item.

### 4.3 Normalization is a useful baseline but not a full solution

Several papers use standardization or normalization:

- Conversation Robustness uses GPT-4 to remove dialectal information from en-IN.
- ReDial tests prompting models to standardize AAVE before answering.
- EnDive compares dialectal inputs against SAE equivalents.

These methods can improve performance, but ReDial shows they do not fully close the gap and can impose extra cost or erase speaker identity.

**Design lesson:** include normalization as a baseline system, but evaluate whether it preserves meaning, social tone, and downstream answer correctness.

### 4.4 Reasoning and conversation are more revealing than classification alone

DialectBench includes classic NLP tasks, but later papers focus on reasoning and dialogue. ReDial and EnDive show dialect brittleness in algorithmic, math, logic, and integrated reasoning. Conversation Robustness shows cultural/pragmatic failures.

**Design lesson:** a Vietnamese benchmark should not stop at sentiment or dialect ID. It should include at least:

- QA / reading comprehension,
- sentiment or intent classification,
- math word problems,
- logic/NLI,
- instruction following,
- dialogue/conversation understanding,
- possibly code or structured-output tasks if the target project is LLM-focused.

---

## 5. Recommended Design for a Vietnamese Dialect Robustness Benchmark

### 5.1 Proposed dialect coverage

A practical first version should define explicit coverage rather than claiming to represent all Vietnamese dialects. A suggested initial structure:

- **Standard / reference variety:** formal written Vietnamese, likely close to standardized school/news register.
- **Northern Vietnamese:** one or more sub-varieties if annotator coverage allows.
- **Central Vietnamese:** should be subdivided if possible because Central varieties are internally diverse.
- **Southern Vietnamese:** one or more sub-varieties.
- Optional metadata: speaker region, age, urban/rural background, formality level, and whether the text is written, conversational, social-media-like, or transcribed speech.

This report does **not** verify a complete Vietnamese dialect taxonomy; that should be completed with Vietnamese linguists/native speakers before dataset release.

### 5.2 Data creation pipeline

Recommended hybrid pipeline:

1. **Feature inventory.** Build a Vietnamese dialect-feature inventory covering lexical, morphosyntactic, pragmatic, discourse-particle, pronoun/address-term, and orthographic/spelling variants. Mark each rule with source, dialect, context, and confidence.
2. **Synthetic perturbation.** Implement Multi-VALUE-style transformations for high-confidence meaning-preserving features. Use feature density controls.
3. **LLM candidate rewriting.** Use an EnDive-style few-shot prompt with verified Vietnamese examples to generate candidate dialect rewrites.
4. **Human rewriting.** For a core benchmark, ask native speakers to rewrite standard inputs naturally in their dialect, following ReDial.
5. **Human validation.** Validate dialect naturalness and meaning preservation separately. Use at least two validators when possible.
6. **Normalization baseline.** Create normalized versions of dialectal text back to standard Vietnamese and evaluate both direct answering and normalize-then-answer pipelines.
7. **Audit labels.** Store whether each item is synthetic, LLM-generated, human-written, human-validated, or unvalidated.

### 5.3 Suggested benchmark tasks

A balanced Vietnamese benchmark could start with:

| Task type | Why include it | Possible item source |
|---|---|---|
| Dialect identification | Sanity check; not the main fairness task | Human-written regional sentences |
| Sentiment / intent classification | Common applied NLP task | Existing Vietnamese sentiment/intent datasets if licensed |
| Reading comprehension / QA | Tests semantic preservation | Vietnamese QA datasets or translated curated passages |
| NLI / truth judgment | Sensitive to negation and particles | Manually authored pairs or translated NLI |
| Math word problems | Tests quantity, comparison, and reasoning | Vietnamese-translated GSM8K/SVAMP-style items, with validation |
| Instruction following | Direct LLM user scenario | Short commands and tasks in dialectal style |
| Dialogue target prediction or response selection | Captures culture/pragmatics | Human dialogues or scripted role-play validated by speakers |
| Code/task generation prompts | Optional; tests robustness in technical use | HumanEval/MBPP-style Vietnamese prompts if project targets coding LLMs |

### 5.4 Evaluation metrics

Use both performance and robustness metrics:

- **Task score:** accuracy, F1, exact match, pass@1, BLEU/COMET where appropriate.
- **Dialect gap:** `score_standard - score_dialect`, plus relative gap `(score_standard - score_dialect) / score_standard`.
- **Pairwise consistency:** whether the model gives the same correct answer on standard and dialectal versions.
- **Semantic preservation score:** human rating or validated label consistency.
- **Naturalness/faithfulness score:** native-speaker Likert rating.
- **Normalization utility:** direct dialect score vs normalize-then-answer score.
- **Synthetic-vs-human gap:** difference between rule-based perturbation results and human dialect rewrite results.

### 5.5 Minimum viable benchmark design

If time or budget is limited, build two tiers:

**Tier 1: controlled synthetic stress test**

- 3 dialect regions × 300–500 standard inputs.
- Rule-based perturbations with feature-density levels.
- Tasks: sentiment/intent, QA, math, NLI.
- Purpose: fast ablations and model comparison.

**Tier 2: human-authentic gold set**

- 3 dialect regions × 100–200 items per task.
- Human dialect rewrites and validation.
- Tasks: QA, math, NLI/logic, dialogue/instruction following.
- Purpose: credible fairness claims.

The key paper-based warning is that Tier 1 may underestimate real-world dialect brittleness. Tier 2 is necessary for strong conclusions.

---

## 6. Risks and Open Questions for the Vietnamese Project

1. **Dialect authenticity risk:** LLM-generated Vietnamese dialect may sound stereotyped or unnatural. Native-speaker validation is required.
2. **Meaning preservation risk:** lexical substitutions, pronouns, address terms, and particles may change social relationship or implicature even if literal meaning seems preserved.
3. **Dialect taxonomy risk:** Northern/Central/Southern labels may be too coarse.
4. **Normalization ethics risk:** forcing dialect text into standard Vietnamese can erase identity and may not be equivalent in tone.
5. **Task contamination risk:** translated English benchmarks may not match Vietnamese cultural contexts.
6. **Synthetic over/under-shift risk:** feature density may be unnatural, as noted by Multi-VALUE and later papers.
7. **Evaluation leakage risk:** if public datasets are used, LLMs may have memorized standard items; dialectal rewrites can partially mitigate but not eliminate this.

---

## 7. Recommended Next Steps

1. **Create a Vietnamese dialect feature sheet** with examples, dialect labels, meaning-preservation constraints, and transformation rules.
2. **Select 4–6 task families** for the first release: QA, NLI/logic, math, intent/sentiment, instruction following, and dialogue.
3. **Implement a small Multi-VALUE-style perturbation prototype** for high-confidence Vietnamese features.
4. **Collect a small human rewrite pilot**: e.g., 50 items × 3 dialect regions × 3 tasks.
5. **Compare three data-generation methods** on the same items:
   - rule-based perturbation,
   - LLM few-shot rewrite,
   - native-speaker rewrite.
6. **Run baseline models** on standard vs dialectal versions and compute dialect gaps.
7. **Use the pilot to decide scaling strategy**: which dialects, tasks, and generation methods are reliable enough for the full benchmark.

---

## 8. Sources

- Ziems et al. **“Multi-VALUE: A Framework for Cross-Dialectal English NLP”** (ACL 2023), arXiv:2212.08011. https://arxiv.org/abs/2212.08011 and https://aclanthology.org/2023.acl-long.44.pdf
- Faisal et al. **“DIALECTBENCH: An NLP Benchmark for Dialects, Varieties, and Closely-Related Languages”** (ACL 2024). https://www.aclanthology.org/2024.acl-long.777/ and https://fahimfaisal.info/DialectBench.io/
- Srirag, Sahoo, and Joshi. **“Evaluating Dialect Robustness of Language Models via Conversation Understanding”** (arXiv:2405.05688; SUMEval 2025). https://arxiv.org/abs/2405.05688 and https://github.com/dipankarsrirag/mmd3-dialect-robust
- Lin et al. **“Assessing Dialect Fairness and Robustness of Large Language Models in Reasoning Tasks”** (ACL 2025). https://aclanthology.org/2025.acl-long.317/ and https://redial-demo.netlify.app/
- Gupta et al. **“EnDive: A Cross-Dialect Benchmark for Fairness and Performance in Large Language Models”** (Findings of EMNLP 2025). https://aclanthology.org/2025.findings-emnlp.913/ and https://openreview.net/forum?id=gMrFqtiSzC
