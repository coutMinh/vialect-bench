# Literature Review: Dialect-Robustness Benchmarks for Large Language Models

**Date:** 2026-05-14  
**Scope:** Text and speech benchmarks that evaluate whether LLMs or NLP systems preserve performance under within-language dialectal variation. I emphasize work useful for designing a meaning-preserving Vietnamese dialect-robustness benchmark.

## Executive summary

Dialect robustness is now being treated as a fairness and reliability problem, not merely a dialect-identification problem. The central benchmark design pattern is: create semantically equivalent standard/dialectal inputs, evaluate the same task under each variety, and report the performance gap. Recent benchmarks differ mainly in how dialectal variants are produced: rule-based transformations, human rewriting/translation, MT plus human post-editing, or naturally occurring dialectal data.

The strongest evidence so far shows consistent performance drops on non-standard dialectal inputs, including African American Vernacular English (AAVE), Indian English, multiple English dialects, Arabic dialects, and large multilingual dialect collections. For Vietnamese, the literature is thinner: existing work provides dialect-transfer corpora and speech dialect resources, plus general Vietnamese LLM benchmarks, but not yet a mature LLM benchmark that directly tests meaning-preserving regional Vietnamese text variation across downstream reasoning or instruction-following tasks.

## Core concepts and evaluation framing

**Dialect robustness** means that a model's task performance should remain stable when the input changes dialect while preserving task-relevant meaning. Most papers operationalize this as a standard-vs-dialect performance gap:

\[
\Delta_{dialect}=Score(x_{standard})-Score(x_{dialect})
\]

or a relative version normalized by standard-variety performance. Robust benchmark design therefore requires three checks:

1. **Semantic invariance:** the dialectal item must preserve the answer, label, or intended instruction.
2. **Dialect authenticity:** the dialectal wording must be natural to target speakers, not just noisy paraphrase.
3. **Task sensitivity:** the task must be difficult enough that dialectal variation can reveal failure, but not so hard that all models fail regardless of dialect.

## Major benchmark families

| Benchmark / paper | Language varieties | Task types | Construction method | Main contribution | Key caveat |
|---|---:|---|---|---|---|
| **Multi-VALUE** (Ziems et al., 2023; arXiv:2212.08011) | 50 English dialects | QA, MT, semantic parsing | Rule-based transformations grounded in 189 linguistic features (The paper does not randomly perturb text. It uses linguistic features from eWAVE, the Electronic World Atlas of Varieties of English); Pipeline: Start with SAE text -> Select a target dialect, e.g. Indian English -> Apply dialect-specific linguistic rules -> Create synthetic dialectal benchmarks -> Evaluate models on these benchmarks -> Use the same transformations for data augmentation -> Check whether training on dialectal synthetic data improves robustness.| Scalable stress testing and augmentation for English dialect invariance | Synthetic rules can miss pragmatic and phrase-level naturalness |
| **DIALECTBENCH** (Faisal et al., 2024; arXiv:2403.11009; ACL 2024) | 281 varieties across 40 clusters | 10 NLP tasks: parsing, POS, NER, dialect ID, sentiment, topic, NLI, MRC, QA, MT | Aggregates existing datasets; includes translate-test NLI | Broadest multilingual/multivarietal benchmark; quantifies standard vs non-standard gaps | Mostly NLP models/tasks; LLM in-context evaluation is limited relative to current LLM usage |
| **Evaluating Dialect Robustness via Conversation Understanding** (Srirag et al., 2024; arXiv:2405.05688) | US English, Indian English | Target-word prediction and selection in taboo-game conversations | Extends MD3 into masked M-MD3 (use https://arxiv.org/abs/2212.08011 as augumentation methods); creates dialect-added and dialect-removed subsets | Tests discourse-level conversation understanding under dialect variation | Focused on English national varieties and a specialized game-dialogue task |
| **ReDial / Assessing Dialect Fairness and Robustness in Reasoning Tasks** (Lin et al., 2025; ACL 2025) | Standardized English and AAVE | Algorithm, math, logic, integrated reasoning | Human AAVE rewrites of 7 reasoning benchmarks; 1.2K+ parallel pairs | High-quality human benchmark showing broad LLM brittleness on AAVE reasoning | One dialect; costly annotation process |
| **EnDive** (Gupta et al., 2025; Findings EMNLP 2025) | Standard American English + 5 underrepresented English dialects | Language understanding, algorithmic reasoning, math, logic | Few-shot LLM translation with native-speaker examples; human quality checks; filters near-identical translations | Compares LLM-based dialect translation to rule-based approaches and finds consistent dialect performance drops | LLM-generated variants may inherit model biases despite human evaluation |
| **AraDiCE** (Mousi et al., 2025; COLING 2025) | MSA, Gulf, Egyptian, Levantine | Dialect ID, generation/translation, reasoning, reading, knowledge, culture | MT plus human post-editing; ≈45K post-edited samples | Adds cultural evaluation and shows Arabic-centric models outperform multilingual models | Synthetic translation pipeline; dialect coverage still selective |
| **AL-QASIDA** (2025) | 8 dialectal Arabic varieties | Fidelity, understanding, quality, diglossia | Evaluation framework across 9 LLMs | Separates understanding from generation; finds LLMs are often reluctant to generate dialectal Arabic | Abstract-level evidence inspected; full methodology not deeply audited here |
| **DialectalArabicMMLU** (2025/2026; arXiv:2510.27543) | Syrian, Egyptian, Emirati, Saudi, Moroccan + MSA/English | 32-domain multiple-choice MMLU-Redux | Manual translation/adaptation of 3K QA pairs into 5 dialects | Human-curated MMLU-style dialect benchmark; evaluates 19 open-weight LLMs | Arabic-specific; later arXiv version current as of 2026 |

## Findings across the literature

### 1. Dialect gaps are consistent across models and tasks

The repeated observation is that models perform better on standard/prestige varieties than on non-standard varieties. DIALECTBENCH reports broad disparities across 281 varieties. ReDial reports significant AAVE drops across algorithmic, math, logic, and integrated reasoning tasks for GPT, Claude, Llama, Mistral, and Phi families. EnDive likewise reports consistent underperformance on dialectal English inputs compared with Standard American English.

**Inference:** dialect robustness is not solved by general multilingual pretraining or model scaling. It should be measured explicitly in any benchmark claiming fairness or deployment readiness.

### 2. Human-authored dialectal data is more reliable than purely synthetic perturbation

Multi-VALUE is important because it gives controlled, scalable rule-based perturbations, but later work emphasizes that real dialect use involves pragmatic markers, phrase-level constructions, and conversational norms. ReDial's human AAVE rewrites and EnDive's native-speaker validation reflect this shift. The ReDial study explicitly argues that rule-based or LLM-translated data can miss subtle context.

**Benchmark implication:** for Vietnamese, purely lexical replacement rules are likely insufficient. Central and Southern dialect variation includes vocabulary, particles, tone/phonology reflected in spelling, idioms, and register. A credible benchmark should combine rule coverage with native-speaker validation.

### 3. Reasoning benchmarks reveal harms beyond sentiment or toxicity bias

Early dialect-fairness work often focused on toxicity, hate speech, sentiment, or dialect identification. ReDial and EnDive move into reasoning tasks: code, math, logic, and integrated planning. These are higher-stakes because they affect educational support, workplace productivity, and access to AI problem-solving tools.

**Benchmark implication:** a Vietnamese benchmark should not stop at dialect identification or normalization. It should include tasks where the answer is invariant under dialect shift: multiple-choice QA, factual QA, math word problems, instruction following, and perhaps tool-use-style tasks if labels are available.

### 4. Dialect understanding and dialect generation are different capabilities

AraDiCE and AL-QASIDA both indicate that models may understand dialectal Arabic better than they generate it, and may default to Modern Standard Arabic because of post-training preferences. This is relevant for chat LLMs, where users may ask in dialect and expect either dialectal or standard responses.

**Benchmark implication:** separate Vietnamese evaluation into at least two tracks:

- **Comprehension robustness:** answer should be the same for standard and dialectal prompts.
- **Generation fidelity:** response should preserve requested dialect/register when the task asks for dialectal output.

### 5. Normalization can help, but can also hide unfairness

Vietnamese and Arabic work often uses dialect-to-standard transfer/normalization. The Vietnamese Central-Northern corpus and ViDia2Std show normalization can improve downstream processing. However, ReDial notes that asking LLMs to standardize dialect before answering can add token cost and may still not eliminate the gap.

**Benchmark implication:** evaluate both direct dialect input and a normalization pipeline. Report whether normalization helps, but do not treat forced standardization as the only acceptable user experience.

## Vietnamese-specific literature

### A Parallel Corpus for Vietnamese Central-Northern Dialect Text Transfer (Le & Luu, 2023; Findings EMNLP)

This paper introduces a parallel corpus for Central-to-Northern Vietnamese dialect text transfer. The motivation is that Northern Vietnamese functions as the prestige/standard variety, while Central Vietnamese contains more unique vocabulary and lower mutual intelligibility. The authors report that monolingual Vietnamese language models outperform multilingual models on dialect transfer, and that fine-tuned transfer models can improve downstream translation and text-image retrieval on Central dialect inputs.

**Relevance:** this is a direct precedent for Vietnamese meaning-preserving dialect transformation, but it focuses on Central-Northern transfer rather than a broad LLM robustness benchmark.

### Multi-Dialect Vietnamese: Task, Dataset, Baseline Models and Challenges (Nguyen et al., 2024; arXiv:2410.03458)

This work introduces ViMD, a speech dataset covering all 63 Vietnamese provincial dialects: 102.56 hours, about 19,000 utterances, and transcripts totaling more than 1.2M words. It benchmarks dialect identification and speech recognition using pretrained speech models. The paper's key contribution is fine-grained provincial dialect coverage rather than text-only LLM evaluation.

**Relevance:** valuable for speech/accent robustness and possible ASR-front-end evaluation, but not a text LLM benchmark by itself.

### ViLLM-Eval (Nguyen et al., 2024; arXiv:2404.11086)

ViLLM-Eval is a Vietnamese LLM evaluation suite with LAMBADA-style next-word prediction, exam questions, general knowledge, and comprehension QA. It evaluates Vietnamese and multilingual LLMs such as ChatGPT, Vistral, VinaLLaMA, PhoGPT, SeaLLM, and Dama.

**Relevance:** provides task formats and Vietnamese content domains that could be dialectalized. However, it does not primarily evaluate within-Vietnamese dialect robustness.

### ViDia2Std (Ta et al., 2026; arXiv:2603.10211)

ViDia2Std is a manually annotated dialect-to-standard Vietnamese corpus covering all 63 provinces. It contains over 13,000 sentence pairs sourced from Facebook comments and annotated by native speakers across North, Central, and South. The paper reports semantic mapping agreement rates of 86% North, 82% Central, and 85% South, and benchmarks seq2seq models; mBART-large-50 is reported as best among the tested models. The abstract states that dialect normalization substantially improves downstream tasks.

**Relevance:** closest current Vietnamese resource for broad text dialect normalization. It could support benchmark construction, but normalization corpora alone do not test LLM answer robustness unless paired with task labels and invariant answers.

## Methodological lessons for a Vietnamese dialect-robustness benchmark

A strong Vietnamese benchmark should adopt the following design:

1. **Parallel standard/dialectal items.** Each item should have a standard Vietnamese prompt and one or more meaning-preserving dialectal variants.
2. **Native-speaker validation.** Use annotators from Northern, Central, and Southern regions, ideally with province-level metadata for dialect labels.
3. **Answer invariance tests.** For QA/math/reasoning, verify that the gold answer is unchanged after dialect rewriting.
4. **Multiple transformation sources.** Compare: human rewrites, rule-based lexical/particle transformations, LLM-generated variants, and real-world dialectal comments normalized to standard.
5. **Report both absolute and relative gaps.** Include per-region and per-feature gaps, not only aggregate accuracy.
6. **Separate comprehension from generation.** Test whether the model understands dialectal prompts and whether it can produce dialectal Vietnamese when requested.
7. **Include a normalization baseline.** Evaluate direct LLM answering versus dialect-to-standard normalization followed by answering.
8. **Control for confounds.** Track prompt length, spelling noise, topic/domain, named entities, code-switching, and formality.

## Open gaps

- **Vietnamese lacks a ReDial-like LLM reasoning benchmark** with human-written dialectal versions of math, logic, and instruction-following tasks.
- **Dialect authenticity remains hard to scale.** LLM-generated dialect may be fluent-looking but stereotyped or regionally inconsistent.
- **Few benchmarks evaluate cost and user burden.** Forced standardization may reduce accuracy gaps but imposes unequal effort or token cost on dialect speakers.
- **Most text benchmarks underrepresent speech-to-text interaction.** Vietnamese dialect robustness should consider ASR errors from regional accents if deployed in voice systems.
- **Cultural and pragmatic meaning are under-tested.** AraDiCE shows culture matters; Vietnamese regional idioms and discourse particles may similarly affect LLM responses.

## Recommended reading order

1. **DIALECTBENCH** for the broad benchmark taxonomy and gap metric.
2. **Multi-VALUE** for controllable dialect perturbation and augmentation.
3. **ReDial** for high-quality human parallel reasoning evaluation.
4. **EnDive** for LLM-assisted dialect translation with human validation.
5. **AraDiCE / AL-QASIDA / DialectalArabicMMLU** for non-English LLM dialect evaluation and generation-vs-understanding distinctions.
6. **Vietnamese Central-Northern corpus, ViMD, ViLLM-Eval, and ViDia2Std** for Vietnamese resources that can be adapted into an LLM robustness benchmark.

## Sources

- Ziems et al. **Multi-VALUE: A Framework for Cross-Dialectal English NLP**. 2023. arXiv:2212.08011. https://arxiv.org/abs/2212.08011
- Faisal et al. **DIALECTBENCH: An NLP Benchmark for Dialects, Varieties, and Closely-Related Languages**. ACL 2024. arXiv:2403.11009. https://aclanthology.org/2024.acl-long.777/
- Srirag et al. **Evaluating Dialect Robustness of Language Models via Conversation Understanding**. 2024. arXiv:2405.05688. https://arxiv.org/abs/2405.05688
- Lin et al. **Assessing Dialect Fairness and Robustness of Large Language Models in Reasoning Tasks**. ACL 2025. https://aclanthology.org/2025.acl-long.317/
- Gupta et al. **EnDive: A Cross-Dialect Benchmark for Fairness and Performance in Large Language Models**. Findings EMNLP 2025. https://aclanthology.org/2025.findings-emnlp.913/
- Mousi et al. **AraDiCE: Benchmarks for Dialectal and Cultural Capabilities in LLMs**. COLING 2025. https://aclanthology.org/2025.coling-main.283/
- **AL-QASIDA: Analyzing LLM Quality and Accuracy Systematically in Dialectal Arabic**. Findings ACL 2025. https://aclanthology.org/2025.findings-acl.1137/
- **DialectalArabicMMLU: Benchmarking Dialectal Capabilities in Arabic and Multilingual Language Models**. 2025/2026. arXiv:2510.27543. https://arxiv.org/abs/2510.27543
- Le & Luu. **A Parallel Corpus for Vietnamese Central-Northern Dialect Text Transfer**. Findings EMNLP 2023. https://aclanthology.org/2023.findings-emnlp.925/
- Nguyen et al. **Multi-Dialect Vietnamese: Task, Dataset, Baseline Models and Challenges**. 2024. arXiv:2410.03458. https://arxiv.org/abs/2410.03458
- Nguyen et al. **ViLLM-Eval: A Comprehensive Evaluation Suite for Vietnamese Large Language Models**. 2024. arXiv:2404.11086. https://arxiv.org/abs/2404.11086
- Ta et al. **ViDia2Std: A Parallel Corpus and Methods for Low-Resource Vietnamese Dialect Normalization**. 2026. arXiv:2603.10211. https://arxiv.org/abs/2603.10211
- Joshi et al. **Natural Language Processing for Dialects of a Language: A Survey**. 2024. arXiv:2401.05632. https://arxiv.org/abs/2401.05632
