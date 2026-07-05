# Dialectal Robustness of LLMs under Meaning-Preserving Vietnamese Variation

This repository contains the ongoing research for the robustness of Large Language Models (LLMs) when prompted with meaning-preserving Vietnamese dialectal variations. 

Authors: Minh Tran, Duc Hoang, Cuong Dang Cao, Luan Thanh Nguyen, Trinh Chau, Hai Nam

<p align="left">
  <a href="https://github.com/coutMinh"><img src="https://github.com/coutMinh.png?size=64" width="64" height="64" style="border-radius: 50%; border: 2px solid #e1e4e8;" alt="coutMinh" /></a>
  <a href="https://github.com/Duchstf"><img src="https://github.com/Duchstf.png?size=64" width="64" height="64" style="border-radius: 50%; border: 2px solid #e1e4e8;" alt="Duchstf" /></a>
  <a href="https://github.com/CaptainCuong"><img src="https://github.com/CaptainCuong.png?size=64" width="64" height="64" style="border-radius: 50%; border: 2px solid #e1e4e8;" alt="CaptainCuong" /></a>
  <a href="https://github.com/tarudesu"><img src="https://github.com/tarudesu.png?size=64" width="64" height="64" style="border-radius: 50%; border: 2px solid #e1e4e8;" alt="tarudesu" /></a>
  <a href="https://github.com/suzhentxt"><img src="https://github.com/suzhentxt.png?size=64" width="64" height="64" style="border-radius: 50%; border: 2px solid #e1e4e8;" alt="suzhentxt" /></a>
  <a href="https://github.com/haizznaam"><img src="https://github.com/haizznaam.png?size=64" width="64" height="64" style="border-radius: 50%; border: 2px solid #e1e4e8;" alt="haizznaam" /></a>
</p>

## Setup

Install the Python dependencies:

```bash
pip install -r requirements.txt
```

## Sampling

Select label-balanced samples with dialect lexicon coverage:

```bash
python -m src.select_samples \
  --tasks sentiment nli qa mcqa \
  --sample-size 100 \
  --lexicon data/DIALECT_LEXICON_v2.xlsx \
  --output-dir data
```

This writes task files such as `data/sentiment_100.jsonl`, `data/nli_100.jsonl`, `data/qa_100.jsonl`, `data/mcqa_100.jsonl`, and `data/selection_summary.csv`.

For a quick smoke test:

```bash
python -m src.select_samples \
  --tasks sentiment nli qa mcqa \
  --sample-size 10 \
  --max-source-rows 500 \
  --output-dir data/smoke
```

## Running Probes

The probing script reads YAML configs and runs one model with one prompt setting:

```bash
python -m src.probe_models --config configs/probe.yaml
```

The active prompt configs are organized by folder:

```text
configs/direct_prompting/
```

Run direct prompting:

```bash
python -m src.probe_models --config configs/direct_prompting/qwen_2_5_3b_instruct.yaml
```

The benchmark now focuses on two inference settings: direct dialect input and
direct prompting after a ViDia2Std-style normalizer rewrites each dialect variant.
The normalized setting should use `prompt_strategy: normalized_direct` with a
normalized case file once that file is produced.

Each config writes results to a separate JSONL file under `outputs/<prompt_setting>/`. The output rows include `prompt_strategy`, `variant`, `dialect_group`, `gold`, `prediction`, and `raw_output`.
For MCQA, NLI, and sentiment, probing also stores fixed-candidate confidence
signals by default: `label_candidates`, `label_logprobs`, `label_probs`,
`confidence`, `prob_prediction`, and `gold_prob`. Use
`--no-collect-probabilities` to disable this extra scoring pass.

To run another model, choose the matching config file in the same prompt-setting folder. Some Hugging Face models may require login or license acceptance before they can be downloaded.
