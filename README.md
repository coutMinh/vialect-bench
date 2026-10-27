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
  <a href="https://github.com/digi-lad"><img src="https://github.com/digi-lad.png?size=64" width="64" height="64" style="border-radius: 50%; border: 2px solid #e1e4e8;" alt="digi-lad" /></a>
</p>

## Rebuttal workspace

The `rebuttal/cleanup` branch shares the finalized benchmark, cached model results,
and paper analysis. See [the rebuttal handoff](docs/rebuttal.md) for the file inventory,
reviewer-question mapping, known gaps, and branch workflow.

```text
data/Finalized/probe_dialects.jsonl       400 cases, 100 per task, 7 variants each
data/vialectbench_finalized_6.json        2,400 annotated dialect rewrites
data/*_100.jsonl                         original selected task samples
configs/                                model, prompt, and sampling configurations
src/                                    sampling, preparation, and model runners
scripts/                                intrinsic and lexical analysis
outputs/direct_prompting/               10 cached models, 2,800 rows each
outputs/finalized_6_complete_six_intrinsic_analysis/
outputs/lexical_substitution_analysis/   includes completed manual review sheets
notebooks/EDA.ipynb                      paper tables and figures from cached outputs
notebooks/probe_dialects_quality_audit.ipynb
notebooks/ViDia2std_Normalizer.ipynb      optional normalization research
paper_artifacts/{dataset_analysis,direct_prompting}/
submission.pdf                          submitted manuscript reference
```

## Analyze existing results

Run commands from the repository root. No GPU, API key, or model download is needed
for cached-result analysis:

```bash
python -m pip install -r requirements-analysis.txt
python scripts/check_rebuttal_bundle.py
python -m nbconvert --to notebook --execute notebooks/EDA.ipynb --output EDA.executed.ipynb --output-dir .local
python -m nbconvert --to notebook --execute notebooks/probe_dialects_quality_audit.ipynb --output audit.executed.ipynb --output-dir .local
```

EDA rewrites the tracked tables and figures under `paper_artifacts/`; review their
diff before committing. It does not regenerate the intrinsic PPL caches or the
separate PPL figures already supplied under `paper_artifacts/dataset_analysis/`.
The audit reports data inconsistencies for inspection; finishing execution does
not mean all quality checks passed.

## Run additional inference only when needed

```bash
python -m pip install -r requirements.txt
python -m src.probe_models --config configs/direct_prompting/qwen_2_5_3b_instruct.yaml --output .local/qwen_rebuttal.jsonl
python -m src.probe_gemma3 --config configs/direct_prompting/gemma_3_4b_it.yaml --output .local/gemma3_rebuttal.jsonl
python -m src.probe_openai --config configs/direct_prompting/gpt_4o.yaml --output .local/gpt4o_rebuttal.jsonl
```

Use a separate output path while checking a new run to preserve the submitted
baseline. Export `HF_TOKEN` or `OPENAI_API_KEY` in the process environment when
needed; `.env.example` lists the names, and runners do not load `.env` automatically.

To rebuild probe cases for comparison without overwriting the shared dataset:

```bash
python -m src.prepare_annotated_probe_cases --output .local/probe_rebuilt.jsonl
```

This command validates source consistency and can stop on existing annotation
issues. Inspect the audit before changing source data or regenerating cases.
