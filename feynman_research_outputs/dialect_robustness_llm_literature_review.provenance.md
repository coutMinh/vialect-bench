# Provenance: Dialect-Robustness LLM Literature Review

**Created:** 2026-05-14

## Search and retrieval steps

- Web search queries run:
  - `LLM dialect robustness benchmark dialectal variation benchmark language models 2024 2025`
  - `Vietnamese dialect robustness benchmark large language models meaning-preserving variation`
  - `dialectal robustness NLP benchmark African American English Arabic dialects LLMs`
- alpha CLI searches run:
  - `alpha --json search --mode all "dialect robustness benchmark large language models dialectal variation"`
  - `alpha --json search --mode all "Vietnamese dialect robustness large language models benchmark"`
  - `alpha --json search --mode all "meaning preserving dialectal variation NLP benchmark"`
  - `alpha --json search --mode all "A Parallel Corpus for Vietnamese Central-Northern Dialect Text Transfer"`
  - `alpha --json search --mode all "Multi-VALUE cross-dialectal English NLP benchmark"`
- Direct paper reads / content fetches:
  - ACL Anthology pages for DIALECTBENCH, ReDial, EnDive, AraDiCE, Vietnamese Central-Northern dialect corpus, AL-QASIDA.
  - arXiv pages for Multi-VALUE, Evaluating Dialect Robustness via Conversation Understanding, DialectalArabicMMLU, ViDia2Std.
  - alpha CLI `get` for DIALECTBENCH, ReDial, AraDiCE, ViLLM-Eval, Multi-VALUE, NLP dialect survey.

## Tool limitations / blockers

- `alpha ask` failed repeatedly with an MCP input validation error: `Invalid arguments for tool answer_pdf_queries ... queries expected array`. I therefore used `alpha get`, direct ACL/arXiv pages, and web fetches instead.
- Some alpha CLI outputs exceeded the 50KB display limit; full logs were read via local temp files and summarized with Python where needed.

## Verification status

- Claims about dataset sizes and benchmark scope are drawn from paper abstracts/pages or alpha-generated paper reports.
- I did not independently download and inspect all benchmark datasets or code repositories.
- Quantitative model scores are mostly omitted unless present in inspected abstracts/pages; no experimental reproduction was run.
