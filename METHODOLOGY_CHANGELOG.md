# Methodology Changelog

Tracks changes to the benchmark **methodology** that affect score comparability —
separate from the code's version/git tags. The current methodology version lives in
[`benchmarks/provenance.py`](./benchmarks/provenance.py) (`METHODOLOGY_VERSION`) and is
stamped into every result file and the aggregated `output/` data.

**Rule:** bump the version whenever a change makes new scores **non-comparable** to prior
runs — new or changed prompts, a scoring-rubric change, a different judge model, or a
change in sample selection. Never silently re-score under an existing version.

## 1.0 — 2026-07-02

First versioned methodology. Baseline as of this date:

- **Benchmarks (4):** `practical_knowledge` + `creative_technical` (custom, LLM-as-judge)
  and `ifeval` + `gsm8k` (via openbench). Openbench benchmarks run on Groq + Cohere;
  custom benchmarks run on all free-tier providers.
- **Judge:** Groq `llama-3.3-70b-versatile` (stronger than the models under test).
- **Sampling:** small (default 3 samples/benchmark) — scores are directional, not precise.
- Result files and aggregated output now record `methodology_version`, `git_commit`, and
  run params.
