# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project aims to follow
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

This is the **code** changelog. Changes that affect **score comparability** are tracked
separately in [METHODOLOGY_CHANGELOG.md](METHODOLOGY_CHANGELOG.md).

## [Unreleased]

## [1.0.0] - 2026-07-07

First public release.

### Added
- Four benchmarks: **practical knowledge** and **creative + technical** (custom,
  LLM-as-judge) plus **IFEval** and **GSM8K** (via openbench), on free-tier models.
- Weekly automated run (GitHub Actions) that commits results and refreshes the leaderboard.
- Stronger free-tier judge (`llama-3.3-70b-versatile`) with bounded rate-limit retry.
- OpenBench benchmarks across Groq + Cohere (Google/HuggingFace excluded — see LIMITATIONS).
- Provenance stamping (methodology version + git commit + run params) on every result.
- Result-tracking: `run-health.json` (per-run health) and rolling-average trends.
- Standalone project site (GitHub Pages) with tabbed leaderboard, performance-over-time
  chart, and pipeline-health view.
- Community health files: CITATION.cff, CODE_OF_CONDUCT, SECURITY, issue/PR templates.
- Contamination canary in prompt files; contamination + limitations policies.

[Unreleased]: https://github.com/Ceaustin117/ai-egg-index/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/Ceaustin117/ai-egg-index/releases/tag/v1.0.0
