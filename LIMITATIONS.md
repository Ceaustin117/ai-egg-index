# Limitations & Known Issues

The AI Egg Index is a **rough, directional signal** — like an egg-price index, not a
precision instrument. Read it with these caveats.

## Small samples → directional, not precise

Each benchmark runs a small number of samples per weekly cycle (default 3). At that size,
scores swing run-to-run from sampling noise alone. **Do not over-read small gaps** — a
few points between models is likely noise, not a real difference. We don't yet report
formal confidence intervals; until we do, treat close scores as ties. (Reference for
where we're headed: Miller, *Adding Error Bars to Evals*, arXiv:2411.00640.)

## The cross-model "overall" isn't perfectly apples-to-apples

The overall score averages whichever benchmarks a model actually ran. Because the
OpenBench benchmarks (IFEval, GSM8K) only run on some providers (see below), a model's
overall may average 4 benchmarks or 2. Compare per-benchmark columns for a fairer read.

## OpenBench benchmarks run on Groq + Cohere only

IFEval and GSM8K don't run for Google (free-tier rate limits cause the eval harness to
time out) or HuggingFace (its eval path runs the model locally, which isn't viable in CI).
Those cells show `—` (not run), distinct from `N/A` (attempted but errored that run).

## LLM-as-judge

The custom benchmarks are graded by an LLM judge (a stronger free model,
`llama-3.3-70b-versatile`, than the models under test). LLM judges have known biases
(verbosity, position, self-preference). We mitigate by using a stronger, non-competing
judge and publishing the judge prompts, but it is not a substitute for human grading.
Where possible we prefer deterministic scoring (e.g. exact-match / unit-tested evals).

## Free-tier version drift

Free-tier model IDs (e.g. "gemini-2.5-flash") can change behind the scenes over time. We
pin exact model IDs and stamp the run date on every result so trends stay interpretable,
but the underlying model may still shift under a stable name.

## Contamination

The custom prompt set is fixed and public, so contamination risk grows over time. See
[CONTAMINATION.md](CONTAMINATION.md).

## Scope

This measures **relative performance on everyday tasks for free-tier models** — not
general intelligence, not paid/frontier models, not a definitive ranking. It complements
academic benchmarks (MMLU, LiveBench, Chatbot Arena); it doesn't replace them.
