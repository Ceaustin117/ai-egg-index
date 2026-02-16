# AI Egg Index

Benchmarking free-tier LLMs on tasks regular people actually care about.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

**Live leaderboard:** [chrisaustin.dev/benchmarks](https://chrisaustin.dev/benchmarks)

---

## What Is This?

Most AI benchmarks are like complex economic models — abstract scores that don't mean much to regular people. The AI Egg Index is more like the egg price index: a simple, grounded measure of what actually matters.

Can free AI help with your taxes? Can it handle a complex reasoning task that someone might actually ask? Can it follow your instructions? That's what we measure.

**We only test what you can actually use without paying** — including rate limits and context constraints.

---

## Quick Start

```bash
# Clone the repo
git clone https://github.com/Ceaustin117/ai-egg-index.git
cd ai-egg-index

# Install dependencies
pip install -r requirements.txt

# Set your API key
export GROQ_API_KEY=your_key

# Run practical knowledge benchmark
python benchmarks/practical-knowledge/eval.py --model llama-3.1-8b-instant --limit 3

# Run creative+technical benchmark
python benchmarks/creative-technical/eval.py --model llama-3.1-8b-instant --limit 2

# Aggregate results for the leaderboard
python aggregate.py
```

For the standard benchmarks (HumanEval, IFEval, GSM8K), install [openbench](https://github.com/groq/openbench):

```bash
pip install openbench
bench eval humaneval --model groq/llama-3.1-8b-instant --limit 10
bench eval ifeval --model groq/llama-3.1-8b-instant --limit 10
bench eval gsm8k --model groq/llama-3.1-8b-instant --limit 10
```

---

## The 5 Benchmarks

| Benchmark | Type | What It Tests |
|-----------|------|---------------|
| **Practical Knowledge** | Custom | Real-world info people actually search for: tax brackets, retirement limits, minimum wage, consumer rights |
| **HumanEval** | [openbench](https://github.com/groq/openbench) | Simple coding help — "Write a function that..." |
| **IFEval** | [openbench](https://github.com/groq/openbench) | Instruction following — "Format as a table," "respond in exactly 3 bullet points" |
| **GSM8K** | [openbench](https://github.com/groq/openbench) | Grade school math word problems — if it can't help your kid with homework, is it useful? |
| **Creative+Technical** | Custom | Combining creativity with code — "Write a budget tracker that outputs summaries as haiku" |

**Custom** benchmarks are ours, with LLM-as-judge scoring and custom rubrics.
**openbench** benchmarks use Groq's [openbench](https://github.com/groq/openbench) framework (Inspect AI-based, provider-agnostic evaluation).

---

## Free-Tier Providers Tested

| Provider | Rate Limit |
|----------|------------|
| Groq | 30 req/min |
| Together | 60 req/min |
| Google | 60 req/min |
| Cohere | 100 req/min |
| HuggingFace | Variable |

---

## Project Structure

```
ai-egg-index/
├── benchmarks/
│   ├── practical-knowledge/   # Custom: tax, finance, consumer rights questions
│   │   ├── questions.json     # Question bank with expected topics & sources
│   │   └── eval.py            # LLM-as-judge evaluator
│   └── creative-technical/    # Custom: code + creativity challenges
│       ├── prompts.json       # Challenge prompts with scoring rubrics
│       └── eval.py            # Code execution + LLM-as-judge evaluator
├── results/                   # Raw per-model results by date
│   ├── 2026-01-15/
│   ├── 2026-02-04/
│   └── latest.json
├── output/                    # Aggregated data (consumed by the website)
│   ├── latest.json
│   └── historical.json
├── aggregate.py               # Builds latest.json + historical.json from results
├── requirements.txt
├── CONTRIBUTING.md
├── LICENSE
└── README.md
```

---

## Methodology

- **Free tier only.** If you have to pay, it's not included.
- **LLM-as-judge** for custom benchmarks, with rubrics for factual accuracy, completeness, and recency awareness.
- **Code execution** for creative+technical tasks — code is extracted and run in a sandbox.
- **Historical tracking.** Benchmarks run regularly, scores tracked over time.
- **Overall score** is the average across all 5 benchmarks (only counting benchmarks with data).

---

## Open Questions

This isn't a solved problem — it's the reason this project exists:

1. **How do we properly evaluate AI responses?** LLM-as-judge has known biases. When does it break down? How do we score tasks with multiple valid answers?

2. **Are we even asking the right questions?** A benchmark is only as good as its test set. What do regular people *actually* ask AI for? How do we keep questions current as laws change?

---

## How We Compare

| Benchmark | Focus | Free Tier? | Practical Tasks? |
|-----------|-------|------------|------------------|
| LMSYS Chatbot Arena | Human preference voting | All models | Vibes, not tasks |
| ACE (AI Consumer Index) | Consumer activities | Frontier models | Yes |
| Scale Leaderboard | Enterprise tool use | Enterprise | Dev/Agent tasks |
| OpenAI Evals | Academic benchmarks | All models | Academic |
| **AI Egg Index** | **Everyday layperson tasks** | **Free only** | **Yes** |

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for how to:
- Add a practical knowledge question
- Add a new free-tier provider
- Add a creative+technical task
- Run benchmarks locally

---

## License

MIT
