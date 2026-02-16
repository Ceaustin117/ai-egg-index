# Contributing to AI Egg Index

Thanks for your interest in contributing! This guide explains exactly how to add questions, providers, and tasks.

---

## Adding a Practical Knowledge Question

Questions live in `benchmarks/practical-knowledge/questions.json`.

**1.** Pick a category: `taxes`, `regulations`, `practical_finance`, or `consumer_rights`.

**2.** Add an entry to that category's array:

```json
{
  "id": "short-descriptive-id",
  "question": "The question a regular person would actually ask",
  "expected_topics": ["topic1", "topic2", "topic3"],
  "verification_source": "Official source (e.g., IRS.gov, DOL.gov)",
  "difficulty": "basic or intermediate"
}
```

**3.** Guidelines:
- Questions should be things real people search for, not trivia
- Include at least 2-3 expected topics the answer should cover
- Verification source must be an official/authoritative site
- Prefer questions with objectively verifiable answers

**4.** Run the eval to test your question:

```bash
export GROQ_API_KEY=your_key
python benchmarks/practical-knowledge/eval.py --model llama-3.1-8b-instant --limit 1
```

---

## Adding a Creative+Technical Task

Tasks live in `benchmarks/creative-technical/prompts.json`.

**1.** Add an entry to the `prompts` array:

```json
{
  "id": "short-descriptive-id",
  "prompt": "The full prompt given to the model",
  "scoring": {
    "code_works": {
      "weight": 0.4,
      "description": "What 'code works' means for this task"
    },
    "meets_requirements": {
      "weight": 0.3,
      "description": "What the requirements are"
    },
    "creative_output": {
      "weight": 0.3,
      "description": "What counts as creative success"
    }
  },
  "difficulty": "easy, intermediate, or hard"
}
```

**2.** Guidelines:
- Tasks should require both working code AND creative constraint
- The prompt should ask for complete, runnable code
- Scoring descriptions should be specific enough for an LLM judge to evaluate
- Test that at least one model can partially succeed (avoid impossible tasks)

---

## Adding a New Free-Tier Provider

**1.** Verify the provider has a free tier with reasonable rate limits.

**2.** The custom eval scripts currently use the Groq SDK. To add a new provider, you'll need to update the `_call_llm` method in both eval scripts to support the new provider's API.

**3.** Run benchmarks against the new provider and save results to `results/YYYY-MM-DD/`.

**4.** Update the "Free-Tier Providers Tested" table in `README.md`.

For standard benchmarks (HumanEval, IFEval, GSM8K), openbench already supports 30+ providers — just set the API key and use the appropriate model string:

```bash
export TOGETHER_API_KEY=your_key
bench eval humaneval --model together/mistral-7b --limit 10
```

---

## Running Benchmarks Locally

### Prerequisites

- Python 3.10+
- A free API key from any supported provider (Groq is easiest: [console.groq.com](https://console.groq.com))

### Custom benchmarks

```bash
pip install -r requirements.txt
export GROQ_API_KEY=your_key

# Practical knowledge (all questions)
python benchmarks/practical-knowledge/eval.py --model llama-3.1-8b-instant

# Creative+technical (all tasks)
python benchmarks/creative-technical/eval.py --model llama-3.1-8b-instant

# With limits for quick testing
python benchmarks/practical-knowledge/eval.py --model llama-3.1-8b-instant --limit 2
```

### Standard benchmarks (via openbench)

```bash
pip install openbench
bench eval humaneval --model groq/llama-3.1-8b-instant --limit 10
bench eval ifeval --model groq/llama-3.1-8b-instant --limit 10
bench eval gsm8k --model groq/llama-3.1-8b-instant --limit 10
```

### Aggregating results

After running benchmarks, aggregate into the format the website consumes:

```bash
python aggregate.py
```

This reads from `results/` and writes `latest.json` and `historical.json`.

---

## Submitting Your Contribution

1. Fork the repo
2. Create a branch: `git checkout -b add-my-contribution`
3. Make your changes
4. Test locally (run the relevant eval script)
5. Open a pull request with a brief description of what you added and why
