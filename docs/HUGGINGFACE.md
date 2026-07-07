# Hugging Face setup (Space + Dataset + OpenEvals)

Getting the AI Egg Index onto Hugging Face is high-leverage: the retired Open LLM
Leaderboard left a gap, and HF is where people hunt for a benchmark's home. This is a
one-time setup you run on your own HF account; afterward the data can auto-refresh.

**Prereqs**
```bash
pip install -U huggingface_hub
huggingface-cli login    # paste a WRITE token from https://huggingface.co/settings/tokens
```
Replace `USER` below with your HF username.

---

## A. Dataset — publish the results (do this first; easiest, high value)

Aggregators ingest from HF, and a dataset is citable.

1. Create it (web: https://huggingface.co/new-dataset, or CLI):
   ```bash
   huggingface-cli repo create ai-egg-index --type dataset
   git clone https://huggingface.co/datasets/USER/ai-egg-index hf-dataset
   ```
2. Add the results + a dataset card (`hf-dataset/README.md`):
   ```markdown
   ---
   license: cc-by-4.0
   pretty_name: AI Egg Index Results
   tags: [llm, benchmark, leaderboard, evaluation, free-tier]
   task_categories: [text-generation]
   ---
   # AI Egg Index — results

   Weekly benchmark of free-tier LLMs on everyday tasks.
   Leaderboard: https://ceaustin117.github.io/ai-egg-index/ ·
   Code: https://github.com/Ceaustin117/ai-egg-index

   - `latest.json` — current standings
   - `historical.json` — scores over time (+ rolling averages)
   - `run-health.json` — per-run pipeline health

   Small-sample / directional. See the repo's LIMITATIONS.md.
   ```
   ```bash
   cp output/*.json hf-dataset/
   cd hf-dataset && git add . && git commit -m "AI Egg Index results" && git push && cd ..
   ```

---

## B. Space — host the leaderboard (static)

A static Space serves from **root**, so build with `SITE_BASE=/`.

1. Create it (web: https://huggingface.co/new-space → SDK **Static**, or CLI):
   ```bash
   huggingface-cli repo create ai-egg-index --type space --space_sdk static
   git clone https://huggingface.co/spaces/USER/ai-egg-index hf-space
   ```
2. Build the site for root and copy it in:
   ```bash
   SITE_BASE=/ npm --prefix site run build
   cp -r site/dist/* hf-space/
   ```
   > On **Windows Git Bash**, a lone `/` gets mangled — prefix with `MSYS_NO_PATHCONV=1`
   > (`MSYS_NO_PATHCONV=1 SITE_BASE=/ npm --prefix site run build`), or build on Linux/WSL/CI.
3. Add the Space card (`hf-space/README.md`) — this frontmatter drives discovery. Use the
   **exact HF repo IDs** for `models:` so HF cross-links your Space from each model page:
   ```markdown
   ---
   title: AI Egg Index
   emoji: 🥚
   colorFrom: yellow
   colorTo: blue
   sdk: static
   pinned: true
   short_description: The everyday index for AI — a free-tier LLM benchmark
   tags: [leaderboard, benchmark, llm, evaluation]
   models:
     - meta-llama/Llama-3.1-8B-Instruct
     - meta-llama/Meta-Llama-3-8B-Instruct
     - CohereLabs/c4ai-command-r-08-2024
     - google/gemma-2-9b-it   # adjust to the exact models you test
   ---
   ```
   ```bash
   cd hf-space && git add . && git commit -m "AI Egg Index leaderboard" && git push && cd ..
   ```
4. Chase early **likes** to hit the `model-benchmarking` trending sort.

### Keeping the Space fresh
Re-run steps B2–B3's push after each weekly run, **or** automate it: add a GitHub Actions
job (triggered after the benchmark run) that builds with `SITE_BASE=/` and pushes to the
Space using an `HF_TOKEN` repo secret. Ask and I'll write that workflow.

---

## C. List it in OpenEvals "Find a leaderboard"

This is where people now look for the retired Open LLM Leaderboard's replacement.
Go to https://huggingface.co/spaces/OpenEvals/find-a-leaderboard and follow its
submission instructions (usually a PR/form adding your Space). Describe the niche:
**free-tier models, everyday tasks, weekly.**

---

## Also worth doing
- **Zenodo DOI**: connect https://zenodo.org with GitHub, toggle the repo on, cut a
  GitHub Release → auto-minted DOI. Add the DOI badge to the README.
