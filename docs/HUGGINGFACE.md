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

## C. Getting into OpenEvals "Find a leaderboard" (the real mechanism)

`find-a-leaderboard` does **not** take a form or a Space submission. It indexes
**registered benchmark _datasets_** (`benchmark:official`). To become one
(per https://huggingface.co/docs/hub/eval-results):

1. Add an **`eval.yaml`** to the dataset repo root with `name`, `description`,
   `evaluation_framework` (one value from HF's maintained enum — e.g. `inspect-ai`,
   `math-arena`, `lighteval`), and `tasks[]`. Validated at push time.
2. Submit per-model scores as **`.eval_results/*.yaml` PRs into each model's repo**
   (referencing `dataset.id` + `task_id`).
3. **Ask the HF team to add it to the allow-list** (beta — this step is manual).

⚠️ **Blocker for us:** the AI Egg Index uses a custom LLM-as-judge pipeline, which is
**not** one of HF's enumerated `evaluation_framework` values, and our dataset is
results-JSON (not an HF questions-dataset with `field_spec`/`solvers`/`scores`). So
official registration would require either adding a new framework to HF's enum
(a reviewed PR to huggingface.js) or restructuring the benchmark to fit `inspect-ai`.

**Recommendation:** treat official registration as a later project. For now the Space
is the HF presence (discoverable via HF search + the `leaderboard`/`benchmark` tags). If
you want to pursue official listing, the sanctioned first move is to **contact the HF
team** (the allow-list step says "get in touch") and ask how a custom-framework
benchmark can be represented.

---

## Also worth doing
- **Zenodo DOI**: connect https://zenodo.org with GitHub, toggle the repo on, cut a
  GitHub Release → auto-minted DOI. Add the DOI badge to the README.
