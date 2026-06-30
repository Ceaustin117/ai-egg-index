## What does this PR do?

<!-- One or two sentences. Link any related issue: "Closes #123". -->

## Type of change

- [ ] New model / provider
- [ ] New benchmark / task
- [ ] Bug fix
- [ ] Methodology change (⚠️ requires a methodology version bump — see CONTRIBUTING)
- [ ] Docs / tooling

## Checklist

- [ ] I ran the relevant benchmark locally (paste the exact command below)
- [ ] I ran `python aggregate.py` and the `output/` files are updated/valid
- [ ] For score-affecting changes: I bumped the methodology version and noted it in `METHODOLOGY_CHANGELOG.md`
- [ ] CI (Ruff + pytest) passes
- [ ] I did **not** add paid-tier models or paid-only features (free-tier only)

## Reproduction command

```bash
# e.g. python benchmarks/practical-knowledge/eval.py --provider groq --model llama-3.1-8b-instant --limit 3
```

## Score deltas (if applicable)

<!-- Before/after scores, or "n/a". Note this is a small-sample, directional benchmark. -->

> Note: a PR does not guarantee a merge. Methodology and scope changes are reviewed
> against the project's goals (free-tier, everyday-task, reproducible).
