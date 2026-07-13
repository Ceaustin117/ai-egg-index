#!/usr/bin/env python3
"""Inject the current leaderboard into README.md between the RESULTS markers.

Reads output/latest.json and rewrites the table so the README's headline results stay
fresh (and crawlable). Run in the weekly workflow after aggregate.py.
"""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LATEST = ROOT / "output" / "latest.json"
README = ROOT / "README.md"
START = "<!-- RESULTS:START -->"
END = "<!-- RESULTS:END -->"

BENCHES = [
    ("practical_knowledge", "Practical"),
    ("ifeval", "IFEval"),
    ("gsm8k", "GSM8K"),
    ("creative_technical", "Creative"),
]


def cell(b: dict | None) -> str:
    if b is None:
        return "—"
    if b.get("status") == "error" or b.get("score") is None:
        return "N/A"
    return f"{round(b['score'] * 100)}%"


# Overall ranks on the two custom benchmarks EVERY provider runs (apples-to-apples).
# IFEval/GSM8K are shown as columns but excluded from the ranking, because they only run
# on Groq/Cohere — folding them in would let a model's overall average a different
# (smaller) benchmark set and rank unfairly.
OVERALL_KEYS = ("practical_knowledge", "creative_technical")


def overall(benchmarks: dict):
    """Mean of the custom benchmarks, but only if the model has ALL of them (a valid
    score for both). Missing/errored → None (shown but unranked), so a model can't top
    the board on a single benchmark."""
    vals = [
        benchmarks[k]["score"]
        for k in OVERALL_KEYS
        if isinstance(benchmarks.get(k), dict)
        and isinstance(benchmarks[k].get("score"), (int, float))
    ]
    return sum(vals) / len(vals) if len(vals) == len(OVERALL_KEYS) else None


def main() -> None:
    data = json.loads(LATEST.read_text(encoding="utf-8"))
    # Ranked models (both customs) first, by overall desc; unranked (partial) last.
    models = sorted(
        data["models"],
        key=lambda m: (overall(m.get("benchmarks", {})) is not None, overall(m.get("benchmarks", {})) or 0),
        reverse=True,
    )
    lines = [
        "| Model | Provider | Practical | IFEval | GSM8K | Creative | Overall |",
        "|---|---|---|---|---|---|---|",
    ]
    for m in models:
        bs = m.get("benchmarks", {})
        ov = overall(bs)
        row = [m["model"].split("/")[-1], m.get("provider", "")]
        row += [cell(bs.get(k)) for k, _ in BENCHES]
        row.append(f"**{round(ov * 100)}%**" if ov is not None else "—")
        lines.append("| " + " | ".join(row) + " |")
    table = "\n".join(lines)
    note = (
        f"_Updated {data.get('last_updated', '')} · small-sample, directional · "
        "Overall = Practical + Creative (the benchmarks every provider runs); a model needs "
        "both to be ranked · `—` = not run / not ranked, `N/A` = errored that run._"
    )
    block = f"{START}\n\n{note}\n\n{table}\n\n{END}"
    text = README.read_text(encoding="utf-8")
    new = re.sub(re.escape(START) + r".*?" + re.escape(END), block, text, flags=re.DOTALL)
    README.write_text(new, encoding="utf-8")
    print("README results table updated")


if __name__ == "__main__":
    main()
