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


def overall(benchmarks: dict) -> float:
    vals = [
        v["score"]
        for v in benchmarks.values()
        if isinstance(v, dict) and isinstance(v.get("score"), (int, float))
    ]
    return sum(vals) / len(vals) if vals else 0.0


def main() -> None:
    data = json.loads(LATEST.read_text(encoding="utf-8"))
    models = sorted(
        data["models"], key=lambda m: overall(m.get("benchmarks", {})), reverse=True
    )
    lines = [
        "| Model | Provider | Practical | IFEval | GSM8K | Creative | Overall |",
        "|---|---|---|---|---|---|---|",
    ]
    for m in models:
        bs = m.get("benchmarks", {})
        row = [m["model"].split("/")[-1], m.get("provider", "")]
        row += [cell(bs.get(k)) for k, _ in BENCHES]
        row.append(f"**{round(overall(bs) * 100)}%**")
        lines.append("| " + " | ".join(row) + " |")
    table = "\n".join(lines)
    note = (
        f"_Updated {data.get('last_updated', '')} · small-sample, directional · "
        "`—` = not run for that provider, `N/A` = errored that run._"
    )
    block = f"{START}\n\n{note}\n\n{table}\n\n{END}"
    text = README.read_text(encoding="utf-8")
    new = re.sub(re.escape(START) + r".*?" + re.escape(END), block, text, flags=re.DOTALL)
    README.write_text(new, encoding="utf-8")
    print("README results table updated")


if __name__ == "__main__":
    main()
