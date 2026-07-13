#!/usr/bin/env python3
"""Generate a short "This Week's Egg Winner" recap from the latest results.

Reads output/latest.json (current standings) and output/historical.json (for the biggest
mover) and writes output/weekly-recap.md — a ready-to-post summary. Run in the weekly
workflow after aggregate.py / the README table refresh. Pure stdlib.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LATEST = ROOT / "output" / "latest.json"
HIST = ROOT / "output" / "historical.json"
OUT = ROOT / "output" / "weekly-recap.md"
SITE = "https://ceaustin117.github.io/ai-egg-index/"
BENCH = ["practical_knowledge", "ifeval", "gsm8k", "creative_technical"]


def short(model_id: str) -> str:
    return model_id.split("/")[-1]


def _mean(values) -> float | None:
    nums = [v for v in values if isinstance(v, (int, float))]
    return sum(nums) / len(nums) if nums else None


def overall_latest(model: dict) -> float | None:
    return _mean([b.get("score") for b in model.get("benchmarks", {}).values() if isinstance(b, dict)])


def overall_hist(entry: dict) -> float | None:
    return _mean([entry.get(k) for k in BENCH])


def main() -> None:
    latest = json.loads(LATEST.read_text(encoding="utf-8"))
    date = latest.get("last_updated", "")
    ranked = sorted(
        ((overall_latest(m) or 0.0, m) for m in latest.get("models", [])),
        key=lambda x: x[0],
        reverse=True,
    )
    if not ranked:
        OUT.write_text("_No results yet._\n", encoding="utf-8")
        print("No results yet.")
        return

    winner_score, winner = ranked[0]

    # Biggest mover: largest overall change vs the previous run, among current models.
    hist = json.loads(HIST.read_text(encoding="utf-8")) if HIST.exists() else {"models": {}}
    mover = None  # (delta, model, prev, cur)
    for _, m in ranked:
        h = hist.get("models", {}).get(m["model"])
        if not h:
            continue
        series = h.get("rolling") or h.get("scores") or []
        pts = [overall_hist(e) for e in series]
        pts = [v for v in pts if v is not None]
        if len(pts) < 2:
            continue
        delta = pts[-1] - pts[-2]
        if mover is None or abs(delta) > abs(mover[0]):
            mover = (delta, m, pts[-2], pts[-1])

    lines = [
        f"## 🥚 This Week's Egg Winner: {short(winner['model'])} ({round(winner_score * 100)}%)",
        "",
        f"_Free-tier LLMs on everyday tasks — week of {date}._",
        "",
        "**Rankings (overall):**",
    ]
    for i, (o, m) in enumerate(ranked, 1):
        lines.append(f"{i}. **{short(m['model'])}** ({m.get('provider', '')}) — {round(o * 100)}%")

    if mover:
        delta, m, prev, cur = mover
        arrow = "▲" if delta >= 0 else "▼"
        sign = "+" if delta >= 0 else ""
        lines += [
            "",
            f"**Biggest mover:** {short(m['model'])} {arrow} {round(prev * 100)}% → "
            f"{round(cur * 100)}% ({sign}{round(delta * 100)} pts)",
        ]

    lines += [
        "",
        "_Small-sample, directional — don't over-read a few points._",
        "",
        f"Full leaderboard → {SITE}",
    ]
    md = "\n".join(lines) + "\n"
    OUT.write_text(md, encoding="utf-8")
    try:
        print(md)
    except UnicodeEncodeError:  # e.g. Windows cp1252 console
        import sys

        sys.stdout.buffer.write(md.encode("utf-8"))


if __name__ == "__main__":
    main()
