#!/usr/bin/env python3
"""Openbench wrapper.

Invokes `bench eval <name>` for each requested benchmark (HumanEval / IFEval / GSM8K),
parses the score out of stdout, and writes a JSON file into `results/<today>/` matching
the schema that `aggregate.py` expects (filenames containing 'humaneval', 'ifeval',
or 'gsm8k' with a top-level `model` and `score` field).

This is intentionally best-effort: openbench's stdout format may evolve, so the score
parser tries several common patterns. If it can't find one, the step logs the raw output
and exits non-zero so a human notices.
"""

import argparse
import json
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

BENCHMARKS = {
    "humaneval": {"extra_field": "pass_at_1"},
    "ifeval": {"extra_field": "strict"},
    "gsm8k": {"extra_field": "accuracy"},
}

SCORE_PATTERNS = [
    r"pass@1[:\s=]+(\d+\.?\d*)",
    r"accuracy[:\s=]+(\d+\.?\d*)",
    r"\bscore[:\s=]+(\d+\.?\d*)",
    r"mean[:\s=]+(\d+\.?\d*)",
]


def parse_score(output: str) -> Optional[float]:
    for pattern in SCORE_PATTERNS:
        match = re.search(pattern, output, re.IGNORECASE)
        if match:
            score = float(match.group(1))
            return score / 100 if score > 1 else score
    return None


def run_benchmark(name: str, model: str, limit: int, results_dir: Path) -> bool:
    print(f"\n=== openbench: {name} (model=groq/{model}, limit={limit}) ===", flush=True)
    cmd = ["bench", "eval", name, "--model", f"groq/{model}", "--limit", str(limit)]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
    except FileNotFoundError:
        print("ERROR: `bench` CLI not found. Install openbench: pip install openbench", flush=True)
        return False
    except subprocess.TimeoutExpired:
        print(f"ERROR: {name} timed out after 15 min", flush=True)
        return False

    combined = (result.stdout or "") + "\n" + (result.stderr or "")
    print(combined[-3000:], flush=True)

    if result.returncode != 0:
        print(f"ERROR: openbench exited {result.returncode} for {name}", flush=True)
        return False

    score = parse_score(combined)
    if score is None:
        print(
            f"WARNING: could not parse score for {name}. "
            f"Update SCORE_PATTERNS in benchmarks/openbench/run.py.",
            flush=True,
        )
        return False

    output = {
        "model": f"groq/{model}",
        "score": round(score, 3),
        BENCHMARKS[name]["extra_field"]: round(score, 3),
        "timestamp": datetime.now().isoformat(),
    }

    sanitized = model.replace("/", "-").replace(":", "-")
    output_path = results_dir / f"{name}-groq-{sanitized}.json"
    output_path.write_text(json.dumps(output, indent=2))
    print(f"Saved: {output_path}", flush=True)
    return True


def main():
    parser = argparse.ArgumentParser(description="Run openbench evals and write our schema")
    parser.add_argument("--model", default="llama-3.1-8b-instant", help="Groq model to evaluate")
    parser.add_argument("--limit", type=int, default=10, help="Samples per benchmark")
    parser.add_argument(
        "--benchmarks",
        nargs="+",
        default=list(BENCHMARKS.keys()),
        choices=list(BENCHMARKS.keys()),
    )
    parser.add_argument("--results-dir", help="Override results dir (default: ../../results/<today>)")
    args = parser.parse_args()

    if args.results_dir:
        results_dir = Path(args.results_dir)
    else:
        results_dir = (
            Path(__file__).resolve().parent.parent.parent
            / "results"
            / datetime.now().strftime("%Y-%m-%d")
        )
    results_dir.mkdir(parents=True, exist_ok=True)

    successes = sum(
        run_benchmark(name, args.model, args.limit, results_dir) for name in args.benchmarks
    )
    print(f"\n=== openbench: {successes}/{len(args.benchmarks)} benchmarks succeeded ===")
    sys.exit(0 if successes > 0 else 1)


if __name__ == "__main__":
    main()
