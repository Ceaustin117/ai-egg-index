#!/usr/bin/env python3
"""Openbench wrapper.

Invokes `bench eval <name>` for each requested benchmark (HumanEval / IFEval / GSM8K)
and writes a JSON file into `results/<today>/` matching the schema that `aggregate.py`
expects (filenames containing 'humaneval', 'ifeval', or 'gsm8k' with a top-level `model`
and `score` field).

openbench is built on Inspect AI, which reports scores in a rich-formatted table — NOT
as a plain `accuracy: 0.8` line — so scraping stdout with simple regexes silently fails.
We instead run with `--json` and pull the metric out of the structured output, scanning
the result recursively so we don't depend on openbench's exact JSON shape (which can
change between versions). Strategies are tried in order; if all fail the raw output is
logged and the benchmark returns False so a human notices.
"""

import argparse
import json
import re
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Optional

BENCHMARKS = {
    "humaneval": {"extra_field": "pass_at_1"},
    "ifeval": {"extra_field": "strict"},
    "gsm8k": {"extra_field": "accuracy"},
}

# Metric names to look for, in order of preference, when walking structured output.
PREFERRED_METRICS = ("accuracy", "pass_at_1", "pass@1", "mean", "score")

SCORE_PATTERNS = [
    r"pass@1[^\d]{0,12}(\d+\.?\d*)",
    r"accuracy[^\d]{0,12}(\d+\.?\d*)",
    r"\bscore[^\d]{0,12}(\d+\.?\d*)",
    r"\bmean[^\d]{0,12}(\d+\.?\d*)",
]


def _normalize(value: float) -> float:
    """Inspect reports either 0-1 or 0-100 depending on metric; normalize to 0-1."""
    return value / 100 if value > 1 else value


def extract_score_from_obj(obj) -> Optional[float]:
    """Recursively search a parsed-JSON object for a benchmark metric value.

    Inspect's eval log nests metrics under results.scores[].metrics.<name>.value, but
    different openbench versions / --json shapes vary, so we collect every
    '<preferred-metric>': number (or {'value': number}) we can find and pick the most
    preferred one. Returns a 0-1 score or None.
    """
    found: dict = {}

    def walk(node, key_hint: Optional[str] = None):
        if isinstance(node, dict):
            # {"value": 0.8} under a metric-named key
            if key_hint and "value" in node and isinstance(node["value"], (int, float)):
                found.setdefault(key_hint.lower(), float(node["value"]))
            for k, v in node.items():
                if isinstance(v, (int, float)) and k.lower() in PREFERRED_METRICS:
                    found.setdefault(k.lower(), float(v))
                walk(v, k)
        elif isinstance(node, list):
            for item in node:
                walk(item, key_hint)

    walk(obj)
    for metric in PREFERRED_METRICS:
        if metric in found:
            return _normalize(found[metric])
    return None


def extract_score_from_text(output: str) -> Optional[float]:
    """Last-resort: try to parse a JSON blob out of stdout, then fall back to regex."""
    # A JSON document may be embedded among log lines; try the largest {...} span.
    start, end = output.find("{"), output.rfind("}")
    if 0 <= start < end:
        try:
            score = extract_score_from_obj(json.loads(output[start:end + 1]))
            if score is not None:
                return score
        except json.JSONDecodeError:
            pass
    for pattern in SCORE_PATTERNS:
        match = re.search(pattern, output, re.IGNORECASE)
        if match:
            return _normalize(float(match.group(1)))
    return None


def _score_from_log_dir(log_dir: Path) -> Optional[float]:
    """Fallback: read the most recent Inspect JSON log and extract its score."""
    if not log_dir.is_dir():
        return None
    json_logs = sorted(log_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    for log in json_logs:
        try:
            score = extract_score_from_obj(json.loads(log.read_text()))
        except (json.JSONDecodeError, OSError):
            continue
        if score is not None:
            print(f"  (recovered score from log {log.name})", flush=True)
            return score
    return None


def run_benchmark(name: str, model: str, limit: int, results_dir: Path) -> bool:
    print(f"\n=== openbench: {name} (model=groq/{model}, limit={limit}) ===", flush=True)

    # openbench (>=0.5) dropped the `--json` stdout flag; the score now only lives in
    # the Inspect eval log. We write JSON logs to a throwaway dir (one per benchmark so
    # we never pick up a sibling benchmark's log) and read the metric back out of it.
    with tempfile.TemporaryDirectory(prefix=f"openbench_{name}_") as log_dir:
        cmd = [
            "bench", "eval", name,
            "--model", f"groq/{model}",
            "--limit", str(limit),
            "--log-format", "json",
            "--log-dir", log_dir,
            "--display", "none",
        ]

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

        # Read the score from the JSON eval log; fall back to scraping stdout just in case.
        score = _score_from_log_dir(Path(log_dir))
        if score is None:
            score = extract_score_from_text(result.stdout or "")

    if score is None:
        print(
            f"WARNING: ran {name} but could not find a score in the JSON eval log. "
            f"Inspect the raw output above; openbench's output shape may have changed.",
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
