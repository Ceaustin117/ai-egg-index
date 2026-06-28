#!/usr/bin/env python3
"""Openbench wrapper.

Invokes `bench eval <name>` for each requested benchmark (IFEval / GSM8K) and writes a
JSON file into `results/<today>/` matching the schema that `aggregate.py` expects
(filenames containing 'ifeval' or 'gsm8k' with a top-level `model` and `score` field).

openbench is built on Inspect AI, which writes a structured JSON eval log (via
`--log-format json --log-dir`). We read the aggregate metric straight out of that log's
`results.scores[].metrics.<name>.value`, using a per-benchmark list of metric names
(see BENCHMARKS). The lookup is targeted on purpose: a generic "find any score-ish
number" walk grabs the wrong value — e.g. ifeval exposes strict/loose prompt- and
instruction-level accuracies and no plain "accuracy", so a generic walk silently
returned 0.0. If the metric isn't found (or the eval errored), the benchmark returns
False so a human notices.
"""

import argparse
import json
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Optional

# HumanEval is intentionally omitted: grading it means executing model-generated code
# against unit tests in a Docker sandbox — an abstract academic benchmark that doesn't
# fit this index's "tasks regular people care about" mission. See README.
#
# Per benchmark: `metric_keys` is the priority-ordered list of Inspect metric names to
# read; `extra_field` is the extra key we mirror the score into for aggregate.py. ifeval
# reports four accuracies — we use prompt-level strict (did the model satisfy ALL
# instructions in a prompt), the standard strict headline number.
BENCHMARKS = {
    "ifeval": {
        "extra_field": "strict",
        "metric_keys": ("strict_prompt_accuracy", "strict_instruction_accuracy"),
    },
    "gsm8k": {
        "extra_field": "accuracy",
        "metric_keys": ("accuracy",),
    },
}


def _normalize(value: float) -> float:
    """Inspect reports either 0-1 or 0-100 depending on metric; normalize to 0-1."""
    return value / 100 if value > 1 else value


def _metric_from_results(log_obj: dict, metric_keys) -> Optional[float]:
    """Pull a benchmark's aggregate score from an Inspect eval log.

    Inspect stores aggregate metrics at results.scores[].metrics.<name>.value. We read
    the first of `metric_keys` that's present — and only from a successful eval, since a
    log with status != 'success' (e.g. ifeval erroring on a missing nltk dependency) has
    no real results to read.
    """
    if log_obj.get("status") != "success":
        return None
    for score in (log_obj.get("results") or {}).get("scores", []):
        metrics = score.get("metrics", {})
        for key in metric_keys:
            entry = metrics.get(key)
            if isinstance(entry, dict) and isinstance(entry.get("value"), (int, float)):
                return _normalize(float(entry["value"]))
    return None


def _score_from_log_dir(log_dir: Path, metric_keys) -> Optional[float]:
    """Read the most recent Inspect JSON log in log_dir and extract its metric."""
    if not log_dir.is_dir():
        return None
    json_logs = sorted(log_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    for log in json_logs:
        try:
            score = _metric_from_results(json.loads(log.read_text()), metric_keys)
        except (json.JSONDecodeError, OSError):
            continue
        if score is not None:
            print(f"  (read score from log {log.name})", flush=True)
            return score
    return None


def run_benchmark(name: str, provider: str, model: str, limit: int, results_dir: Path) -> bool:
    spec = f"{provider}/{model}"
    print(f"\n=== openbench: {name} (model={spec}, limit={limit}) ===", flush=True)

    # openbench (>=0.5) dropped the `--json` stdout flag; the score now only lives in
    # the Inspect eval log. We write JSON logs to a throwaway dir (one per benchmark so
    # we never pick up a sibling benchmark's log) and read the metric back out of it.
    with tempfile.TemporaryDirectory(prefix=f"openbench_{name}_") as log_dir:
        cmd = [
            "bench", "eval", name,
            "--model", spec,
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

        # Read the score straight from the JSON eval log's aggregate metrics.
        score = _score_from_log_dir(Path(log_dir), BENCHMARKS[name]["metric_keys"])

    if score is None:
        print(
            f"WARNING: ran {name} but could not find a score in the JSON eval log. "
            f"Inspect the raw output above; openbench's output shape may have changed.",
            flush=True,
        )
        return False

    output = {
        "model": spec,
        "score": round(score, 3),
        BENCHMARKS[name]["extra_field"]: round(score, 3),
        "timestamp": datetime.now().isoformat(),
    }

    sanitized = model.replace("/", "-").replace(":", "-")
    output_path = results_dir / f"{name}-{provider}-{sanitized}.json"
    output_path.write_text(json.dumps(output, indent=2))
    print(f"Saved: {output_path}", flush=True)
    return True


def main():
    parser = argparse.ArgumentParser(description="Run openbench evals and write our schema")
    parser.add_argument("--provider", default="groq", help="Inspect provider prefix (e.g. groq, cohere)")
    parser.add_argument("--model", default="llama-3.1-8b-instant", help="Model id (without provider prefix)")
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
        run_benchmark(name, args.provider, args.model, args.limit, results_dir)
        for name in args.benchmarks
    )
    print(f"\n=== openbench: {successes}/{len(args.benchmarks)} benchmarks succeeded ===")
    # Fail unless EVERY requested benchmark produced a score. The workflow's openbench
    # step is continue-on-error and a later step surfaces a non-success as a red job, so
    # a single silently-missing benchmark (e.g. ifeval) can't hide behind another's pass.
    sys.exit(0 if successes == len(args.benchmarks) else 1)


if __name__ == "__main__":
    main()
