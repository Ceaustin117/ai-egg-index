#!/usr/bin/env python3
"""
Benchmark Results Aggregator

Aggregates individual benchmark results into a unified format for the frontend.
Builds historical summary from dated results directories.
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any


def _summarize_error(msg: str) -> str:
    """Map a raw provider error string to a short, human-readable reason for the UI tooltip."""
    low = msg.lower()
    # Check quota/rate before credit: some quota errors (e.g. Google 429) also mention
    # "billing" in their help URL, but a 402 credit error never mentions quota/429.
    if "429" in low or "quota" in low or "resourceexhausted" in low or "rate limit" in low:
        return "Free-tier quota/rate limit exceeded"
    if "credit" in low or "402" in low or "billing" in low:
        return "Credit/billing limit exceeded"
    if "401" in low or "403" in low or "permission" in low or "api key" in low or "unauthorized" in low:
        return "Authentication/permission error"
    if "404" in low or "not found" in low or "does not exist" in low or "decommission" in low:
        return "Model unavailable / not found"
    return "Provider API error"


def _api_error_status(responses: list) -> tuple:
    """Decide whether a whole benchmark failed due to provider/API errors rather than
    genuine low performance.

    providers.py returns API failures as strings starting with 'ERROR:'. A benchmark is
    considered failed only if EVERY response errored — a single real response means the
    model answered (possibly poorly), which is a legitimate score, not an outage.

    Returns (failed: bool, reason: str | None).
    """
    texts = [r for r in responses if isinstance(r, str)]
    if not texts:
        return False, None
    errors = [t for t in texts if t.startswith("ERROR:")]
    if len(errors) < len(texts):
        return False, None
    return True, _summarize_error(errors[0])


class BenchmarkAggregator:
    """Aggregates benchmark results from multiple sources."""

    def __init__(self, base_dir: str = None):
        self.base_dir = Path(base_dir) if base_dir else Path(__file__).parent
        self.results_dir = self.base_dir / "results"
        self.historical_dir = self.base_dir / "historical"

    def _parse_model_results(self, date_dir: Path) -> dict:
        """Parse all model results from a dated directory."""
        models = {}

        for result_file in date_dir.glob("*.json"):
            try:
                with open(result_file) as f:
                    data = json.load(f)

                model = data.get("model", result_file.stem)

                if model not in models:
                    models[model] = {
                        "model": model,
                        "provider": self._infer_provider(model),
                        "tier": "free",
                        "date": date_dir.name,
                        "methodology_version": data.get("methodology_version"),
                        "git_commit": data.get("git_commit"),
                        "benchmarks": {}
                    }
                # Backfill provenance if the first file for this model lacked it.
                if models[model].get("methodology_version") is None:
                    models[model]["methodology_version"] = data.get("methodology_version")

                # Determine benchmark type from filename or content
                if "practical-knowledge" in result_file.name:
                    responses = [
                        q.get("response", "")
                        for cat in data.get("categories", {}).values()
                        for q in cat.get("questions", [])
                    ]
                    failed, reason = _api_error_status(responses)
                    if failed:
                        models[model]["benchmarks"]["practical_knowledge"] = {
                            "score": None, "status": "error", "error": reason
                        }
                    else:
                        models[model]["benchmarks"]["practical_knowledge"] = {
                            "score": data.get("overall_score", 0),
                            "details": {
                                cat: info.get("average_score", 0)
                                for cat, info in data.get("categories", {}).items()
                            }
                        }
                elif "creative-technical" in result_file.name:
                    responses = [c.get("response_preview", "") for c in data.get("challenges", [])]
                    failed, reason = _api_error_status(responses)
                    if failed:
                        models[model]["benchmarks"]["creative_technical"] = {
                            "score": None, "status": "error", "error": reason
                        }
                    else:
                        models[model]["benchmarks"]["creative_technical"] = {
                            "score": data.get("overall_score", 0),
                            "details": {
                                c["prompt_id"]: c.get("weighted_score", 0)
                                for c in data.get("challenges", [])
                            }
                        }
                # HumanEval intentionally dropped from the index (see README/run.py);
                # any legacy humaneval result files are ignored, not aggregated.
                elif "ifeval" in result_file.name.lower():
                    models[model]["benchmarks"]["ifeval"] = {
                        "score": data.get("score", 0),
                        "strict": data.get("strict", 0),
                        "loose": data.get("loose", 0)
                    }
                elif "gsm8k" in result_file.name.lower():
                    models[model]["benchmarks"]["gsm8k"] = {
                        "score": data.get("score", data.get("accuracy", 0)),
                        "accuracy": data.get("accuracy", 0)
                    }

            except (json.JSONDecodeError, KeyError) as e:
                print(f"Warning: Could not parse {result_file}: {e}")
                continue

        return models

    def _infer_provider(self, model: str) -> str:
        """Infer provider from model name. Prefers explicit '<provider>/<model>' prefix
        (e.g., 'huggingface/mistralai/Mistral-7B') over substring matching, which has
        collisions (e.g., 'mistral' appears in both Together and HuggingFace model IDs).
        """
        model_lower = model.lower()
        known = ("groq", "together", "google", "cohere", "huggingface")
        if "/" in model_lower:
            prefix = model_lower.split("/", 1)[0]
            if prefix in known:
                return prefix
        # Fallback substring matching for unprefixed legacy entries
        if "groq" in model_lower or "llama" in model_lower:
            return "groq"
        elif "together" in model_lower:
            return "together"
        elif "gemini" in model_lower:
            return "google"
        elif "cohere" in model_lower or "command" in model_lower:
            return "cohere"
        elif "hugging" in model_lower or "hf" in model_lower:
            return "huggingface"
        return "unknown"

    def aggregate_latest(self) -> dict:
        """Aggregate results from the most recent date directory."""
        if not self.results_dir.exists():
            return {"models": [], "last_updated": None}

        # Find most recent dated directory
        date_dirs = sorted(
            [d for d in self.results_dir.iterdir() if d.is_dir() and d.name[0].isdigit()],
            reverse=True
        )

        if not date_dirs:
            return {"models": [], "last_updated": None}

        latest_dir = date_dirs[0]
        models = self._parse_model_results(latest_dir)

        return {
            "last_updated": latest_dir.name,
            "models": list(models.values())
        }

    def build_historical_summary(self) -> dict:
        """Build historical summary from all dated result directories."""
        if not self.results_dir.exists():
            return {"last_updated": None, "models": {}}

        historical = {
            "last_updated": datetime.now().strftime("%Y-%m-%d"),
            "models": {}
        }

        # Process each dated directory
        date_dirs = sorted(
            [d for d in self.results_dir.iterdir() if d.is_dir() and d.name[0].isdigit()]
        )

        for date_dir in date_dirs:
            models = self._parse_model_results(date_dir)

            for model_name, model_data in models.items():
                if model_name not in historical["models"]:
                    historical["models"][model_name] = {
                        "provider": model_data["provider"],
                        "tier": model_data["tier"],
                        "scores": []
                    }

                # Add score entry for this date
                score_entry = {"date": date_dir.name}
                if model_data.get("methodology_version") is not None:
                    score_entry["methodology_version"] = model_data["methodology_version"]
                for bench_name, bench_data in model_data.get("benchmarks", {}).items():
                    score_entry[bench_name] = bench_data.get("score", 0)

                historical["models"][model_name]["scores"].append(score_entry)

        # Add a trailing-window rolling average per model/benchmark. Single-run scores are
        # noisy (small samples), so the smoothed series is the more meaningful trend.
        for model_data in historical["models"].values():
            model_data["rolling"] = self._rolling_averages(model_data["scores"])

        return historical

    @staticmethod
    def _rolling_averages(scores: list, window: int = 3) -> list:
        """Trailing-`window` mean per benchmark, computed run-over-run. Null/missing
        scores are skipped (not treated as 0), so an errored run doesn't drag the trend."""
        rolling = []
        for i, entry in enumerate(scores):
            window_slice = scores[max(0, i - window + 1): i + 1]
            smoothed = {"date": entry["date"]}
            for key in entry:
                if key == "date":
                    continue
                vals = [w[key] for w in window_slice if isinstance(w.get(key), (int, float))]
                if vals:
                    smoothed[key] = round(sum(vals) / len(vals), 3)
            rolling.append(smoothed)
        return rolling

    @staticmethod
    def _cell_status(bench_data: dict) -> str:
        """ok if a benchmark produced a real score; error if it failed (API/quota error
        or a null score)."""
        if bench_data.get("status") == "error" or bench_data.get("score") is None:
            return "error"
        return "ok"

    def build_run_health(self) -> dict:
        """Per-run health derived from the result files: for each run, which
        (model, benchmark) cells scored ok vs errored, plus summary counts. Lets us track
        pipeline degradation over time (e.g. Groq getting rate-limited, Google dropping
        out) without digging through CI logs. A model absent from a run's `models` list
        produced no results at all that run."""
        health = {"last_updated": datetime.now().strftime("%Y-%m-%d"), "runs": []}
        if not self.results_dir.exists():
            return health

        date_dirs = sorted(
            [d for d in self.results_dir.iterdir() if d.is_dir() and d.name[0].isdigit()]
        )
        for date_dir in date_dirs:
            models = self._parse_model_results(date_dir)
            cells, ok, err = {}, 0, 0
            for model_name, model_data in models.items():
                statuses = {}
                for bench_name, bench_data in model_data.get("benchmarks", {}).items():
                    status = self._cell_status(bench_data)
                    statuses[bench_name] = status
                    if status == "ok":
                        ok += 1
                    else:
                        err += 1
                cells[model_name] = statuses
            health["runs"].append({
                "date": date_dir.name,
                "models": sorted(cells.keys()),
                "cells": cells,
                "summary": {"ok": ok, "error": err, "total": ok + err},
            })
        return health

    def save_run_health(self, output_path: str = None) -> str:
        """Save run-health history."""
        if output_path is None:
            output_path = self.results_dir / "run-health.json"
        with open(output_path, "w") as f:
            json.dump(self.build_run_health(), f, indent=2)
        return str(output_path)

    def save_latest(self, output_path: str = None) -> str:
        """Save aggregated latest results."""
        if output_path is None:
            output_path = self.results_dir / "latest.json"

        data = self.aggregate_latest()

        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)

        return str(output_path)

    def save_historical(self, output_path: str = None) -> str:
        """Save historical summary."""
        if output_path is None:
            self.historical_dir.mkdir(parents=True, exist_ok=True)
            output_path = self.historical_dir / "summary.json"

        data = self.build_historical_summary()

        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)

        return str(output_path)

    def export_for_frontend(self, output_dir: str) -> dict:
        """Export both latest and historical data for frontend consumption."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        latest_path = self.save_latest(output_dir / "latest.json")
        historical_path = self.save_historical(output_dir / "historical.json")
        health_path = self.save_run_health(output_dir / "run-health.json")

        return {
            "latest": latest_path,
            "historical": historical_path,
            "run_health": health_path,
        }


def main():
    """Run the aggregation."""
    import argparse

    parser = argparse.ArgumentParser(description="Aggregate benchmark results")
    parser.add_argument("--base-dir", help="Base directory for benchmark-data")
    parser.add_argument("--output", help="Output directory for aggregated files")
    parser.add_argument("--frontend", help="Export to frontend public directory")

    args = parser.parse_args()

    aggregator = BenchmarkAggregator(args.base_dir)

    if args.frontend:
        paths = aggregator.export_for_frontend(args.frontend)
        print(f"Exported to frontend:")
        print(f"  Latest: {paths['latest']}")
        print(f"  Historical: {paths['historical']}")
        print(f"  Run health: {paths['run_health']}")
    else:
        latest_path = aggregator.save_latest()
        historical_path = aggregator.save_historical()
        health_path = aggregator.save_run_health()
        print(f"Saved latest results: {latest_path}")
        print(f"Saved historical summary: {historical_path}")
        print(f"Saved run health: {health_path}")

    # Print summary
    latest = aggregator.aggregate_latest()
    print(f"\nSummary:")
    print(f"  Last updated: {latest['last_updated']}")
    print(f"  Models: {len(latest['models'])}")

    for model in latest["models"]:
        print(f"\n  {model['model']} ({model['provider']}):")
        for bench, data in model.get("benchmarks", {}).items():
            print(f"    {bench}: {data.get('score', 'N/A')}")


if __name__ == "__main__":
    main()
