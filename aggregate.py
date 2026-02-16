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
                        "benchmarks": {}
                    }

                # Determine benchmark type from filename or content
                if "practical-knowledge" in result_file.name:
                    models[model]["benchmarks"]["practical_knowledge"] = {
                        "score": data.get("overall_score", 0),
                        "details": {
                            cat: info.get("average_score", 0)
                            for cat, info in data.get("categories", {}).items()
                        }
                    }
                elif "creative-technical" in result_file.name:
                    models[model]["benchmarks"]["creative_technical"] = {
                        "score": data.get("overall_score", 0),
                        "details": {
                            c["prompt_id"]: c.get("weighted_score", 0)
                            for c in data.get("challenges", [])
                        }
                    }
                elif "humaneval" in result_file.name.lower():
                    models[model]["benchmarks"]["humaneval"] = {
                        "score": data.get("score", data.get("pass_at_1", 0)),
                        "pass_at_1": data.get("pass_at_1", 0)
                    }
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
        """Infer provider from model name."""
        model_lower = model.lower()
        if "groq" in model_lower or "llama" in model_lower:
            return "groq"
        elif "together" in model_lower or "mistral" in model_lower:
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
                for bench_name, bench_data in model_data.get("benchmarks", {}).items():
                    score_entry[bench_name] = bench_data.get("score", 0)

                historical["models"][model_name]["scores"].append(score_entry)

        return historical

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

        return {
            "latest": latest_path,
            "historical": historical_path
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
    else:
        latest_path = aggregator.save_latest()
        historical_path = aggregator.save_historical()
        print(f"Saved latest results: {latest_path}")
        print(f"Saved historical summary: {historical_path}")

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
