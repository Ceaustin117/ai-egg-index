"""Provenance stamping for benchmark result files.

Every result JSON records the methodology version + code commit + run params so scores
stay comparable over time and non-comparable runs can be told apart.

METHODOLOGY_VERSION must be bumped whenever a change makes new scores NON-comparable to
old ones — new/changed prompts, a scoring-rubric change, a different judge model, or a
change in sample selection. Never silently re-score under the same version. Record every
bump in METHODOLOGY_CHANGELOG.md. (This is separate from the code's SemVer / git tags.)
"""
import os
import subprocess

METHODOLOGY_VERSION = "1.0"


def _git_commit() -> str:
    """Best-effort short commit hash: CI env first, then local git, else 'unknown'."""
    sha = os.environ.get("GITHUB_SHA")
    if sha:
        return sha[:12]
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return "unknown"


def provenance(**extra) -> dict:
    """A provenance block to merge into a result record. `extra` fields with a value of
    None are dropped (e.g. an omitted --limit means a full run)."""
    block = {
        "methodology_version": METHODOLOGY_VERSION,
        "git_commit": _git_commit(),
    }
    block.update({k: v for k, v in extra.items() if v is not None})
    return block
