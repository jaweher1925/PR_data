"""Configuration constants for the analysis pipeline."""

from __future__ import annotations

import os
from pathlib import Path

# Project root: the repository root.
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

DATA_DIR = PROJECT_ROOT / "data"

# Shipped with the repository (committed):
#   data/judge/    -- the LLM judge's pinned inputs, raw run checkpoints, and
#                     population verdict CSVs (RQ1 verification).
#   data/digests/  -- row-level digests of every result frame; the tests
#                     re-derive the paper's headline numbers from these.
JUDGE_DIR = DATA_DIR / "judge"
DIGESTS_DIR = DATA_DIR / "digests"

# Rebuilt locally by reproduction path B (all gitignored):
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"
DUCKDB_DIR = DATA_DIR / "duckdb"
DUCKDB_PATH = DUCKDB_DIR / "prfix.duckdb"

FIGURES_DIR = PROJECT_ROOT / "figures"

# AIDev primary dataset on Hugging Face.
AIDEV_HF_REPO = "hao-li/AIDev"

# Mapping of friendly table name -> Parquet filename within the HF repo.
# Top-level tables cover the full corpus; the rest belong to AIDev-pop.
AIDEV_TABLES: dict[str, str] = {
    # Top-level (full corpus).
    "all_pull_request":      "all_pull_request.parquet",
    "all_repository":        "all_repository.parquet",
    "all_user":              "all_user.parquet",
    # AIDev-pop curated subset.
    "pull_request":          "pull_request.parquet",
    "repository":            "repository.parquet",
    "pr_timeline":           "pr_timeline.parquet",
    "pr_comments":           "pr_comments.parquet",
    "pr_reviews":            "pr_reviews.parquet",
    "pr_review_comments_v2": "pr_review_comments_v2.parquet",
    "pr_commits":            "pr_commits.parquet",
    "pr_commit_details":     "pr_commit_details.parquet",
    "pr_task_type":          "pr_task_type.parquet",
    "issue":                 "issue.parquet",
    "related_issue":         "related_issue.parquet",
    "user":                  "user.parquet",
    # Human-PR comparison cohort (≥500★ subset, sampled from same repos).
    "human_pull_request":    "human_pull_request.parquet",
    "human_pr_task_type":    "human_pr_task_type.parquet",
}

# DuckDB tuning. Override via env vars.
DUCKDB_MEMORY_LIMIT = os.environ.get("DUCKDB_MEMORY_LIMIT", "16GB")
DUCKDB_THREADS = int(os.environ.get("DUCKDB_THREADS", "8"))

# Agent labels in AIDev's `agent` column. Stored with underscores, no "GitHub" prefix.
# Observed counts in all_pull_request (verified 2026-06-03):
#   OpenAI_Codex 814,522  Copilot 50,447  Cursor 32,941  Devin 29,744  Claude_Code 5,137
AGENTS: list[str] = ["OpenAI_Codex", "Devin", "Copilot", "Cursor", "Claude_Code"]


def ensure_dirs() -> None:
    """Create all data and output directories if they don't already exist."""
    for d in (DATA_DIR, JUDGE_DIR, DIGESTS_DIR, RAW_DIR, INTERIM_DIR,
              PROCESSED_DIR, DUCKDB_DIR, FIGURES_DIR):
        d.mkdir(parents=True, exist_ok=True)
