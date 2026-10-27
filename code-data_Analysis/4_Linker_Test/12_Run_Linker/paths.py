"""Paths used by every step. Change REPO if Mohamed's repo is somewhere else."""
import os

REPO = os.environ.get("EA_REPO", r"C:\Users\GIGABYTE\Desktop\code_theis\evolution-assurance")
if not os.path.isdir(REPO):                       # fallback when run on Linux
    REPO = os.path.expanduser("~/ea")
RAW_LINKER = os.path.join(REPO, "data", "raw_linker")     # the 5 AIDev tables (step 1)
DUCKDB = os.path.join(REPO, "data", "duckdb", "prfix.duckdb")
HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
TABLES = ["pull_request", "repository", "pr_task_type", "human_pull_request", "human_pr_task_type"]
