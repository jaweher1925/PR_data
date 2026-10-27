"""Paths used by every step. Change REPO if Mohamed's repo is somewhere else."""
import os

REPO = os.environ.get("EA_REPO", r"C:\Users\GIGABYTE\Desktop\code_theis\evolution-assurance")
if not os.path.isdir(REPO):                       # fallback when run on Linux
    REPO = os.path.expanduser("~/ea")
HERE = os.path.dirname(os.path.abspath(__file__))
RAW_V5 = os.path.join(REPO, "data", "raw_v5")           # v5 tables (step 1) + task type (step 2)
RAW_V3 = os.path.join(REPO, "data", "raw_linker")       # v3 labels used to train the classifier
DB = os.path.join(REPO, "data", "duckdb", "aidev_v5.duckdb")
RESULTS = os.path.join(HERE, "results")
V5_REPO = "hao-li/AIDev-7.6M"
