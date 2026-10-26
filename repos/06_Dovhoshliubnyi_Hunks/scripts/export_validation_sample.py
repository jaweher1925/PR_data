"""Generate a blind validation sample for inter-rater agreement (Fleiss κ).

Reads perf_improvement_parts.csv, draws a random sample of N accepted hunks,
and writes a CSV with the categories column removed so raters classify blindly.

Usage:
  .venv/bin/python3 scripts/export_validation_sample.py [--n 60] [--seed 42]

Each rater:
  1. Opens validation_sample.csv
  2. Adds a column 'categories' and fills in comma-separated category IDs (0-17)
     e.g.  13,3   or   5   or leave blank if none apply
  3. Saves as validation_rater_<name>.csv

Then run compute_agreement.py to get Fleiss κ.
"""
import csv
import os
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from analysis.patch_classifier import MUTATION_CATEGORIES

PARTS_CSV = "perf_improvement_parts.csv"
OUTPUT_CSV = "validation_sample.csv"
DEFAULT_N = 60
DEFAULT_SEED = 42


def main():
    n = DEFAULT_N
    seed = DEFAULT_SEED
    args = sys.argv[1:]
    for i, arg in enumerate(args):
        if arg == "--n" and i + 1 < len(args):
            n = int(args[i + 1])
        if arg == "--seed" and i + 1 < len(args):
            seed = int(args[i + 1])

    if not os.path.exists(PARTS_CSV):
        print(f"ERROR: {PARTS_CSV} not found. Run --export first.")
        sys.exit(1)

    with open(PARTS_CSV, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    # Only sample rows that have been classified (non-empty categories)
    classified = [r for r in rows if r.get("categories", "").strip()]
    if len(classified) < n:
        print(f"Warning: only {len(classified)} classified hunks available, sampling all.")
        n = len(classified)

    rng = random.Random(seed)
    sample = rng.sample(classified, n)
    sample.sort(key=lambda r: int(r["part_id"]))

    # Blind CSV: drop categories + reasoning, keep everything else
    blind_fields = [
        "part_id", "pr_id", "pr_url", "final_sub_pattern",
        "commit_sha", "filename", "hunk_index", "hunk_content",
        "categories",  # blank — rater fills this in
    ]

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=blind_fields)
        writer.writeheader()
        for row in sample:
            writer.writerow({
                "part_id": row["part_id"],
                "pr_id": row["pr_id"],
                "pr_url": row["pr_url"],
                "final_sub_pattern": row["final_sub_pattern"],
                "commit_sha": row["commit_sha"],
                "filename": row["filename"],
                "hunk_index": row["hunk_index"],
                "hunk_content": row["hunk_content"],
                "categories": "",  # intentionally blank
            })

    print(f"Wrote {n} blind hunks → {OUTPUT_CSV}  (seed={seed})")
    print("\n18 mutation categories for reference:")
    for cid, (name, desc) in MUTATION_CATEGORIES.items():
        print(f"  {cid:2d}  {name:<30}  {desc}")
    print(f"\nInstructions for raters:")
    print(f"  1. Open {OUTPUT_CSV}")
    print(f"  2. For each row, fill 'categories' with comma-separated IDs (e.g. 5,13)")
    print(f"     Leave blank if the hunk contains no meaningful mutation")
    print(f"  3. Save as validation_rater_<yourname>.csv")
    print(f"  4. Send all rater files to: python scripts/compute_agreement.py "
          f"validation_rater_A.csv validation_rater_B.csv validation_rater_C.csv")


if __name__ == "__main__":
    main()
