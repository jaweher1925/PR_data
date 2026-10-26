"""Calibrate the patch classifier on a sample of patches.

Samples 40 random patches from commit_details (excluding Mochi compiler
output), runs the 18-category mutation classifier, and writes results
to CSV for manual inspection.

Usage:
    python scripts/calibrate_patch_classifier.py
"""
import csv
import json
import random
import sys
from collections import Counter
from pathlib import Path

import anthropic

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import ANTHROPIC_MODEL, DB_PATH
from db.connection import get_connection
from analysis.patch_classifier import _build_prompt, _parse_response, MUTATION_CATEGORIES

SAMPLE_SIZE = 40
RANDOM_SEED = 777
OUTPUT_CSV = Path(__file__).parent.parent / "calibration_results.csv"


def main():
    with get_connection(DB_PATH) as conn:
        rows = conn.execute("""
            SELECT cd.pr_id, cd.sha, cd.filename, cd.patch
            FROM commit_details cd
            JOIN perf_pr p ON p.pr_id = cd.pr_id
            WHERE cd.patch IS NOT NULL AND LENGTH(cd.patch) > 0
              AND p.pr_url NOT LIKE '%mochilang/mochi%'
        """).fetchall()

    print(f"Total patches (excl. Mochi): {len(rows)}")
    random.seed(RANDOM_SEED)
    sample = random.sample(rows, min(SAMPLE_SIZE, len(rows)))
    print(f"Sampled {len(sample)} patches for calibration")
    print()

    client = anthropic.Anthropic()
    category_counts = Counter()
    errors = 0
    csv_rows = []

    for pr_id, sha, filename, patch in sample:
        prompt = _build_prompt(filename, patch)
        try:
            msg = client.messages.create(
                model=ANTHROPIC_MODEL,
                max_tokens=256,
                messages=[{"role": "user", "content": prompt}],
            )
            categories, reasoning = _parse_response(msg.content[0].text)
        except Exception as exc:
            print(f"  ERROR on {sha[:8]}/{filename}: {exc}")
            errors += 1
            continue

        for c in categories:
            category_counts[c["id"]] += 1

        cat_str = "; ".join(f"{c['id']}:{c['name']}({c['confidence']:.1f})" for c in categories)
        patch_preview = (patch or "")[:200].replace("\n", "\\n")
        csv_rows.append({
            "pr_id": pr_id,
            "sha": sha[:8],
            "filename": filename,
            "categories": cat_str,
            "reasoning": reasoning,
            "patch_preview": patch_preview,
        })

    # Write CSV
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["pr_id", "sha", "filename", "categories", "reasoning", "patch_preview"])
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"Results written to {OUTPUT_CSV}")

    # Print distribution summary
    print()
    print("CATEGORY DISTRIBUTION")
    print("-" * 50)
    for cat_id in sorted(MUTATION_CATEGORIES.keys()):
        name, _ = MUTATION_CATEGORIES[cat_id]
        count = category_counts.get(cat_id, 0)
        bar = "#" * count
        print(f"  {cat_id:2d}  {name:28s}  {count:3d}  {bar}")

    print(f"\n  Total: {sum(category_counts.values())} labels across {len(csv_rows)} patches, {errors} errors")
    print()
    print("Next: python pipeline.py --skip-load --classify-patches")


if __name__ == "__main__":
    main()
