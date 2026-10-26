"""Test the validation prompt against the 60 hand-labeled PRs.

Reports precision, recall, and F1 so you can confirm the prompt meets
the >=80% precision target before spending on all 629 PRs.

Usage:
    python scripts/calibrate_validator.py
"""
import csv
import json
import sys
from pathlib import Path

import anthropic

# Ensure project root is on the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import ANTHROPIC_MODEL, DB_PATH
from db.connection import get_connection
from analysis.pr_validator import _build_enriched_input, _build_prompt, _parse_response

VALIDATION_CSV = Path(__file__).parent.parent / "validation" / "validation_results.csv"


def load_ground_truth():
    ground_truth = {}
    with open(VALIDATION_CSV, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            ground_truth[int(row["pr_id"])] = row["is_performance_related"] == "True"
    return ground_truth


def main():
    ground_truth = load_ground_truth()
    pr_ids = list(ground_truth.keys())
    print(f"Ground truth: {len(pr_ids)} labeled PRs")
    print(f"  Positive: {sum(ground_truth.values())}")
    print(f"  Negative: {sum(1 for v in ground_truth.values() if not v)}")
    print()

    with get_connection(DB_PATH) as conn:
        pr_data = _build_enriched_input(conn, pr_ids)

    client = anthropic.Anthropic()
    results = {}
    errors = 0

    print(f"Classifying {len(pr_ids)} PRs with validation prompt ...")
    for pr_id in pr_ids:
        row = pr_data.get(pr_id)
        if row is None:
            print(f"  WARNING: PR {pr_id} not found in DB, skipping")
            errors += 1
            continue
        prompt = _build_prompt(row)
        try:
            message = client.messages.create(
                model=ANTHROPIC_MODEL,
                max_tokens=256,
                messages=[{"role": "user", "content": prompt}],
            )
            is_perf, reasoning = _parse_response(message.content[0].text)
            results[pr_id] = is_perf
        except Exception as exc:
            print(f"  ERROR on PR {pr_id}: {exc}")
            errors += 1

    if not results:
        print("No results — check API key and connectivity.")
        return

    # Compute precision, recall, F1
    tp = sum(1 for pr_id, pred in results.items() if pred and ground_truth[pr_id])
    fp = sum(1 for pr_id, pred in results.items() if pred and not ground_truth[pr_id])
    fn = sum(1 for pr_id, pred in results.items() if not pred and ground_truth[pr_id])
    tn = sum(1 for pr_id, pred in results.items() if not pred and not ground_truth[pr_id])

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    print()
    print("=== Calibration Results ===")
    print(f"  TP: {tp}  FP: {fp}  FN: {fn}  TN: {tn}  Errors: {errors}")
    print(f"  Precision: {precision:.1%}  (target: >=80%)")
    print(f"  Recall:    {recall:.1%}  (target: >=70%)")
    print(f"  F1:        {f1:.1%}")
    print()

    # Show disagreements
    disagreements = [
        (pr_id, results[pr_id], ground_truth[pr_id])
        for pr_id in results
        if results[pr_id] != ground_truth[pr_id]
    ]
    if disagreements:
        print(f"Disagreements ({len(disagreements)}):")
        for pr_id, pred, truth in disagreements:
            label = "FP (predicted True, actually False)" if pred else "FN (predicted False, actually True)"
            print(f"  {pr_id}: {label}")

    if precision >= 0.80 and recall >= 0.70:
        print("\nTarget met — safe to run on all 629 PRs via Batch API.")
        print("  Run: python pipeline.py --skip-load --skip-classify --skip-collect --validate")
    else:
        print("\nTarget NOT met — adjust the prompt in analysis/pr_validator.py and re-run.")


if __name__ == "__main__":
    main()
