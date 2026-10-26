"""Compute inter-rater agreement (Fleiss κ) across 3 rater CSVs.

Usage:
  .venv/bin/python3 scripts/compute_agreement.py \\
      validation_rater_A.csv validation_rater_B.csv validation_rater_C.csv

Each input CSV must have:
  - part_id  (int, links rows across files)
  - categories  (comma-separated category IDs, e.g. "5,13" or blank)

Outputs:
  - Per-category Fleiss κ (treating each category as binary present/absent)
  - Overall macro-average κ
  - Exact-match agreement rate (all raters agree on identical category set)
  - agreement_results.csv
"""
import csv
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from analysis.patch_classifier import MUTATION_CATEGORIES

OUTPUT_CSV = "agreement_results.csv"
ALL_CATS = sorted(MUTATION_CATEGORIES.keys())  # 0-17


# ── Parsing ───────────────────────────────────────────────────────────────────

def parse_categories(raw: str) -> set[int]:
    """Parse a comma-separated category string like '5,13' into a set of ints."""
    result = set()
    for token in raw.replace(";", ",").split(","):
        token = token.strip()
        if token.isdigit():
            result.add(int(token))
    return result


def load_rater(path: str) -> dict[int, set[int]]:
    """Load {part_id: set_of_category_ids} from a rater CSV."""
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    result = {}
    for row in rows:
        pid = int(row["part_id"])
        result[pid] = parse_categories(row.get("categories", ""))
    return result


# ── Fleiss κ for a single binary category ────────────────────────────────────

def fleiss_kappa_binary(ratings: list[list[int]]) -> float:
    """
    Compute Fleiss κ for a binary item across n raters.

    ratings: list of lists, shape (n_items, n_raters), each value 0 or 1.
    Returns κ (float), or None if undefined (all raters agree on same value always).
    """
    n_items = len(ratings)
    n_raters = len(ratings[0]) if ratings else 0
    if n_items == 0 or n_raters < 2:
        return float("nan")

    # P̄: mean proportion of agreeing rater pairs per item
    p_bar = 0.0
    for item_ratings in ratings:
        yes = sum(item_ratings)
        no = n_raters - yes
        p_i = (yes * (yes - 1) + no * (no - 1)) / (n_raters * (n_raters - 1))
        p_bar += p_i
    p_bar /= n_items

    # P̄_e: expected agreement under independence
    p_yes = sum(sum(r) for r in ratings) / (n_items * n_raters)
    p_no = 1.0 - p_yes
    p_e = p_yes ** 2 + p_no ** 2

    if abs(1.0 - p_e) < 1e-10:
        return float("nan")  # undefined: all raters always agree

    return (p_bar - p_e) / (1.0 - p_e)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) < 4:
        print("Usage: compute_agreement.py rater_A.csv rater_B.csv rater_C.csv")
        sys.exit(1)

    rater_paths = sys.argv[1:4]
    raters = [load_rater(p) for p in rater_paths]
    rater_names = [Path(p).stem for p in rater_paths]

    # Find common part_ids
    common_ids = sorted(
        set(raters[0].keys()) & set(raters[1].keys()) & set(raters[2].keys())
    )
    if not common_ids:
        print("ERROR: No common part_ids found across the three files.")
        sys.exit(1)

    print(f"\nRaters: {', '.join(rater_names)}")
    print(f"Common items: {len(common_ids)}\n")

    # ── Per-category Fleiss κ ──────────────────────────────────────────────
    kappas = {}
    for cat_id in ALL_CATS:
        binary_ratings = [
            [1 if cat_id in rater.get(pid, set()) else 0 for rater in raters]
            for pid in common_ids
        ]
        kappas[cat_id] = fleiss_kappa_binary(binary_ratings)

    # ── Exact-match agreement ──────────────────────────────────────────────
    exact_matches = sum(
        1 for pid in common_ids
        if len({frozenset(r.get(pid, set())) for r in raters}) == 1
    )
    exact_rate = exact_matches / len(common_ids)

    # ── Print summary ──────────────────────────────────────────────────────
    valid_kappas = [k for k in kappas.values() if k == k]  # exclude NaN
    macro_kappa = sum(valid_kappas) / len(valid_kappas) if valid_kappas else float("nan")

    print(f"{'Category':<32} {'Fleiss κ':>9}  Interpretation")
    print("─" * 62)
    for cat_id in ALL_CATS:
        k = kappas[cat_id]
        name = MUTATION_CATEGORIES[cat_id][0]
        if k != k:  # NaN
            interp = "(all agree — trivially)"
        elif k < 0:
            interp = "poor (worse than chance)"
        elif k < 0.2:
            interp = "slight"
        elif k < 0.4:
            interp = "fair"
        elif k < 0.6:
            interp = "moderate"
        elif k < 0.8:
            interp = "substantial"
        else:
            interp = "almost perfect"
        k_str = f"{k:.3f}" if k == k else "  n/a"
        print(f"  {cat_id:2d}  {name:<28} {k_str:>9}  {interp}")

    print("─" * 62)
    print(f"  Macro-average κ (excl. trivial): {macro_kappa:.3f}")
    print(f"  Exact-match rate:                {exact_rate:.1%}  ({exact_matches}/{len(common_ids)})")

    if macro_kappa >= 0.6:
        print("\n  ✓ κ ≥ 0.6 — substantial agreement, suitable for paper.")
    elif macro_kappa >= 0.4:
        print("\n  ~ κ 0.4–0.6 — moderate agreement. Consider reconciliation pass.")
    else:
        print("\n  ✗ κ < 0.4 — low agreement. Discuss disagreements with raters.")

    # ── Write CSV ──────────────────────────────────────────────────────────
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["category_id", "category_name", "fleiss_kappa"])
        writer.writeheader()
        for cat_id in ALL_CATS:
            k = kappas[cat_id]
            writer.writerow({
                "category_id": cat_id,
                "category_name": MUTATION_CATEGORIES[cat_id][0],
                "fleiss_kappa": f"{k:.4f}" if k == k else "nan",
            })
    print(f"\n  Wrote per-category κ → {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
