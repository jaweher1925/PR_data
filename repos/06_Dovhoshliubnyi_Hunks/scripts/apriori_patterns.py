"""
apriori_patterns.py — Result 2: Apriori co-occurrence pattern mining

Finds which mutation categories frequently appear together within the same PR.
Each PR is one transaction; categories across all its hunks are the items.

Outputs:
  tables/apriori_itemsets.csv   — frequent itemsets (support >= MIN_SUPPORT)
  tables/apriori_rules.csv      — association rules sorted by lift
  figs/apriori_top_rules.png    — scatter: confidence vs lift, sized by support
"""

import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent.parent
INPUT_CSV = ROOT / "perf_improvement_parts.csv"

OUT_ITEMSETS = ROOT / "tables" / "apriori_itemsets.csv"
OUT_RULES = ROOT / "tables" / "apriori_rules.csv"
OUT_FIG = ROOT / "figs" / "apriori_top_rules.png"

MIN_SUPPORT = 0.10      # itemset must appear in >=10% of PRs
MIN_CONFIDENCE = 0.3    # rule confidence threshold
MIN_LIFT = 1.0          # only rules with lift > 1 (positive correlation)
TOP_N_RULES = 20        # rules shown in figure

CATEGORY_NAMES = {
    0: "added_code_from_github",
    1: "no_change",
    2: "comment_modification",
    3: "deleted_blocks",
    4: "duplicate_code",
    5: "return_statement_mods",
    6: "method_name_change",
    7: "type_change",
    8: "inlining",
    9: "exception_handling",
    10: "extra_brackets",
    11: "synchronization",
    12: "name_modification",
    13: "control_flow",
    14: "object_creation",
    15: "statement_splitting",
    16: "arithmetic_manipulation",
    17: "dead_code",
}

def parse_category_ids(raw: str) -> set[int]:
    if not isinstance(raw, str) or not raw.strip():
        return set()
    ids = set()
    for token in raw.split(";"):
        token = token.strip()
        m = re.match(r"(\d+):", token)
        if m:
            ids.add(int(m.group(1)))
    return ids


def build_transaction_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Return binary DataFrame: rows=PRs, columns=category names."""
    # Aggregate all category IDs per PR across all its hunks
    pr_cats: dict[int, set[int]] = {}
    for _, row in df.iterrows():
        pr_id = row["pr_id"]
        ids = parse_category_ids(row["categories"])
        pr_cats.setdefault(pr_id, set()).update(ids)

    # Build one-hot matrix
    all_ids = sorted(CATEGORY_NAMES.keys())
    records = []
    for pr_id, cat_set in pr_cats.items():
        records.append({CATEGORY_NAMES[i]: (i in cat_set) for i in all_ids})

    matrix = pd.DataFrame(records)
    # Drop columns that never appear (e.g. extra_brackets = 0 everywhere)
    return matrix.loc[:, matrix.any()]


def main() -> None:
    try:
        from mlxtend.frequent_patterns import association_rules, fpgrowth
    except ImportError:
        print("mlxtend not installed. Run: pip install mlxtend", file=sys.stderr)
        sys.exit(1)

    if not INPUT_CSV.exists():
        print(f"Input not found: {INPUT_CSV}", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(INPUT_CSV, low_memory=False)
    n_prs = df["pr_id"].nunique()
    print(f"Loaded {len(df):,} hunks across {n_prs} PRs")

    matrix = build_transaction_matrix(df)
    print(f"Transaction matrix: {len(matrix)} PRs × {len(matrix.columns)} categories")

    # Frequent itemsets (single call — reused for both export and rules)
    raw_itemsets = fpgrowth(matrix, min_support=MIN_SUPPORT, use_colnames=True)
    export_df = raw_itemsets.copy()
    export_df["size"] = export_df["itemsets"].apply(len)
    export_df["itemset"] = export_df["itemsets"].apply(lambda s: ", ".join(sorted(s)))
    export_df = export_df[["itemset", "support", "size"]].sort_values(
        ["size", "support"], ascending=[False, False]
    ).reset_index(drop=True)

    OUT_ITEMSETS.parent.mkdir(exist_ok=True)
    export_df.to_csv(OUT_ITEMSETS, index=False)
    print(f"Wrote {len(export_df)} itemsets → {OUT_ITEMSETS}")

    # Association rules
    if len(raw_itemsets) < 2:
        print("Too few itemsets for association rules at current min_support.")
        return

    rules = association_rules(raw_itemsets, metric="lift", min_threshold=MIN_LIFT)
    rules = rules[rules["confidence"] >= MIN_CONFIDENCE].copy()
    rules["antecedents"] = rules["antecedents"].apply(lambda s: ", ".join(sorted(s)))
    rules["consequents"] = rules["consequents"].apply(lambda s: ", ".join(sorted(s)))
    rules = rules[["antecedents", "consequents", "support", "confidence", "lift"]].sort_values(
        "lift", ascending=False
    ).reset_index(drop=True)

    rules.to_csv(OUT_RULES, index=False)
    print(f"Wrote {len(rules)} rules → {OUT_RULES}")

    # Figure: top N rules, confidence vs lift, bubble size = support
    _plot_rules(rules.head(TOP_N_RULES))


def _plot_rules(rules: pd.DataFrame) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping figure.")
        return

    if rules.empty:
        print("No rules to plot.")
        return

    fig, ax = plt.subplots(figsize=(9, 6))
    scatter = ax.scatter(
        rules["confidence"],
        rules["lift"],
        s=rules["support"] * 2000,
        alpha=0.6,
        edgecolors="black",
        linewidths=0.5,
        c=rules["lift"],
        cmap="YlOrRd",
    )
    plt.colorbar(scatter, ax=ax, label="Lift")

    # Label each point with "A → C"
    for _, row in rules.iterrows():
        label = f"{row['antecedents']} → {row['consequents']}"
        # Truncate long labels
        if len(label) > 50:
            label = label[:47] + "..."
        ax.annotate(
            label,
            (row["confidence"], row["lift"]),
            fontsize=6,
            xytext=(4, 4),
            textcoords="offset points",
        )

    ax.set_xlabel("Confidence")
    ax.set_ylabel("Lift")
    ax.set_title(f"Top {len(rules)} Association Rules (bubble size ∝ support)")
    ax.axhline(1.0, color="grey", linestyle="--", linewidth=0.8, label="Lift = 1")
    ax.legend(fontsize=8)
    plt.tight_layout()

    OUT_FIG.parent.mkdir(exist_ok=True)
    fig.savefig(OUT_FIG, dpi=150)
    plt.close(fig)
    print(f"Wrote figure → {OUT_FIG}")


if __name__ == "__main__":
    main()
