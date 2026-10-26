"""Produce all tables and figures for the paper from perf_improvement_parts.csv.

Usage:
  .venv/bin/python3 scripts/analysis.py

Outputs:
  tables/category_freq.csv          — Table 1: per-category count + %
  tables/category_x_subpattern.csv  — 18 × 33 co-occurrence matrix (sub-pattern level)
  tables/category_x_pattern.csv     — 18 × 9 co-occurrence matrix (pattern level)
  tables/category_x_agent.csv       — 18 × 5 agents breakdown
  figs/category_dist.png            — Bar chart of category frequencies
  figs/heatmap_cat_subpattern.png   — Heatmap of category × sub-pattern
  figs/heatmap_cat_pattern.png      — Heatmap of category × performance pattern (grouped)
"""
import csv
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from analysis.patch_classifier import MUTATION_CATEGORIES
from config import MSR_DATA_DIR

PARTS_CSV = "perf_improvement_parts.csv"

# Mapping from AIDev sub-pattern to the 9 higher-level performance patterns
SUB_PATTERN_TO_PATTERN: dict[str, str] = {
    # Algorithm-Level Optimizations
    "Event-Driven Throttling":                   "Algorithm-Level",
    "Select Algorithm Based on Instruction Speed":"Algorithm-Level",
    "Select Computationally Efficient Algorithms":"Algorithm-Level",
    # Build & Compilation & Infrastructure
    "Compiler Optimization Hints/Directives":    "Build & Infrastructure",
    "Enable Build-Time Optimizations":           "Build & Infrastructure",
    "Enable Lazy loading for resource":          "Build & Infrastructure",
    "Performance-Optimized Dependency Selection":"Build & Infrastructure",
    "Reduce rebuilding size":                    "Build & Infrastructure",
    # Code Smells and Structural Simplification
    "Remove Unnecessary Method Calls":           "Code Smells",
    "Remove code bloat by removing optional features": "Code Smells",
    "Remove duplicate code":                     "Code Smells",
    "Remove long method by extracting new method":"Code Smells",
    "Replace Reflection with Direct Method Calls":"Code Smells",
    # Control-Flow and Branching Optimizations
    "Combining branches":                        "Control-Flow & Branching",
    "Rearranging branches":                      "Control-Flow & Branching",
    "Rearranging for early return":              "Control-Flow & Branching",
    "Remove branches by doing extra work":       "Control-Flow & Branching",
    # Data Structure Selection and Adaptation
    "Darwinian Data Structure Selection":        "Data Structure",
    # I/O and Synchronization
    "Concurrency Control / Lock Optimization":   "I/O & Sync",
    "Concurrency Control / Rate Limiting":       "I/O & Sync",
    "Enable Multi-threading":                    "I/O & Sync",
    "Non-Blocking I/O":                          "I/O & Sync",
    "Polling":                                   "I/O & Sync",
    "Selection I/O size":                        "I/O & Sync",
    "Streaming Implementation Or Optimization":  "I/O & Sync",
    # Loop Transformations
    "Loop Distribution (Fission)":               "Loop Transforms",
    "Loop Invariant Branches":                   "Loop Transforms",
    # Memory and Data Locality
    "Buffering":                                 "Memory & Locality",
    "Cache Eviction Policy":                     "Memory & Locality",
    "Caching":                                   "Memory & Locality",
    "Improve cache locality - spatial locality": "Memory & Locality",
    "Improve cache locality - temporal locality":"Memory & Locality",
    "Optimize Object Use":                       "Memory & Locality",
    "Use Smaller Data Types":                    "Memory & Locality",
    # Network, Database, and Data Access
    "Batch API Requests (N+1)":                  "Network & DB",
    "Lazy Loading / Deferred Initialization":    "Network & DB",
    "Minimize Payload Size / Pass Reference Instead of Value": "Network & DB",
    "Relational Query Consolidation (N+1)":      "Network & DB",
}

os.makedirs("tables", exist_ok=True)
os.makedirs("figs", exist_ok=True)


# ── Parsing ───────────────────────────────────────────────────────────────────

def parse_category_ids(raw: str) -> list[int]:
    """Extract integer category IDs from formatted string like '5:return_...(0.90)@L3-5; 13:...'"""
    return [int(m) for m in re.findall(r"^(\d+):|;\s*(\d+):", raw)
            for m in m if m]


def _parse_ids(raw: str) -> list[int]:
    """Simpler: match every leading ID in each semicolon-separated token."""
    ids = []
    for token in raw.split(";"):
        token = token.strip()
        m = re.match(r"(\d+):", token)
        if m:
            ids.append(int(m.group(1)))
    return ids


# ── Load data ─────────────────────────────────────────────────────────────────

def load_parts() -> pd.DataFrame:
    if not os.path.exists(PARTS_CSV):
        print(f"ERROR: {PARTS_CSV} not found. Run --export first.")
        sys.exit(1)
    df = pd.read_csv(PARTS_CSV)
    df = df[df["categories"].notna() & (df["categories"].str.strip() != "")]
    df["cat_ids"] = df["categories"].apply(_parse_ids)
    return df


def load_agent_map() -> dict[int, str]:
    """Return {pr_id: agent_name} from performance_prs_ai_vs_human.csv."""
    path = os.path.join(MSR_DATA_DIR, "performance_prs_ai_vs_human.csv")
    if not os.path.exists(path):
        return {}
    with open(path, newline="", encoding="utf-8") as f:
        return {int(r["id"]): r["agent"] for r in csv.DictReader(f) if r.get("agent")}


# ── Tables ────────────────────────────────────────────────────────────────────

def table_category_freq(df: pd.DataFrame) -> pd.DataFrame:
    counts = defaultdict(int)
    total_hunks = len(df)
    for ids in df["cat_ids"]:
        for cid in ids:
            counts[cid] += 1

    rows = []
    for cid in sorted(MUTATION_CATEGORIES):
        n = counts[cid]
        rows.append({
            "category_id": cid,
            "category_name": MUTATION_CATEGORIES[cid][0],
            "count": n,
            "pct_of_hunks": round(100 * n / total_hunks, 1) if total_hunks else 0,
        })
    result = pd.DataFrame(rows).sort_values("count", ascending=False)
    result.to_csv("tables/category_freq.csv", index=False)
    print(f"  → tables/category_freq.csv  ({total_hunks} total hunks)")
    return result


def table_category_x_subpattern(df: pd.DataFrame) -> pd.DataFrame:
    subpatterns = sorted(df["final_sub_pattern"].dropna().unique())
    cat_names = {cid: MUTATION_CATEGORIES[cid][0] for cid in MUTATION_CATEGORIES}

    matrix = defaultdict(lambda: defaultdict(int))
    for _, row in df.iterrows():
        sp = row["final_sub_pattern"]
        for cid in row["cat_ids"]:
            matrix[cat_names[cid]][sp] += 1

    result = pd.DataFrame(matrix).T.fillna(0).astype(int)
    # Ensure all sub-patterns present even if 0
    for sp in subpatterns:
        if sp not in result.columns:
            result[sp] = 0
    result = result[sorted(result.columns)]
    result.index.name = "category"
    result.to_csv("tables/category_x_subpattern.csv")
    print(f"  → tables/category_x_subpattern.csv  ({len(result)} categories × {len(result.columns)} sub-patterns)")
    return result


def table_category_x_agent(df: pd.DataFrame, agent_map: dict) -> pd.DataFrame:
    df = df.copy()
    df["agent"] = df["pr_id"].map(agent_map).fillna("unknown")
    agents = sorted(df["agent"].unique())
    cat_names = {cid: MUTATION_CATEGORIES[cid][0] for cid in MUTATION_CATEGORIES}

    matrix = defaultdict(lambda: defaultdict(int))
    for _, row in df.iterrows():
        agent = row["agent"]
        for cid in row["cat_ids"]:
            matrix[cat_names[cid]][agent] += 1

    result = pd.DataFrame(matrix).T.fillna(0).astype(int)
    for a in agents:
        if a not in result.columns:
            result[a] = 0
    result = result[sorted(result.columns)]
    result.index.name = "category"
    result.to_csv("tables/category_x_agent.csv")
    print(f"  → tables/category_x_agent.csv  ({len(result)} categories × {len(agents)} agents)")
    return result


# ── Figures ───────────────────────────────────────────────────────────────────

def fig_category_dist(freq_df: pd.DataFrame):
    top = freq_df[freq_df["count"] > 0].head(18)
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(top["category_name"][::-1], top["count"][::-1], color="#4878CF")
    ax.set_xlabel("Number of accepted hunks")
    ax.set_title("Mutation category frequency in performance-improving hunks")
    ax.bar_label(bars, padding=3, fontsize=8)
    plt.tight_layout()
    plt.savefig("figs/category_dist.png", dpi=150)
    plt.close()
    print("  → figs/category_dist.png")


def fig_heatmap_cat_subpattern(matrix_df: pd.DataFrame):
    if matrix_df.empty:
        return
    # Keep only columns (sub-patterns) with at least 1 occurrence
    matrix_df = matrix_df.loc[:, (matrix_df > 0).any(axis=0)]
    # Shorten long sub-pattern names for readability
    matrix_df.columns = [c[:28] + "…" if len(c) > 28 else c for c in matrix_df.columns]

    fig_h = max(6, len(matrix_df) * 0.4)
    fig_w = max(12, len(matrix_df.columns) * 0.5)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    sns.heatmap(
        matrix_df,
        ax=ax,
        cmap="YlOrRd",
        linewidths=0.3,
        linecolor="lightgray",
        annot=True,
        fmt="d",
        annot_kws={"size": 7},
        cbar_kws={"label": "Hunk count"},
    )
    ax.set_title("Mutation category × performance sub-pattern co-occurrence")
    ax.set_xlabel("Performance sub-pattern")
    ax.set_ylabel("Mutation category")
    plt.xticks(rotation=45, ha="right", fontsize=7)
    plt.yticks(fontsize=8)
    plt.tight_layout()
    plt.savefig("figs/heatmap_cat_subpattern.png", dpi=150)
    plt.close()
    print("  → figs/heatmap_cat_subpattern.png")


# ── Pattern-level (grouped) table and figure ──────────────────────────────────

def table_category_x_pattern(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate the sub-pattern co-occurrence matrix into the 9 performance patterns."""
    cat_names = {cid: MUTATION_CATEGORIES[cid][0] for cid in MUTATION_CATEGORIES}

    # Canonical pattern order for the figure
    pattern_order = [
        "Algorithm-Level",
        "Build & Infrastructure",
        "Code Smells",
        "Control-Flow & Branching",
        "Data Structure",
        "I/O & Sync",
        "Loop Transforms",
        "Memory & Locality",
        "Network & DB",
    ]

    matrix = defaultdict(lambda: defaultdict(int))
    for _, row in df.iterrows():
        sp = row["final_sub_pattern"]
        pattern = SUB_PATTERN_TO_PATTERN.get(sp)
        if pattern is None:
            continue
        for cid in row["cat_ids"]:
            matrix[cat_names[cid]][pattern] += 1

    result = pd.DataFrame(matrix).T.fillna(0).astype(int)
    for p in pattern_order:
        if p not in result.columns:
            result[p] = 0
    # Keep only patterns that actually appear
    present = [p for p in pattern_order if p in result.columns and result[p].sum() > 0]
    result = result[present]
    # Drop rows (categories) with zero across all patterns
    result = result[result.sum(axis=1) > 0]
    result.index.name = "category"
    result.to_csv("tables/category_x_pattern.csv")
    print(f"  → tables/category_x_pattern.csv  ({len(result)} categories × {len(result.columns)} patterns)")
    return result


def fig_heatmap_cat_pattern(matrix_df: pd.DataFrame):
    """Produce a readable 18×9 heatmap grouped by performance pattern."""
    if matrix_df.empty:
        return

    # Sort rows by total descending so most common categories are at top
    matrix_df = matrix_df.loc[matrix_df.sum(axis=1).sort_values(ascending=True).index]

    fig_h = max(5, len(matrix_df) * 0.45)
    fig_w = max(9, len(matrix_df.columns) * 1.1)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    sns.heatmap(
        matrix_df,
        ax=ax,
        cmap="YlOrRd",
        linewidths=0.5,
        linecolor="lightgray",
        annot=True,
        fmt="d",
        annot_kws={"size": 9},
        cbar_kws={"label": "Hunk count"},
    )
    ax.set_title("Mutation category × performance pattern co-occurrence", fontsize=11)
    ax.set_xlabel("Performance pattern", fontsize=10)
    ax.set_ylabel("Mutation category", fontsize=10)
    plt.xticks(rotation=30, ha="right", fontsize=9)
    plt.yticks(fontsize=9)
    plt.tight_layout()
    plt.savefig("figs/heatmap_cat_pattern.png", dpi=150)
    plt.close()
    print("  → figs/heatmap_cat_pattern.png")


# ── Summary stats ─────────────────────────────────────────────────────────────

def print_summary(df: pd.DataFrame, freq_df: pd.DataFrame, agent_map: dict):
    total_hunks = len(df)
    total_prs = df["pr_id"].nunique()
    cats_used = freq_df[freq_df["count"] > 0]["category_name"].tolist()
    top3 = freq_df.head(3)["category_name"].tolist()

    df_agent = df.copy()
    df_agent["agent"] = df_agent["pr_id"].map(agent_map).fillna("unknown")
    agents = df_agent["agent"].value_counts().to_dict()

    print(f"\n{'─'*55}")
    print(f"  Dataset summary")
    print(f"{'─'*55}")
    print(f"  Accepted hunks:         {total_hunks}")
    print(f"  Unique PRs:             {total_prs}")
    print(f"  Categories observed:    {len(cats_used)}/18")
    print(f"  Top 3 categories:       {', '.join(top3)}")
    print(f"  Agents:")
    for agent, cnt in sorted(agents.items(), key=lambda x: -x[1]):
        print(f"    {agent:<30} {cnt} hunks")
    print(f"{'─'*55}\n")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    print("Loading data...")
    df = load_parts()
    agent_map = load_agent_map()
    print(f"  {len(df)} classified hunks from {df['pr_id'].nunique()} PRs\n")

    print("Generating tables...")
    freq_df = table_category_freq(df)
    matrix_df = table_category_x_subpattern(df)
    pattern_df = table_category_x_pattern(df)
    table_category_x_agent(df, agent_map)

    print("\nGenerating figures...")
    fig_category_dist(freq_df)
    fig_heatmap_cat_subpattern(matrix_df)
    fig_heatmap_cat_pattern(pattern_df)

    print_summary(df, freq_df, agent_map)
    print("Done. See tables/ and figs/ directories.")


if __name__ == "__main__":
    main()
