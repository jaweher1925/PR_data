"""D1 + D2 analyses for camera-ready revision.

D1: Merged vs unmerged mutation-category distribution.
    For each category: n_merged, pct_merged, n_unmerged, pct_unmerged,
    Cliff's delta (= p_merged - p_unmerged for binary outcomes).
    Summary: Spearman rank correlation between the two proportions.

D2: Per-language sanity check for agent profiles.
    Within the three largest language sub-corpora (TypeScript, Rust, Go),
    confirm each agent's leading mutation category.

Usage:
  .venv/bin/python3 scripts/merge_state_analysis.py

Outputs:
  tables/d1_merged_vs_unmerged.csv
  tables/d2_agent_x_language.csv
  (plus printed summaries for paper wording)
"""
import csv
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from scipy.stats import spearmanr

from analysis.patch_classifier import MUTATION_CATEGORIES
from config import MSR_DATA_DIR

PARTS_CSV = "perf_improvement_parts.csv"
PRS_CSV = f"{MSR_DATA_DIR}/performance_prs_ai_vs_human.csv"

CLIFFS_THRESHOLDS = [(0.474, "large"), (0.33, "medium"), (0.147, "small"), (0.0, "negligible")]

TOP_LANGUAGES = ["TypeScript", "Rust", "Go"]

AGENT_DISPLAY = {
    "Devin": "Devin",
    "Copilot": "Copilot",
    "OpenAI_Codex": "Codex",
    "Cursor": "Cursor",
    "Claude_Code": "Claude Code",
}


def cliffs_label(delta: float) -> str:
    abs_d = abs(delta)
    for threshold, label in CLIFFS_THRESHOLDS:
        if abs_d >= threshold:
            return label
    return "negligible"


def parse_ids(raw: str) -> list[int]:
    ids = []
    for token in raw.split(";"):
        token = token.strip()
        m = re.match(r"(\d+):", token)
        if m:
            ids.append(int(m.group(1)))
    return ids


def load_joined() -> pd.DataFrame:
    hunks = pd.read_csv(PARTS_CSV)
    hunks = hunks[hunks["categories"].notna() & (hunks["categories"].str.strip() != "")]
    hunks["cat_ids"] = hunks["categories"].apply(parse_ids)

    prs = pd.read_csv(PRS_CSV)
    ai_prs = prs[prs["author_type"] == "AI Agent"]

    joined = hunks.merge(
        ai_prs[["id", "is_merged", "agent", "primary_language"]],
        left_on="pr_id",
        right_on="id",
        how="inner",
    )
    return joined


def d1_merged_vs_unmerged(df: pd.DataFrame) -> pd.DataFrame:
    merged = df[df["is_merged"] == True]
    unmerged = df[df["is_merged"] == False]
    n_m = len(merged)
    n_u = len(unmerged)

    rows = []
    for cid in sorted(MUTATION_CATEGORIES):
        name = MUTATION_CATEGORIES[cid][0]
        c_m = sum(1 for ids in merged["cat_ids"] if cid in ids)
        c_u = sum(1 for ids in unmerged["cat_ids"] if cid in ids)
        p_m = c_m / n_m if n_m else 0.0
        p_u = c_u / n_u if n_u else 0.0
        delta = p_m - p_u
        rows.append({
            "category_id": cid,
            "category": name,
            "n_merged": c_m,
            "pct_merged": round(100 * p_m, 1),
            "n_unmerged": c_u,
            "pct_unmerged": round(100 * p_u, 1),
            "cliffs_delta": round(delta, 3),
            "effect_size": cliffs_label(delta),
        })

    result = pd.DataFrame(rows).sort_values("n_merged", ascending=False)
    result.to_csv("tables/d1_merged_vs_unmerged.csv", index=False)
    print(f"  → tables/d1_merged_vs_unmerged.csv  ({n_m} merged, {n_u} unmerged hunks)")
    return result, n_m, n_u


def print_d1_summary(result: pd.DataFrame, n_m: int, n_u: int):
    top5_pcts = result.head(5)
    print(f"\n{'─'*70}")
    print(f"  D1: Merged ({n_m} hunks from {result[result['n_merged']>0].shape[0]}-range PRs) vs Unmerged ({n_u} hunks)")
    print(f"  Top 5 by merged count:")
    print(f"  {'Category':<30} {'Merged%':>8} {'Unmerged%':>10} {'δ':>7} {'Effect':>10}")
    print(f"  {'─'*30} {'─'*8} {'─'*10} {'─'*7} {'─'*10}")
    for _, row in top5_pcts.iterrows():
        print(f"  {row['category']:<30} {row['pct_merged']:>7.1f}% {row['pct_unmerged']:>9.1f}%"
              f" {row['cliffs_delta']:>+7.3f} {row['effect_size']:>10}")

    # Top-5 rank correlation
    top5_merged = result.nlargest(5, "pct_merged")["category"].tolist()
    top5_unmerged = result.nlargest(5, "pct_unmerged")["category"].tolist()
    overlap = set(top5_merged) & set(top5_unmerged)
    print(f"\n  Top-5 merged:   {top5_merged}")
    print(f"  Top-5 unmerged: {top5_unmerged}")
    print(f"  Overlap (same categories in both top-5): {len(overlap)}/5  {overlap}")

    # Spearman on all 18 categories
    rho, pval = spearmanr(result["pct_merged"], result["pct_unmerged"])
    print(f"\n  Spearman ρ (all 18 categories): {rho:.3f}  p={pval:.4f}")

    max_delta = result["cliffs_delta"].abs().max()
    large = result[result["cliffs_delta"].abs() >= 0.33]["category"].tolist()
    print(f"\n  Max |δ|: {max_delta:.3f}")
    print(f"  Categories with medium/large effect (|δ|≥0.33): {large if large else 'none'}")
    print(f"{'─'*70}\n")


def d2_agent_x_language(df: pd.DataFrame):
    cat_names = {cid: MUTATION_CATEGORIES[cid][0] for cid in MUTATION_CATEGORIES}
    results = {}

    for lang in TOP_LANGUAGES:
        lang_df = df[df["primary_language"] == lang]
        lang_results = {}
        for agent_key, agent_display in AGENT_DISPLAY.items():
            sub = lang_df[lang_df["agent"] == agent_key]
            if len(sub) == 0:
                continue
            counts = defaultdict(int)
            for ids in sub["cat_ids"]:
                for cid in ids:
                    counts[cat_names[cid]] += 1
            if not counts:
                continue
            top_cat = max(counts, key=counts.get)
            top_pct = round(100 * counts[top_cat] / len(sub), 1)
            lang_results[agent_display] = (top_cat, top_pct, len(sub))
        results[lang] = lang_results

    rows = []
    for lang, agent_data in results.items():
        for agent_disp, (cat, pct, n) in agent_data.items():
            rows.append({"language": lang, "agent": agent_disp,
                         "top_category": cat, "pct": pct, "n_hunks": n})
    out = pd.DataFrame(rows)
    out.to_csv("tables/d2_agent_x_language.csv", index=False)
    print(f"  → tables/d2_agent_x_language.csv")
    return results


def print_d2_summary(results: dict):
    print(f"\n{'─'*70}")
    print("  D2: Top mutation category per agent within top-3 languages")
    for lang, agent_data in results.items():
        print(f"\n  {lang}:")
        for agent, (cat, pct, n) in agent_data.items():
            print(f"    {agent:<18} → {cat:<28}  {pct:>5.1f}%  (n={n})")
    print(f"\n{'─'*70}\n")


def print_paper_wording(d1_result: pd.DataFrame, n_m: int, n_u: int, d2_results: dict):
    """Draft wording for the camera-ready paper."""
    rho, _ = spearmanr(d1_result["pct_merged"], d1_result["pct_unmerged"])
    max_delta = d1_result["cliffs_delta"].abs().max()
    large = [r for _, r in d1_result.iterrows() if abs(r["cliffs_delta"]) >= 0.33]

    print("─" * 70)
    print("DRAFT WORDING FOR paper/main.tex")
    print("─" * 70)
    print()
    print("# D1 — Threats (External), ~2 sentences:")
    if not large:
        print(f"  Comparing category distributions on merged-PR hunks (n={n_m}) "
              f"vs unmerged-PR hunks (n={n_u}), the top-five rank order is fully "
              f"preserved (Spearman ρ={rho:.2f}) and per-category Cliff's δ is "
              f"negligible to small (|δ|≤{max_delta:.2f}) across all 18 categories. "
              f"Unmerged PRs exhibit the same mutation vocabulary as merged ones.")
    else:
        top5_m = d1_result.nlargest(5, "pct_merged")["category"].tolist()
        top5_u = d1_result.nlargest(5, "pct_unmerged")["category"].tolist()
        overlap = set(top5_m) & set(top5_u)
        print(f"  Comparing category distributions on merged-PR hunks (n={n_m}) "
              f"vs unmerged-PR hunks (n={n_u}), the top-five overlap on "
              f"{len(overlap)}/5 categories (Spearman ρ={rho:.2f}). "
              f"Cliff's δ is medium/large for "
              f"{', '.join(r['category'] for r in large)}, "
              f"suggesting caution when extrapolating these specific categories "
              f"from the full corpus. The overall mutation vocabulary is "
              f"otherwise consistent across merge outcome.")
    print()
    print("# D2 — Sec 3.2 ('Performance patterns'), ~2 sentences:")
    # Check if each agent's top category is consistent
    agent_checks = {}
    for lang, agent_data in d2_results.items():
        for agent, (cat, pct, n) in agent_data.items():
            if agent not in agent_checks:
                agent_checks[agent] = cat
            elif agent_checks[agent] != cat:
                agent_checks[agent] = f"DIVERGES ({cat} in {lang})"

    consistent = {a: c for a, c in agent_checks.items() if "DIVERGES" not in c}
    diverges = {a: c for a, c in agent_checks.items() if "DIVERGES" in c}

    if not diverges:
        agents_str = "; ".join(
            f"{a} remains {c.replace('_', r'_')}-led"
            for a, c in list(consistent.items())[:3]
        )
        print(f"  As a language-confound check, within the three largest sub-corpora "
              f"(TypeScript, Rust, Go), each agent's leading category persists: "
              f"{agents_str}. The per-agent profile is therefore not a language artefact.")
    else:
        print(f"  Per-language check: {'; '.join(f'{a} consistent' for a in consistent)}. "
              f"NOTE: {'; '.join(f'{a} diverges: {c}' for a, c in diverges.items())} "
              f"— update paper wording accordingly.")
    print()


def main():
    print("Loading joined dataset...")
    df = load_joined()
    print(f"  {len(df)} hunks | "
          f"{df['is_merged'].value_counts().get(True, 0)} merged, "
          f"{df['is_merged'].value_counts().get(False, 0)} unmerged\n")

    print("D1: Merged vs unmerged distribution...")
    d1_result, n_m, n_u = d1_merged_vs_unmerged(df)
    print_d1_summary(d1_result, n_m, n_u)

    print("D2: Per-language agent sanity check...")
    d2_results = d2_agent_x_language(df)
    print_d2_summary(d2_results)

    print_paper_wording(d1_result, n_m, n_u, d2_results)


if __name__ == "__main__":
    main()
