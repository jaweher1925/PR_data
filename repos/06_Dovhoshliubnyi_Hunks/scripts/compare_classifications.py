"""Compare Anthropic vs OpenAI 18-category classifications and report disagreements.

Run after both --classify and --classify-openai are complete.

Usage:
  .venv/bin/python3 scripts/compare_classifications.py [--out disagreements.csv] [--apply-partial]

Flags:
  --apply-partial   For partial-agreement hunks, keep only the intersection of
                    Anthropic and OpenAI category IDs (using Anthropic's category
                    objects) and write the result back to perf_improvement_progress.json.

Outputs:
  - Summary table printed to console
  - disagreements.csv  (only full_disagree rows)

Agreement types:
  full_agree     — Anthropic IDs == OpenAI IDs (including both empty)
  partial        — non-empty intersection but sets differ
  full_disagree  — no intersection (incl. one or both sides empty while other is not)
"""
import csv
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from analysis.patch_classifier import MUTATION_CATEGORIES

PROGRESS_FILE = "perf_improvement_progress.json"
OPENAI_FILE = "openai_classifications.json"
DEFAULT_OUT = "disagreements.csv"


def _part_key(part: dict) -> str:
    return f"{part['pr_id']}:{part['commit_sha']}:{part['filename']}:{part['hunk_index']}"


def _ids_from_cats(cats: list) -> set[int]:
    return {c["id"] for c in cats if isinstance(c.get("id"), int)}


def _fmt_cats(cats: list) -> str:
    if not cats:
        return ""
    return "; ".join(
        f"{c['id']}:{c.get('name', '')}({c.get('confidence', 0):.2f})"
        + (f"@L{c['start_line']}-{c['end_line']}" if c.get("start_line") else "")
        for c in cats
    )


def agreement_type(ant_ids: set, oai_ids: set) -> str:
    if ant_ids == oai_ids:
        return "full_agree"
    if ant_ids & oai_ids:
        return "partial"
    return "full_disagree"


def main():
    out_path = DEFAULT_OUT
    apply_partial = False
    args = sys.argv[1:]
    for i, arg in enumerate(args):
        if arg == "--out" and i + 1 < len(args):
            out_path = args[i + 1]
        elif arg == "--apply-partial":
            apply_partial = True

    if not os.path.exists(PROGRESS_FILE):
        print(f"ERROR: {PROGRESS_FILE} not found. Run --classify first.")
        sys.exit(1)
    if not os.path.exists(OPENAI_FILE):
        print(f"ERROR: {OPENAI_FILE} not found. Run --classify-openai first.")
        sys.exit(1)

    with open(PROGRESS_FILE, encoding="utf-8") as f:
        progress = json.load(f)
    with open(OPENAI_FILE, encoding="utf-8") as f:
        openai_data = json.load(f)

    parts = progress.get("accepted_parts", [])

    # Build Anthropic lookup: key → (part_metadata, categories)
    anthropic_lookup: dict[str, tuple[dict, list]] = {}
    for i, part in enumerate(parts, 1):
        key = _part_key(part)
        anthropic_lookup[key] = (i, part, part.get("categories") or [])

    counts = {"full_agree": 0, "partial": 0, "full_disagree": 0}
    only_anthropic = 0
    only_openai = 0
    disagreement_rows = []
    partial_updates: list[tuple[int, list]] = []  # (parts index, new categories)

    for key, (part_id, part, ant_cats) in anthropic_lookup.items():
        if key not in openai_data:
            only_anthropic += 1
            continue

        oai_entry = openai_data[key]
        oai_cats = oai_entry.get("categories") or []

        ant_ids = _ids_from_cats(ant_cats)
        oai_ids = _ids_from_cats(oai_cats)
        atype = agreement_type(ant_ids, oai_ids)
        counts[atype] += 1

        ant_reasoning = part.get("categories_reasoning", "")
        oai_reasoning = oai_entry.get("reasoning", "")

        if atype == "partial":
            intersection = ant_ids & oai_ids
            intersection_cats = [c for c in ant_cats if c.get("id") in intersection]
            partial_updates.append((part_id - 1, intersection_cats))
            disagreement_rows.append({
                "part_id": part_id,
                "pr_id": part["pr_id"],
                "pr_url": part["pr_url"],
                "filename": part["filename"],
                "hunk_index": part["hunk_index"],
                "final_sub_pattern": part.get("final_sub_pattern", ""),
                "categories_anthropic": _fmt_cats(ant_cats),
                "reasoning_anthropic": ant_reasoning,
                "categories_openai": _fmt_cats(oai_cats),
                "reasoning_openai": oai_reasoning,
                "agreement_type": atype,
                "intersection_categories": _fmt_cats(intersection_cats),
                "hunk_content": part["hunk_content"],
            })
        elif atype == "full_disagree":
            disagreement_rows.append({
                "part_id": part_id,
                "pr_id": part["pr_id"],
                "pr_url": part["pr_url"],
                "filename": part["filename"],
                "hunk_index": part["hunk_index"],
                "final_sub_pattern": part.get("final_sub_pattern", ""),
                "categories_anthropic": _fmt_cats(ant_cats),
                "reasoning_anthropic": ant_reasoning,
                "categories_openai": _fmt_cats(oai_cats),
                "reasoning_openai": oai_reasoning,
                "agreement_type": atype,
                "intersection_categories": "",
                "hunk_content": part["hunk_content"],
            })

    for key in openai_data:
        if key not in anthropic_lookup:
            only_openai += 1

    total_compared = sum(counts.values())
    total = total_compared + only_anthropic + only_openai

    def pct(n):
        return f"{100 * n / total_compared:.1f}%" if total_compared else "n/a"

    print(f"\n{'─'*55}")
    print(f"  Classification comparison: Anthropic vs OpenAI")
    print(f"{'─'*55}")
    print(f"  Total Anthropic hunks:    {len(anthropic_lookup)}")
    print(f"  Total OpenAI hunks:       {len(openai_data)}")
    print(f"  Compared (both present):  {total_compared}")
    print(f"  Only in Anthropic:        {only_anthropic}")
    print(f"  Only in OpenAI:           {only_openai}")
    print(f"{'─'*55}")
    print(f"  Full agreement:           {counts['full_agree']:>6}  ({pct(counts['full_agree'])})")
    print(f"  Partial agreement:        {counts['partial']:>6}  ({pct(counts['partial'])})")
    print(f"  Full disagreement:        {counts['full_disagree']:>6}  ({pct(counts['full_disagree'])})")
    print(f"{'─'*55}\n")

    if apply_partial and partial_updates:
        for parts_idx, intersection_cats in partial_updates:
            parts[parts_idx]["categories"] = intersection_cats
        with open(PROGRESS_FILE, "w", encoding="utf-8") as f:
            json.dump(progress, f, ensure_ascii=False)
        print(f"  Applied intersection categories to {len(partial_updates)} partial-agreement hunks → {PROGRESS_FILE}")

    full_disagree_rows = [r for r in disagreement_rows if r["agreement_type"] == "full_disagree"]
    fieldnames = [
        "part_id", "pr_id", "pr_url", "filename", "hunk_index",
        "final_sub_pattern", "categories_anthropic", "reasoning_anthropic",
        "categories_openai", "reasoning_openai",
        "agreement_type", "intersection_categories", "hunk_content",
    ]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(full_disagree_rows)

    print(f"  Wrote {len(full_disagree_rows)} full-disagree rows → {out_path}")
    if not apply_partial and counts["partial"]:
        print(f"  {counts['partial']} partial-agreement hunks not yet resolved — run with --apply-partial to accept intersections")


if __name__ == "__main__":
    main()
