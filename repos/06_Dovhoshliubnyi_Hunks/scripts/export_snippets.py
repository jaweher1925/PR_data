"""
export_snippets.py — Result 3: per-PR category snippet export

For each hunk in perf_improvement_parts.csv, emits one row per assigned category
including the actual diff content. Also produces a per-category top-3-by-confidence
sample for use as paper examples.

Outputs:
  tables/pr_snippets.csv        — full flat table
  tables/pr_snippets_sample.csv — top 3 per category by confidence
"""

import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent.parent
INPUT_CSV = ROOT / "perf_improvement_parts.csv"
OUT_FULL = ROOT / "tables" / "pr_snippets.csv"
OUT_SAMPLE = ROOT / "tables" / "pr_snippets_sample.csv"

# e.g. "2:comment_modification(0.95)@L4-4"
CAT_PATTERN = re.compile(
    r"(\d+):([a-z_]+)\(([0-9.]+)\)@(L[\d]+-[\d]+)"
)


def parse_categories(raw: str) -> list[dict]:
    """Return list of {category_id, category_name, confidence, line_range}."""
    if not isinstance(raw, str) or not raw.strip():
        return []
    entries = []
    for part in raw.split(";"):
        part = part.strip()
        m = CAT_PATTERN.match(part)
        if m:
            entries.append({
                "category_id": int(m.group(1)),
                "category_name": m.group(2),
                "confidence": float(m.group(3)),
                "line_range": m.group(4),
            })
    return entries


def main() -> None:
    if not INPUT_CSV.exists():
        print(f"Input not found: {INPUT_CSV}", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(INPUT_CSV, low_memory=False)
    required = {"pr_id", "pr_url", "final_sub_pattern", "hunk_index",
                "filename", "categories", "hunk_content"}
    missing = required - set(df.columns)
    if missing:
        print(f"Missing columns: {missing}", file=sys.stderr)
        sys.exit(1)

    rows = []
    for _, hunk in df.iterrows():
        parsed = parse_categories(hunk["categories"])
        if not parsed:
            continue
        for cat in parsed:
            rows.append({
                "pr_id": hunk["pr_id"],
                "pr_url": hunk["pr_url"],
                "final_sub_pattern": hunk["final_sub_pattern"],
                "hunk_index": hunk["hunk_index"],
                "filename": hunk["filename"],
                "category_id": cat["category_id"],
                "category_name": cat["category_name"],
                "confidence": cat["confidence"],
                "line_range": cat["line_range"],
                "hunk_content": hunk["hunk_content"],
            })

    if not rows:
        print("No rows produced — check categories column format.", file=sys.stderr)
        sys.exit(1)

    full = pd.DataFrame(rows)
    OUT_FULL.parent.mkdir(exist_ok=True)
    full.to_csv(OUT_FULL, index=False)
    print(f"Wrote {len(full):,} rows → {OUT_FULL}")

    # Top 3 per category by confidence, keeping only non-empty hunk_content
    sample = (
        full[full["hunk_content"].notna() & (full["hunk_content"].str.strip() != "")]
        .sort_values("confidence", ascending=False)
        .groupby("category_id", sort=False)
        .head(3)
        .sort_values(["category_id", "confidence"], ascending=[True, False])
        .reset_index(drop=True)
    )
    sample.to_csv(OUT_SAMPLE, index=False)
    print(f"Wrote {len(sample):,} rows → {OUT_SAMPLE}")
    print(f"Categories covered: {sample['category_id'].nunique()}/18")


if __name__ == "__main__":
    main()
