"""Generate a CSV for manual review of all manual_review performance PRs.

Filters Agent_PRs.csv to rows where:
  - Label Source == "manual_review"
  - Final Sub Pattern is not empty

Joins with the parquet commit data, applies source-file filtering,
runs the LLM patch classifier (with line-level location evidence),
and outputs a review CSV sorted by sub-pattern then pr_id.
"""
import asyncio
import csv
import json
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import anthropic
import pandas as pd
from tqdm import tqdm

from analysis.patch_classifier import (
    PATCH_CLASSIFICATION_PROMPT,
    _build_prompt,
    _parse_response,
)
from config import (
    ANTHROPIC_MODEL,
    BUSINESS_LOGIC_EXTENSIONS,
    CLASSIFY_CONCURRENCY,
    EXCLUDED_FILE_PATTERNS,
    MSR_DATA_DIR,
    PATCH_TRUNCATE_LENGTH,
)

OUTPUT_CSV = "manual_review_patches.csv"
PATCH_PREVIEW_LEN = 300


def _is_source_file(filename: str) -> bool:
    ext = os.path.splitext(filename)[1].lower()
    if ext not in BUSINESS_LOGIC_EXTENSIONS:
        return False
    for pat in EXCLUDED_FILE_PATTERNS:
        if re.search(pat, filename):
            return False
    return True


def _format_categories(categories: list) -> str:
    """Format categories list as 'id:name(conf)@Lstart-Lend; ...'"""
    parts = []
    for c in categories:
        loc = ""
        if c.get("start_line") is not None and c.get("end_line") is not None:
            loc = f"@L{c['start_line']}-L{c['end_line']}"
        parts.append(f"{c['id']}:{c['name']}({c['confidence']:.2f}){loc}")
    return "; ".join(parts)


async def classify_all(patches: list) -> list:
    """Run LLM classifier on all (pr_id, sha, filename, patch) tuples.
    Returns list of (pr_id, sha, filename, categories, reasoning)."""
    client = anthropic.AsyncAnthropic()
    semaphore = asyncio.Semaphore(CLASSIFY_CONCURRENCY)
    results = [None] * len(patches)

    async def classify_one(idx, pr_id, sha, filename, patch):
        prompt = _build_prompt(filename, patch)
        async with semaphore:
            while True:
                try:
                    msg = await client.messages.create(
                        model=ANTHROPIC_MODEL,
                        max_tokens=512,
                        messages=[{"role": "user", "content": prompt}],
                    )
                    categories, reasoning = _parse_response(msg.content[0].text)
                    results[idx] = (pr_id, sha, filename, categories, reasoning)
                    return
                except anthropic.RateLimitError as exc:
                    wait = None
                    if exc.response is not None:
                        wait = exc.response.headers.get("retry-after") or \
                               exc.response.headers.get("x-ratelimit-reset-requests")
                    wait = float(wait) if wait is not None else 60.0
                    await asyncio.sleep(wait)
                except Exception as exc:
                    results[idx] = (pr_id, sha, filename, [], f"ERROR: {exc}")
                    return
                finally:
                    progress_bar.update(1)

    with tqdm(total=len(patches), desc="Classifying patches") as progress_bar:
        await asyncio.gather(*[
            classify_one(i, *p) for i, p in enumerate(patches)
        ])
    await client.close()
    return results


def main():
    # Load Agent_PRs.csv and filter
    agent_prs_path = os.path.join(MSR_DATA_DIR, "Agent_PRs.csv")
    with open(agent_prs_path, newline="", encoding="utf-8") as f:
        all_prs = list(csv.DictReader(f))

    filtered_prs = [
        r for r in all_prs
        if r["Label Source"] == "manual_review" and r["Final Sub Pattern"].strip()
    ]
    print(f"Filtered PRs (manual_review + non-empty sub pattern): {len(filtered_prs)}")

    pr_meta = {int(r["PR ID"]): r for r in filtered_prs}
    pr_ids = set(pr_meta.keys())

    # Load parquet commit data
    parquet_path = os.path.join(MSR_DATA_DIR, "ai_pr_commit_details.parquet")
    commits = pd.read_parquet(parquet_path)
    mc = commits[commits["pr_id"].isin(pr_ids)].copy()
    print(f"Total commit-file rows for these PRs: {len(mc)}")

    # Apply source file filter
    mc["is_source"] = mc["filename"].apply(_is_source_file)
    source = mc[mc["is_source"]].copy()
    print(f"Source-only commit-file rows: {len(source)}")

    prs_with_source = set(source["pr_id"].unique())
    prs_without = pr_ids - prs_with_source
    if prs_without:
        print(f"PRs with no source files ({len(prs_without)}):")
        for pid in sorted(prs_without):
            meta = pr_meta[pid]
            print(f"  {pid}  {meta['PR URL']}  [{meta['Final Sub Pattern']}]")

    # Sort by sub_pattern then pr_id for efficient batch review
    source = source.sort_values(["pr_id", "sha", "filename"])

    # Build patch list for classification
    patches = []
    for _, row in source.iterrows():
        patches.append((
            int(row["pr_id"]),
            str(row["sha"])[:8] if pd.notna(row["sha"]) else "",
            row["filename"],
            (row["patch"] or "") if pd.notna(row["patch"]) else "",
        ))

    print(f"\nRunning LLM classifier on {len(patches)} patches...")
    llm_results = asyncio.run(classify_all(patches))

    # Build output rows, then sort by sub_pattern -> pr_id
    rows_out = []
    for i, (pr_id, sha, filename, categories, reasoning) in enumerate(llm_results):
        meta = pr_meta[pr_id]
        orig_row = source.iloc[i]
        patch_text = (orig_row["patch"] or "") if pd.notna(orig_row["patch"]) else ""
        rows_out.append({
            "patch_num": i + 1,
            "pr_id": pr_id,
            "pr_url": meta["PR URL"],
            "final_pattern": meta["Final Pattern"],
            "final_sub_pattern": meta["Final Sub Pattern"],
            "sha": sha,
            "filename": filename,
            "additions": int(orig_row["additions"]) if pd.notna(orig_row["additions"]) else "",
            "deletions": int(orig_row["deletions"]) if pd.notna(orig_row["deletions"]) else "",
            "llm_categories": _format_categories(categories or []),
            "llm_reasoning": reasoning or "",
            "patch_preview": patch_text[:PATCH_PREVIEW_LEN],
            "verdict": "",
            "manual_categories": "",
            "notes": "",
        })

    rows_out.sort(key=lambda r: (r["final_sub_pattern"], r["pr_id"], r["sha"], r["filename"]))
    # Re-number after sort
    for i, r in enumerate(rows_out):
        r["patch_num"] = i + 1

    # Write CSV
    fieldnames = [
        "patch_num", "pr_id", "pr_url", "final_pattern", "final_sub_pattern",
        "sha", "filename", "additions", "deletions",
        "llm_categories", "llm_reasoning", "patch_preview",
        "verdict", "manual_categories", "notes",
    ]
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_out)

    print(f"\nWrote {len(rows_out)} rows to {OUTPUT_CSV}")
    print(f"PRs covered: {len(set(r['pr_id'] for r in rows_out))}/{len(filtered_prs)}")


if __name__ == "__main__":
    main()
