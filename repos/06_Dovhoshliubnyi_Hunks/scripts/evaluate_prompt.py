"""Classify 500 random PRs and sample 10 per score bucket for manual review."""

import argparse
import asyncio
import random
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from analysis.pr_classifier import CLASSIFICATION_PROMPT, _parse_response, _run_anthropic, _run_local
from config import ANTHROPIC_MODEL, BODY_TRUNCATE_LENGTH, CLASSIFY_CONCURRENCY, CLASSIFICATION_BATCH_SIZE, LOCAL_MODEL_NAME


def main():
    parser = argparse.ArgumentParser(description="Classify random PRs and sample by score bucket.")
    parser.add_argument("--db", default="aidev.db")
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument("--concurrency", type=int, default=CLASSIFY_CONCURRENCY)
    parser.add_argument("--local", action="store_true")
    parser.add_argument("--sample", type=int, default=10, help="Samples per bucket")
    args = parser.parse_args()

    conn = sqlite3.connect(args.db)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS pr_performance_classification (
            pr_id     INTEGER PRIMARY KEY,
            relevance_score    REAL,
            categories_detected TEXT,
            reasoning          TEXT
        )
    """)
    conn.commit()

    cur.execute(
        """
        SELECT p.id, p.title, p.body
        FROM pull_request p
        LEFT JOIN pr_performance_classification ppc ON p.id = ppc.pr_id
        WHERE p.merged_at IS NOT NULL
          AND ppc.pr_id IS NULL
        ORDER BY RANDOM()
        LIMIT ?
        """,
        (args.limit,),
    )
    prs = cur.fetchall()

    if not prs:
        print("No unclassified merged PRs found.")
        conn.close()
        return

    print(f"Classifying {len(prs)} PRs...")

    results = {}  # pr_id -> (score, categories, reasoning)

    def on_chunk(chunk_results):
        for pr_id, score, cats, reasoning, error in chunk_results:
            if error is not None:
                print(f"  [skip] PR {pr_id}: {error}", file=sys.stderr)
                continue
            results[pr_id] = (score, cats, reasoning)
            cur.execute(
                """
                INSERT OR REPLACE INTO pr_performance_classification
                    (pr_id, relevance_score, categories_detected, reasoning)
                VALUES (?, ?, ?, ?)
                """,
                (pr_id, score, cats, reasoning),
            )
        conn.commit()

    from tqdm import tqdm
    with tqdm(total=len(prs), desc="Classifying") as progress_bar:
        if args.local:
            run_fn = _run_local
            effective_model = LOCAL_MODEL_NAME
        else:
            run_fn = _run_anthropic
            effective_model = ANTHROPIC_MODEL
        asyncio.run(run_fn(prs, effective_model, progress_bar, on_chunk, args.concurrency))

    conn.close()

    # Build lookup for titles
    title_map = {pr_id: title for pr_id, title, _ in prs}

    # Bucket by score
    buckets = {"5": [], "4": [], "3": []}
    for pr_id, (score, cats, reasoning) in results.items():
        title = title_map.get(pr_id, "")
        entry = (pr_id, score, title, reasoning)
        if score == 5:
            buckets["5"].append(entry)
        elif score == 4:
            buckets["4"].append(entry)
        elif score == 3:
            buckets["3"].append(entry)

    print(f"\nClassified {len(results)} PRs total.")
    print(f"  Score 5: {len(buckets['5'])}")
    print(f"  Score 4: {len(buckets['4'])}")
    print(f"  Score 3: {len(buckets['3'])}")

    all_sample_ids = []
    for bucket_name, items in buckets.items():
        sample = random.sample(items, min(args.sample, len(items)))
        print(f"\n{'='*60}")
        print(f"Bucket {bucket_name}  ({len(items)} PRs, showing {len(sample)})")
        print(f"{'='*60}")
        for pr_id, score, title, reasoning in sample:
            all_sample_ids.append(pr_id)
            print(f"  {pr_id}  {score}  {title[:72]}")
            print(f"           -> {reasoning}")

    print("\n" + "="*60)
    print("Claude Code review prompt (paste PR IDs below):")
    print("="*60)
    print("""
You are reviewing a random sample of GitHub PRs that were classified
as performance-related by an LLM classifier. For each PR below, look up
its title and body in the database, then judge:

  TP (True Positive)  = PR genuinely improves runtime performance
  FP (False Positive) = PR was mis-classified, not a perf improvement

Output a table:
  PR_ID | SCORE | VERDICT | ONE_LINE_REASON

Sample PR IDs:""")
    for pr_id in all_sample_ids:
        print(f"  {pr_id}")


if __name__ == "__main__":
    main()
