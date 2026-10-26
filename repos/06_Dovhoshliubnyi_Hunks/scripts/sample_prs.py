"""Sample 60 PRs with relevance score 4 or 5 for manual validation."""

import csv
import random
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import DB_PATH, RELEVANCE_THRESHOLD

SAMPLE_SIZE = 60
RANDOM_SEED = 42
OUTPUT_DIR = Path(__file__).parent.parent / "validation"
OUTPUT_CSV = OUTPUT_DIR / "sampled_prs.csv"


def main():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    cur.execute(
        """
        SELECT p.id, p.title, p.body, ppc.relevance_score,
               ppc.categories_detected, ppc.reasoning
        FROM pull_request p
        JOIN pr_performance_classification ppc ON p.id = ppc.pr_id
        WHERE ppc.relevance_score >= ?
        """,
        (RELEVANCE_THRESHOLD,),
    )
    rows = cur.fetchall()
    conn.close()

    print(f"Total PRs with score >= {RELEVANCE_THRESHOLD}: {len(rows)}")

    random.seed(RANDOM_SEED)
    sample = random.sample(rows, min(SAMPLE_SIZE, len(rows)))
    print(f"Sampled {len(sample)} PRs (seed={RANDOM_SEED})")

    OUTPUT_DIR.mkdir(exist_ok=True)
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_ALL)
        writer.writerow(["pr_id", "title", "body", "relevance_score",
                         "categories_detected", "original_reasoning"])
        for row in sample:
            writer.writerow(row)

    print(f"Written to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
