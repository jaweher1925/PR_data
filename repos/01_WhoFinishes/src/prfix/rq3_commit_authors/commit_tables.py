"""Commit-level author classes INSIDE the verified fix PRs (thin CLI).

All logic lives in prfix.common.metrics (gate-asserted by
consolidation_gate.py); this script just prints the two aggregation views
and writes the CSVs. NEVER compare the per-PR means here against pooled
shares elsewhere: the two answer different questions.

Run:  uv run python -m prfix.rq3_commit_authors.commit_tables
"""
from __future__ import annotations

from prfix.common import metrics


def main() -> None:
    q = metrics.verified_fix_prs()
    print(f"verified fix PRs: {len(q)} unique "
          f"({int((~q.q_is_human).sum())} agent-table, "
          f"{int(q.q_is_human.sum())} human-table)")

    c = metrics.fix_pr_commit_classes(q)
    s = metrics.per_pr_metrics(c)
    print("\n== per opener class (content-only; per-PR AND pooled views) ==")
    print(s.to_string())

    metrics.DIGESTS_DIR.mkdir(parents=True, exist_ok=True)
    out = metrics.DIGESTS_DIR / "commit_tables.csv"
    s.reset_index().to_csv(out, index=False)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
