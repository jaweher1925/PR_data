"""REST-API enrichment for AIDev's contemporaneous Human cohort.

AIDev publishes the PR-level ``human_pull_request`` table but withholds
the six detail tables (``pr_commits``, ``pr_commit_details``,
``pr_reviews``, ``pr_review_comments_v2``, ``pr_comments``,
``pr_timeline``) for the human comparison cohort. We close that gap by
calling the GitHub REST API for every merged human PR and writing
detail tables that mirror AIDev's agent-side schemas.

Outputs (all in ``data/interim/``):

* ``human_pr_commits.parquet``        -- per-commit metadata (matches AIDev's
  pr_commits schema + commit_order)
* ``human_pr_commit_details.parquet`` -- per-file diffs (matches pr_commit_details schema)
* ``human_pr_review_items.parquet``   -- per-PR n_review_items aggregating
  reviews + review-comments + issue-comments

A per-PR commit cap of 30 matches AIDev's incidental cap (GitHub's
default ``per_page=30`` on ``/pulls/N/commits``; see paper §Threats).
Wall-clock estimate: ~9 hours at the GitHub PAT rate cap of 5{,}000
req/hr.
"""

from __future__ import annotations

import pandas as pd
from rich.console import Console

from prfix.common.loaders.base import connect
from prfix.config import INTERIM_DIR

console = Console()

HUMAN_COMMITS_PATH        = INTERIM_DIR / "human_pr_commits.parquet"
HUMAN_DETAILS_PATH        = INTERIM_DIR / "human_pr_commit_details.parquet"
HUMAN_REVIEW_ITEMS_PATH   = INTERIM_DIR / "human_pr_review_items.parquet"


def _load_merged_human_prs() -> pd.DataFrame:
    """Return the merged subset of AIDev's human_pull_request table."""
    con = connect(read_only=True)
    try:
        df = con.execute("""
            SELECT id, number, repo_url, "user"
            FROM human_pull_request
            WHERE merged_at IS NOT NULL
        """).df()
    finally:
        con.close()
    df["id"]     = df["id"].astype("int64")
    df["number"] = df["number"].astype("int64")
    return df


def enrich(
    checkpoint_every: int = 200,
    max_commits_per_pr: int | None = 30,
) -> None:
    """Fetch commits, diffs, and review-event counts for every merged
    human PR. Writes three parquets and registers them as DuckDB views.
    Re-uses the existing ``data/github_cache/`` so a resume after a
    crash skips already-fetched calls.
    """
    from prfix.common.loaders.github_api import (
        count_pr_issue_comments,
        count_pr_review_comments,
        count_pr_reviews,
        get_commit_detail,
        get_pr_commits,
        parse_repo_url,
    )

    prs = _load_merged_human_prs()
    console.log(
        f"enrich_human_cohort: {len(prs):,} merged Human (contemp) PRs"
    )

    commits_rows: list[dict] = []
    details_rows: list[dict] = []
    review_rows:  list[dict] = []

    n_calls   = 0
    n_errors  = 0

    def _flush() -> None:
        pd.DataFrame.from_records(commits_rows).to_parquet(
            HUMAN_COMMITS_PATH, index=False
        )
        pd.DataFrame.from_records(details_rows).to_parquet(
            HUMAN_DETAILS_PATH, index=False
        )
        pd.DataFrame.from_records(review_rows).to_parquet(
            HUMAN_REVIEW_ITEMS_PATH, index=False
        )

    for i, row in prs.iterrows():
        if (i + 1) % checkpoint_every == 0:
            console.log(
                f"  [dim]{i+1}/{len(prs)}[/dim] commits={len(commits_rows):,} "
                f"details={len(details_rows):,} reviews={len(review_rows):,} "
                f"api={n_calls:,} errs={n_errors}"
            )
            if commits_rows:
                _flush()

        try:
            owner, repo = parse_repo_url(row["repo_url"])
            n_pr_number = int(row["number"])
            pr_id_int   = int(row["id"])

            # --- Commits + per-commit diffs (mirrors AIDev pr_commits + pr_commit_details) ---
            commits = get_pr_commits(owner, repo, n_pr_number)
            n_calls += 1
            if max_commits_per_pr and len(commits) > max_commits_per_pr:
                commits = commits[:max_commits_per_pr]
            for commit_order, c in enumerate(commits, start=1):
                sha = c.get("sha")
                if not sha:
                    continue
                commit = c.get("commit") or {}
                a      = commit.get("author") or {}
                cm     = commit.get("committer") or {}
                author    = a.get("name") or (c.get("author") or {}).get("login") or ""
                committer = cm.get("name") or (c.get("committer") or {}).get("login") or ""
                message   = commit.get("message") or ""

                commits_rows.append({
                    "sha":          sha,
                    "pr_id":        pr_id_int,
                    "author":       author,
                    "committer":    committer,
                    "message":      message,
                    "commit_order": commit_order,
                })

                detail = get_commit_detail(owner, repo, sha)
                n_calls += 1
                if not detail:
                    continue
                stats = detail.get("stats") or {}
                files = detail.get("files") or []
                if not files:
                    continue
                for f in files:
                    details_rows.append({
                        "sha":                    sha,
                        "pr_id":                  pr_id_int,
                        "author":                 author,
                        "committer":              committer,
                        "message":                message,
                        "commit_stats_total":     stats.get("total", 0),
                        "commit_stats_additions": stats.get("additions", 0),
                        "commit_stats_deletions": stats.get("deletions", 0),
                        "filename":               f.get("filename") or "",
                        "status":                 f.get("status") or "",
                        "additions":              float(f.get("additions") or 0),
                        "deletions":              float(f.get("deletions") or 0),
                        "changes":                float(f.get("changes") or 0),
                        "patch":                  f.get("patch") or "",
                    })

            # --- Review activity counts (reviews + review comments + PR comments per PR) ---
            n_reviews        = count_pr_reviews(owner, repo, n_pr_number)
            n_review_cmts    = count_pr_review_comments(owner, repo, n_pr_number)
            n_issue_cmts     = count_pr_issue_comments(owner, repo, n_pr_number)
            n_calls += 3
            review_rows.append({
                "pr_id":           pr_id_int,
                "n_review_items":  int(n_reviews + n_review_cmts + n_issue_cmts),
            })

        except Exception as exc:
            n_errors += 1
            console.log(
                f"[yellow]warn[/yellow] PR {row['id']} ({row['repo_url']}#{row['number']}): "
                f"{type(exc).__name__}: {exc}"
            )

    _flush()
    console.log(
        f"[green]wrote[/green] {HUMAN_COMMITS_PATH} ({len(commits_rows):,} rows)"
    )
    console.log(
        f"[green]wrote[/green] {HUMAN_DETAILS_PATH} ({len(details_rows):,} rows)"
    )
    console.log(
        f"[green]wrote[/green] {HUMAN_REVIEW_ITEMS_PATH} ({len(review_rows):,} rows)"
    )
    console.log(
        f"summary: {len(prs)} PRs, {n_calls:,} API calls, {n_errors} errors"
    )

    # Register DuckDB views so per_pr_signals can join them.
    con = connect()
    try:
        for view, path in [
            ("human_pr_commits",        HUMAN_COMMITS_PATH),
            ("human_pr_commit_details", HUMAN_DETAILS_PATH),
            ("human_pr_review_items",   HUMAN_REVIEW_ITEMS_PATH),
        ]:
            con.execute(
                f"CREATE OR REPLACE VIEW {view} AS "
                f"SELECT * FROM read_parquet('{path}')"
            )
            n = con.execute(f"SELECT COUNT(*) FROM {view}").fetchone()[0]
            console.log(f"[green]view[/green]    {view} ({n:,} rows)")
    finally:
        con.close()


if __name__ == "__main__":  # pragma: no cover
    enrich()
