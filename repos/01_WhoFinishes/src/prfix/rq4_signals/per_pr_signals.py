"""Build a per-PR feature table for downstream statistical analysis.

One row per merged AIDev-pop PR (and AIDev human-baseline PR), with the
four merge-time signals:

* ``hours_under_review`` -- merged_at - created_at, hours
* ``n_commits``           -- total commits on the PR branch
* ``lines_post_first``    -- lines added+deleted in commits *after* the
                              first non-empty commit
* ``n_review_items``      -- inline-review-comments + reviews + PR-comments

``n_by_other`` (commits where author != PR opener) is also computed and
stored, but it is a retired split that the paper does not report; see
``prfix.rq1_fix_rates.linker`` for what replaced it.

The four signals are stored together so per-PR statistical tests
(Mann-Whitney U, Cliff's delta) and per-PR predictor models (RQ4) reuse
the same feature table.

**Scope**: AIDev-pop as defined by Li et al. 2025 (Table 3b) -- repos with
>=500 GitHub stars. The published ``pull_request`` table is broader
(>=101 stars, 2,807 repos), so we restrict to the AIDev-pop subset by
joining against ``repository.stars`` in each arm of the UNION.

**Human comparison cohort**: AIDev publishes a PR-level
``human_pull_request`` table (5,044 merged PRs at >=500 stars). The six
detail tables (pr_commits, pr_commit_details, pr_reviews,
pr_review_comments_v2, pr_comments, pr_timeline) are not in the AIDev
release for the human cohort; we re-fetched the corresponding fields per
PR via the GitHub REST API and store them as ``human_pr_commits``,
``human_pr_commit_details``, and ``human_pr_review_items``. With this
enrichment the Human cohort participates in all four signals.
"""

from __future__ import annotations

import pandas as pd

from prfix.common.loaders.base import connect
from prfix.rq3_commit_authors import classify_commit

# Star floor used to restrict every cohort to AIDev-pop scope.
AIDEV_POP_MIN_STARS: int = 500


SQL = f"""
WITH commit_events AS (
    SELECT
        t.pr_id,
        t.commit_id AS sha,
        -- commit_id is a deterministic tiebreaker: multiple 'committed'
        -- timeline events on a PR can share an identical created_at, and
        -- without a tiebreaker DuckDB's parallel scan assigns commit_idx
        -- (hence which commit is "first") non-deterministically across runs,
        -- which leaks into lines_first / lines_post_first / loc_bucket.
        ROW_NUMBER() OVER (PARTITION BY t.pr_id ORDER BY t.created_at, t.commit_id) AS commit_idx
    FROM pr_timeline t
    WHERE t.event = 'committed' AND t.commit_id IS NOT NULL
),
commit_lines AS (
    SELECT
        ce.pr_id, ce.commit_idx,
        SUM(COALESCE(d.additions, 0) + COALESCE(d.deletions, 0)) AS lines
    FROM commit_events ce
    LEFT JOIN pr_commit_details d
           ON d.pr_id = ce.pr_id AND d.sha = ce.sha
    GROUP BY ce.pr_id, ce.commit_idx
),
non_empty AS (
    SELECT pr_id, lines,
           ROW_NUMBER() OVER (PARTITION BY pr_id ORDER BY commit_idx) AS idx_ne
    FROM commit_lines
    WHERE lines > 0
),
edit_pp AS (
    SELECT
        pr_id,
        SUM(CASE WHEN idx_ne = 1 THEN lines ELSE 0 END) AS lines_first,
        SUM(CASE WHEN idx_ne > 1 THEN lines ELSE 0 END) AS lines_post_first
    FROM non_empty
    GROUP BY pr_id
),
commit_pp AS (
    SELECT
        c.pr_id,
        COUNT(*)                                                AS n_commits,
        SUM(CASE WHEN c.author <> p.user THEN 1 ELSE 0 END)     AS n_by_other
    FROM pr_commits c
    JOIN pull_request p ON p.id = c.pr_id
    WHERE p.merged_at IS NOT NULL
    GROUP BY c.pr_id
),
review_comments_pp AS (
    SELECT r.pr_id, COUNT(*) AS n
    FROM pr_review_comments_v2 rc
    JOIN pr_reviews r ON r.id = rc.pull_request_review_id
    GROUP BY r.pr_id
),
reviews_pp AS (
    SELECT pr_id, COUNT(*) AS n FROM pr_reviews GROUP BY pr_id
),
comments_pp AS (
    SELECT pr_id, COUNT(*) AS n FROM pr_comments GROUP BY pr_id
),
-- Human (contemporaneous) parallel CTEs. AIDev publishes only the
-- PR-level human_pull_request; the six detail tables we need are not
-- in the release, so we re-fetched them via the GitHub REST API and
-- stored the results in human_pr_commits / human_pr_commit_details /
-- human_pr_review_items. The CTE shape mirrors the agent arm above.
human_commit_lines AS (
    SELECT
        c.pr_id, c.commit_order AS commit_idx,
        SUM(COALESCE(d.additions, 0) + COALESCE(d.deletions, 0)) AS lines
    FROM human_pr_commits c
    LEFT JOIN human_pr_commit_details d
           ON d.pr_id = c.pr_id AND d.sha = c.sha
    GROUP BY c.pr_id, c.commit_order
),
human_non_empty AS (
    SELECT pr_id, lines,
           ROW_NUMBER() OVER (PARTITION BY pr_id ORDER BY commit_idx) AS idx_ne
    FROM human_commit_lines
    WHERE lines > 0
),
human_edit_pp AS (
    SELECT
        pr_id,
        SUM(CASE WHEN idx_ne = 1 THEN lines ELSE 0 END) AS lines_first,
        SUM(CASE WHEN idx_ne > 1 THEN lines ELSE 0 END) AS lines_post_first
    FROM human_non_empty
    GROUP BY pr_id
),
human_commit_pp AS (
    SELECT
        c.pr_id,
        COUNT(*)                                                AS n_commits,
        SUM(CASE WHEN c.author <> h."user" THEN 1 ELSE 0 END)   AS n_by_other
    FROM human_pr_commits c
    JOIN human_pull_request h ON h.id = c.pr_id
    WHERE h.merged_at IS NOT NULL
    GROUP BY c.pr_id
)
SELECT
    p.id                                    AS pr_id,
    p.agent                                 AS agent,
    EXTRACT(EPOCH FROM (
        TRY_CAST(p.merged_at  AS TIMESTAMP)
      - TRY_CAST(p.created_at AS TIMESTAMP)
    )) / 3600.0                             AS hours_under_review,
    COALESCE(c.n_commits, 0)                AS n_commits,
    COALESCE(c.n_by_other, 0)               AS n_by_other,
    COALESCE(e.lines_first, 0)              AS lines_first,
    COALESCE(e.lines_post_first, 0)         AS lines_post_first,
    COALESCE(rc.n, 0) + COALESCE(rv.n, 0)
                       + COALESCE(cm.n, 0) AS n_review_items,
    -- Covariates for stratum-matched analysis.
    t.type                                  AS task_type,
    r.language                              AS language,
    p.repo_id                               AS repo_id,
    CASE
        WHEN COALESCE(e.lines_first, 0) <= 50  THEN 'small_0_50'
        WHEN COALESCE(e.lines_first, 0) <= 500 THEN 'med_50_500'
        ELSE                                        'large_500plus'
    END                                     AS loc_bucket
FROM pull_request p
LEFT JOIN edit_pp           e  ON e.pr_id  = p.id
LEFT JOIN commit_pp         c  ON c.pr_id  = p.id
LEFT JOIN review_comments_pp rc ON rc.pr_id = p.id
LEFT JOIN reviews_pp        rv ON rv.pr_id = p.id
LEFT JOIN comments_pp       cm ON cm.pr_id = p.id
LEFT JOIN pr_task_type      t  ON t.id     = p.id
LEFT JOIN repository        r  ON r.id     = p.repo_id
WHERE p.merged_at IS NOT NULL
  AND r.stars >= {AIDEV_POP_MIN_STARS}

UNION ALL

-- Human (contemporaneous) PR cohort. AIDev publishes only PR-level
-- columns; the four commit/review-derived signals come from our REST-
-- API enrichment (human_pr_commits, human_pr_commit_details,
-- human_pr_review_items).
SELECT
    h.id                                    AS pr_id,
    h.agent                                 AS agent,
    EXTRACT(EPOCH FROM (
        TRY_CAST(h.merged_at  AS TIMESTAMP)
      - TRY_CAST(h.created_at AS TIMESTAMP)
    )) / 3600.0                             AS hours_under_review,
    COALESCE(hc.n_commits, 0)               AS n_commits,
    COALESCE(hc.n_by_other, 0)              AS n_by_other,
    COALESCE(he.lines_first, 0)             AS lines_first,
    COALESCE(he.lines_post_first, 0)        AS lines_post_first,
    COALESCE(hri.n_review_items, 0)         AS n_review_items,
    th.type                                 AS task_type,
    rh.language                             AS language,
    rh.id                                   AS repo_id,
    CASE
        WHEN COALESCE(he.lines_first, 0) <= 50  THEN 'small_0_50'
        WHEN COALESCE(he.lines_first, 0) <= 500 THEN 'med_50_500'
        ELSE                                         'large_500plus'
    END                                     AS loc_bucket
FROM human_pull_request    h
LEFT JOIN human_pr_task_type    th  ON th.id  = h.id
LEFT JOIN repository            rh  ON rh.url = h.repo_url
LEFT JOIN human_edit_pp         he  ON he.pr_id  = h.id
LEFT JOIN human_commit_pp       hc  ON hc.pr_id  = h.id
LEFT JOIN human_pr_review_items hri ON hri.pr_id = h.id
WHERE h.merged_at IS NOT NULL
  AND rh.stars >= {AIDEV_POP_MIN_STARS}

"""

# Per-commit pull for author-class counts. One row per commit of every
# merged PR across both cohorts (agent / contemporaneous-human),
# restricted to AIDev-pop (>=500 stars). Each row carries the
# commit's (author, committer, message) plus the owning PR's opener
# (``pr_user``) and AIDev agent label (``pr_agent``) so
# ``classify_commit`` can apply the Codex prior.
SQL_AUTHOR_CLASS = f"""
SELECT
    c.pr_id        AS pr_id,
    c.author       AS author,
    c.committer    AS committer,
    c.message      AS message,
    p.agent        AS pr_agent,
    p."user"       AS pr_user
FROM pr_commits c
JOIN pull_request p ON p.id = c.pr_id
JOIN repository   r ON r.id = p.repo_id
WHERE p.merged_at IS NOT NULL
  AND r.stars >= {AIDEV_POP_MIN_STARS}

UNION ALL

SELECT
    hc.pr_id       AS pr_id,
    hc.author      AS author,
    hc.committer   AS committer,
    hc.message     AS message,
    h.agent        AS pr_agent,
    h."user"       AS pr_user
FROM human_pr_commits hc
JOIN human_pull_request h ON h.id = hc.pr_id
JOIN repository        rh ON rh.url = h.repo_url
WHERE h.merged_at IS NOT NULL
  AND rh.stars >= {AIDEV_POP_MIN_STARS}
"""


def _author_class_counts(con) -> pd.DataFrame:
    """Classify every merged-PR commit and aggregate author-class counts.

    Returns one row per pr_id with ``n_agent_commits``,
    ``n_human_commits`` and ``n_automation_commits`` (integer counts).
    """
    commits = con.execute(SQL_AUTHOR_CLASS).df()
    if len(commits) == 0:
        return pd.DataFrame(columns=[
            "pr_id", "n_agent_commits", "n_human_commits", "n_automation_commits",
        ])
    commits["author_class"] = [
        classify_commit(a, c, m, pr_agent=g, pr_user=u)
        for a, c, m, g, u in zip(
            commits["author"], commits["committer"], commits["message"],
            commits["pr_agent"], commits["pr_user"], strict=False,
        )
    ]
    pivot = (
        commits.groupby(["pr_id", "author_class"])
               .size()
               .unstack(fill_value=0)
    )
    for cls in ("agent", "human", "automation"):
        if cls not in pivot.columns:
            pivot[cls] = 0
    pivot = pivot.rename(columns={
        "agent":      "n_agent_commits",
        "human":      "n_human_commits",
        "automation": "n_automation_commits",
    })
    return pivot[[
        "n_agent_commits", "n_human_commits", "n_automation_commits",
    ]].reset_index()


def compute() -> pd.DataFrame:
    """Return the per-PR feature table as a pandas DataFrame."""
    con = connect(read_only=True)
    try:
        df = con.execute(SQL).df()
        counts = _author_class_counts(con)
    finally:
        con.close()

    df = df.merge(counts, on="pr_id", how="left")
    for col in ("n_agent_commits", "n_human_commits", "n_automation_commits"):
        df[col] = df[col].fillna(0).astype(int)
    return df
