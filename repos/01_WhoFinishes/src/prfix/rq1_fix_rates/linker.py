"""RQ1 behavioural follow-up: did a merged agent PR get a subsequent
bug-fix or revert PR that touches the same files?

For each merged AIDev-pop agent PR ``P`` we ask: was there a follow-up PR
``Q`` in the *same repository* that

  * was merged within 14 / 30 days *after* P,
  * is classified ``fix`` by AIDev's Conventional-Commits classifier, OR
    has a revert-style title (``Revert ...``),
  * shares at least one filename with P (per ``pr_commit_details``)?

The behavioural signal tells us how often a merge *actually* turned out
to be a regression, as evidenced by a follow-up fix. It is the candidate
half of RQ1; the LLM judge then verifies each candidate pair.

Implementation notes
--------------------

* The candidate follow-up pool includes BOTH agent fixers
  (``pull_request`` + ``pr_task_type`` type='fix' / revert title) AND
  contemporaneous *human* fixers (``human_pull_request`` +
  ``human_pr_task_type``), unioned in the ``q_fix`` CTE: a regression an
  agent merged is often fixed by a human, so the pool must include human
  fixers. File overlap for human Q is tested against the same
  ``{FILES_SOURCE}`` table (which carries both cohorts' net-diff files).
* The entire pipeline is a single DuckDB query: a self-join on
  ``repo_id`` with a 30-day range predicate, plus a file-overlap
  subquery. On the full 24k merged AIDev-pop cohort this runs in
  well under one second; the wall-clock budget is dominated by Python
  / pandas post-processing (still under 30 s).
* Per-PR results are persisted to CSV so the contingency analysis can
  be re-run without re-querying DuckDB.

Edge cases
----------

* Repos with only one merged AIDev-pop PR contribute no candidate
  follow-ups (correctly resulting in ``has_followup_30d = 0``).
* PRs merged near AIDev's 2025-07-30 collection cut-off have a
  truncated follow-up window. ``censored_30d`` flags PRs whose
  ``merged_at + 30 days`` exceeds the cohort's max ``merged_at``,
  so downstream analyses can either drop them or compute rates among
  uncensored PRs only. The headline ``per_agent_summary`` reports both.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table
from scipy.stats import chi2_contingency

from prfix.common.loaders.base import connect
from prfix.config import DIGESTS_DIR, ensure_dirs
from prfix.rq4_signals.per_pr_signals import compute as per_pr_compute

console = Console()


# Per-PR CSVs are cached so the contingency-test step can be re-run
# without re-querying DuckDB.
PER_PR_CSV = DIGESTS_DIR / "rq1_linker_per_pr.csv"
PER_AGENT_CSV = DIGESTS_DIR / "rq1_linker_per_agent.csv"


# Follow-up windows, in days.
WINDOW_DAYS_SHORT = 14
WINDOW_DAYS_LONG = 30


# Core SQL: produces one row per (P, Q) pair where Q is a candidate
# follow-up fix/revert within 30 days that shares at least one filename
# with P. Aggregation to per-PR happens in pandas.
# P-side population per cohort. "agent" is the paper's default; "human" flips
# the linker to the contemporaneous human cohort as the P side (the incidence
# baseline: do HUMAN merges get followed by fixes at the same rate?). The Q
# side (q_fix) is identical in both modes -- fixes come from both cohorts.
P_POP = {
    "agent": """
    -- All merged AIDev-pop agent PRs. We restrict via the >=500-star
    -- definition of AIDev-pop from Li et al. 2025 (Table 3b) so this
    -- analysis matches every other RQ in the paper.
    SELECT
        p.id        AS p_id,
        p.repo_id   AS repo_id,
        p.agent     AS agent,
        TRY_CAST(p.merged_at AS TIMESTAMP) AS p_merged
    FROM pull_request p
    JOIN repository  r ON r.id = p.repo_id
    WHERE p.merged_at IS NOT NULL
      AND r.stars >= 500
""",
    "human": """
    -- All merged contemporaneous human-cohort PRs (same >=500-star repos;
    -- the human table joins repository via url, it carries no repo_id).
    SELECT
        p.id        AS p_id,
        rh.id       AS repo_id,
        'Human'     AS agent,
        TRY_CAST(p.merged_at AS TIMESTAMP) AS p_merged
    FROM human_pull_request p
    JOIN repository rh ON rh.url = p.repo_url
    WHERE p.merged_at IS NOT NULL
      AND rh.stars >= 500
""",
}

SQL_FOLLOWUP_PAIRS = """
WITH p_pop AS (
{P_POP}
),
q_fix AS (
    -- Candidate follow-up PRs: AIDev-pop agent PRs classified as ``fix``
    -- by AIDev's Conventional-Commits classifier, or with a revert-style
    -- title. We use ``pull_request`` (not ``all_pull_request``) because
    -- ``pr_commit_details`` and ``pr_task_type`` are only published for
    -- AIDev-pop, so we cannot check file overlap or task type outside
    -- that subset.
    SELECT
        q.id        AS q_id,
        q.repo_id   AS repo_id,
        TRY_CAST(q.merged_at AS TIMESTAMP) AS q_merged,
        q.title     AS q_title,
        COALESCE(t.type, '(none)') AS q_type
    FROM pull_request q
    JOIN repository   r ON r.id = q.repo_id
    LEFT JOIN pr_task_type t ON t.id = q.id
    WHERE q.merged_at IS NOT NULL
      AND r.stars >= 500
      AND (
            t.type = 'fix'
         OR q.title LIKE 'Revert %'
         OR q.title LIKE 'revert %'
      )

    UNION ALL

    -- Contemporaneous *human* follow-up PRs in the same AIDev-pop repos.
    -- A regression introduced by an agent PR is often fixed by a human,
    -- so the candidate pool must include human fixers. Fix-typed via
    -- ``human_pr_task_type`` (type='fix') OR a revert-style title; file
    -- overlap is checked against ``human_pr_commit_details`` in the
    -- ``pair_files`` step below. ``repository`` joins on ``url`` because
    -- the human PR table carries ``repo_url`` rather than ``repo_id``.
    SELECT
        h.id        AS q_id,
        rh.id       AS repo_id,
        TRY_CAST(h.merged_at AS TIMESTAMP) AS q_merged,
        h.title     AS q_title,
        COALESCE(th.type, '(none)') AS q_type
    FROM human_pull_request h
    JOIN repository   rh ON rh.url = h.repo_url
    LEFT JOIN human_pr_task_type th ON th.id = h.id
    WHERE h.merged_at IS NOT NULL
      AND rh.stars >= 500
      AND (
            th.type = 'fix'
         OR h.title LIKE 'Revert %'
         OR h.title LIKE 'revert %'
      )
),
pair AS (
    -- Self-join: candidate follow-ups in the same repo, merged
    -- strictly after P but within 30 days. Skip self-matches.
    SELECT
        p.p_id,
        p.agent,
        p.p_merged,
        q.q_id,
        q.q_merged,
        q.q_title,
        q.q_type,
        DATE_DIFF('day', p.p_merged, q.q_merged) AS day_gap
    FROM p_pop p
    JOIN q_fix q
      ON q.repo_id = p.repo_id
     AND q.q_merged >  p.p_merged
     AND q.q_merged <= p.p_merged + INTERVAL 30 DAY
     AND q.q_id     <> p.p_id
),
all_commit_files AS (
    -- Per-PR file set used for the overlap test. The source is chosen by the
    -- ``linker`` mode (see FILES_SOURCE): the default is the NET-DIFF
    -- (base->head) non-boilerplate co-edit (``pr_net_files``), which avoids the
    -- per-commit-union over-linking; ``generic`` reproduces the old per-commit
    -- union for sensitivity analysis.
    {FILES_SOURCE}
),
pair_files AS (
    -- File overlap: keep (P, Q) pairs that share at least one filename.
    SELECT
        pair.p_id,
        pair.agent,
        pair.q_id,
        pair.day_gap,
        pair.q_title,
        pair.q_type
    FROM pair
    JOIN all_commit_files pf
      ON pf.pr_id = pair.p_id
    JOIN all_commit_files qf
      ON qf.pr_id = pair.q_id
     AND qf.filename = pf.filename
    GROUP BY pair.p_id, pair.agent, pair.q_id, pair.day_gap,
             pair.q_title, pair.q_type
)
SELECT * FROM pair_files
ORDER BY p_id, day_gap
"""


# Cohort-level info used to flag censored PRs (those merged near the
# AIDev collection cut-off so their 14 / 30-day follow-up window is
# partially truncated).
SQL_COHORT = {
    "agent": """
SELECT
    p.id        AS pr_id,
    p.agent     AS agent,
    p.repo_id   AS repo_id,
    TRY_CAST(p.merged_at AS TIMESTAMP) AS merged_at
FROM pull_request p
JOIN repository   r ON r.id = p.repo_id
WHERE p.merged_at IS NOT NULL
  AND r.stars >= 500
""",
    "human": """
SELECT
    p.id        AS pr_id,
    'Human'     AS agent,
    rh.id       AS repo_id,
    TRY_CAST(p.merged_at AS TIMESTAMP) AS merged_at
FROM human_pull_request p
JOIN repository rh ON rh.url = p.repo_url
WHERE p.merged_at IS NOT NULL
  AND rh.stars >= 500
""",
}


# Shared observation cut-off date for censoring. The Q (follow-up) pool always
# spans BOTH PR tables regardless of p_side, so a Q is only observable up to the
# earlier of the two tables' last merge. Censoring each P cohort against its own
# table's max would give the agent and human frames different observation
# windows, which repository stratification cannot correct.
# DuckDB's LEAST() ignores NULL arguments, so the CASE is what makes an
# unloaded table propagate as NULL instead of silently yielding the other
# table's max (which would be that cohort's own cut-off, not a shared one).
SQL_SHARED_CUTOFF = """
WITH ends AS (
    SELECT
        (SELECT MAX(TRY_CAST(q.merged_at AS TIMESTAMP))
           FROM pull_request q
           JOIN repository r ON r.id = q.repo_id
          WHERE q.merged_at IS NOT NULL AND r.stars >= 500) AS agent_end,
        (SELECT MAX(TRY_CAST(h.merged_at AS TIMESTAMP))
           FROM human_pull_request h
           JOIN repository rh ON rh.url = h.repo_url
          WHERE h.merged_at IS NOT NULL AND rh.stars >= 500) AS human_end
)
SELECT CASE WHEN agent_end IS NULL OR human_end IS NULL THEN NULL
            ELSE LEAST(agent_end, human_end) END AS shared_cutoff
FROM ends
"""


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------


# File-set source per linker mode (substituted into the all_commit_files CTE).
#   net_sub  -- net-diff base->head co-edit, boilerplate excluded (DEFAULT/primary)
#   net_any  -- net-diff co-edit, boilerplate included
#   generic  -- per-commit union (legacy; over-links; kept for sensitivity)
FILES_SOURCE = {
    "net_sub": "SELECT pr_id, filename FROM pr_net_files WHERE NOT is_boilerplate",
    "net_any": "SELECT pr_id, filename FROM pr_net_files",
    "generic": ("SELECT DISTINCT pr_id, filename FROM pr_commit_details "
                "UNION SELECT DISTINCT pr_id, filename FROM human_pr_commit_details"),
}


def net_sub_excluding(filenames) -> str:
    """The ``net_sub`` file source with ``filenames`` additionally excluded.

    The stored ``pr_net_files.is_boilerplate`` column was precomputed with an
    older, buggier ``is_boilerplate``; subtracting the filenames the corrected
    rule reclassifies stands in for the corrected predicate without a lake
    rebuild. Valid only while the corrected rule is a SUPERSET of the stored one
    (ledger row 34 checks that first). An empty list is the identity.
    """
    lit = ",".join("'" + f.replace("'", "''") + "'" for f in filenames)
    return FILES_SOURCE["net_sub"] + (f" AND filename NOT IN ({lit})" if lit else "")


def pairs_prelude_sql(p_side: str = "agent") -> str:
    """The pair-CTE prelude (p_pop + q_fix + pair), substituted for a P side,
    WITHOUT the file-overlap stage -- for callers that enumerate time/repo
    candidate pairs regardless of linker mode (netdiff fetch, sensitivity)."""
    if p_side not in P_POP:
        raise ValueError(f"unknown p_side {p_side!r}; choose {list(P_POP)}")
    full = SQL_FOLLOWUP_PAIRS.replace("{P_POP}", P_POP[p_side])
    return full.split("all_commit_files AS")[0].rstrip().rstrip(",")


def pairs_sql(linker: str = "net_sub", p_side: str = "agent") -> str:
    """The fully substituted follow-up pair SQL for a linker + P-side cohort."""
    if linker not in FILES_SOURCE:
        raise ValueError(f"unknown linker {linker!r}; choose {list(FILES_SOURCE)}")
    if p_side not in P_POP:
        raise ValueError(f"unknown p_side {p_side!r}; choose {list(P_POP)}")
    return (SQL_FOLLOWUP_PAIRS
            .replace("{P_POP}", P_POP[p_side])
            .replace("{FILES_SOURCE}", FILES_SOURCE[linker]))


def compute(linker: str = "net_sub", p_side: str = "agent",
            cutoff: str = "own") -> pd.DataFrame:
    """Return one row per merged AIDev-pop PR with follow-up flags.

    ``linker`` selects how P and Q must overlap (see FILES_SOURCE). The default
    ``net_sub`` requires a net-diff (base->head) co-edit of a non-boilerplate
    file; ``generic`` reproduces the legacy per-commit union for sensitivity.
    ``p_side`` selects the P cohort: "agent" (default, the paper's cohort) or
    "human" (the contemporaneous human cohort, for the incidence baseline).
    NOTE: the net_sub/net_any linkers only see PRs present in ``pr_net_files``;
    run the net-diff fetch for the human cohort before trusting p_side="human"
    rates (see loaders.fetch_netdiff --p-side human).
    ``cutoff`` selects the censoring cut-off date (hybrid design):
      * "own"    -- the P cohort's own table end. Default; correct for
                    SINGLE-cohort analyses (keeps the full frame and matches
                    the version1 numbers).
      * "shared" -- the EARLIER of the two Q-source table ends, so both
                    cohorts observe fixes over the same calendar period.
                    REQUIRED for any agent-vs-human comparison: with per-cohort
                    cut-offs the two frames have different observation windows
                    (agent table ends later than the human table), which repo
                    stratification cannot correct.

    Columns:
        pr_id, agent, merged_at,
        has_followup_14d, has_followup_30d,
        n_followups_30d, n_followups_14d,
        censored_14d, censored_30d
    """
    if cutoff not in ("own", "shared"):
        raise ValueError(f"unknown cutoff {cutoff!r}; choose 'own' or 'shared'")
    sql = pairs_sql(linker, p_side)
    con = connect(read_only=True)
    try:
        pairs = con.execute(sql).df()
        cohort = con.execute(SQL_COHORT[p_side]).df()
        shared_cut = (con.execute(SQL_SHARED_CUTOFF).fetchone()[0]
                      if cutoff == "shared" else None)
    finally:
        con.close()
    if cutoff == "shared":
        shared_cut = pd.to_datetime(shared_cut)
        if pd.isna(shared_cut):
            raise ValueError(
                "cutoff='shared' asked for, but the shared cut-off date is NULL: "
                "one of the two Q-source tables (pull_request, human_pull_request, "
                "stars >= 500) has no parseable merged_at. Load / rebuild the lake "
                "before running any cross-cohort comparison.")

    # Per-PR aggregation in pandas; pairs is bounded (~60k rows in
    # current data) so this is cheap.
    if len(pairs):
        agg14 = pairs[pairs["day_gap"] <= WINDOW_DAYS_SHORT] \
                    .groupby("p_id").size().rename("n_followups_14d")
        agg30 = pairs.groupby("p_id").size().rename("n_followups_30d")
    else:
        agg14 = pd.Series(dtype=int, name="n_followups_14d")
        agg30 = pd.Series(dtype=int, name="n_followups_30d")

    out = cohort.rename(columns={"pr_id": "pr_id"}).copy()
    out = out.merge(agg14, left_on="pr_id", right_index=True, how="left")
    out = out.merge(agg30, left_on="pr_id", right_index=True, how="left")
    out["n_followups_14d"] = out["n_followups_14d"].fillna(0).astype(int)
    out["n_followups_30d"] = out["n_followups_30d"].fillna(0).astype(int)
    out["has_followup_14d"] = (out["n_followups_14d"] > 0).astype(int)
    out["has_followup_30d"] = (out["n_followups_30d"] > 0).astype(int)

    # Censoring: PRs whose 14 / 30-day window extends past the cut-off date
    # can't possibly observe later follow-ups. cutoff="shared" uses the same
    # date for every p_side (see SQL_SHARED_CUTOFF) so cross-cohort frames are
    # comparable; cutoff="own" (default) uses this cohort's own last merge,
    # keeping the full frame for single-cohort analyses.
    if len(out):
        merged_ts = pd.to_datetime(out["merged_at"])
        cut_ts = shared_cut if cutoff == "shared" else merged_ts.max()
        out["censored_14d"] = (
            merged_ts + pd.Timedelta(days=WINDOW_DAYS_SHORT) > cut_ts
        ).astype(int)
        out["censored_30d"] = (
            merged_ts + pd.Timedelta(days=WINDOW_DAYS_LONG) > cut_ts
        ).astype(int)
    else:
        out["censored_14d"] = []
        out["censored_30d"] = []

    return out[[
        "pr_id", "agent", "repo_id", "merged_at",
        "has_followup_14d", "has_followup_30d",
        "n_followups_14d", "n_followups_30d",
        "censored_14d", "censored_30d",
    ]]


def per_agent_summary(per_pr: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-PR follow-up flags to per-agent rates.

    Reports both the *naive* rate (all PRs) and the *uncensored* rate
    (PRs whose 30-day window fell inside AIDev's collection window).
    """
    rows = []
    for agent, sub in per_pr.groupby("agent"):
        n = len(sub)
        unc30 = sub[sub["censored_30d"] == 0]
        unc14 = sub[sub["censored_14d"] == 0]
        rows.append({
            "agent":               agent,
            "n_prs":               n,
            "n_uncensored_14d":    len(unc14),
            "n_uncensored_30d":    len(unc30),
            "pct_followup_14d":    100.0 * sub["has_followup_14d"].mean() if n else 0.0,
            "pct_followup_30d":    100.0 * sub["has_followup_30d"].mean() if n else 0.0,
            "pct_followup_14d_unc":
                100.0 * unc14["has_followup_14d"].mean() if len(unc14) else float("nan"),
            "pct_followup_30d_unc":
                100.0 * unc30["has_followup_30d"].mean() if len(unc30) else float("nan"),
            "mean_n_followups_30d": float(sub["n_followups_30d"].mean()) if n else 0.0,
        })
    out = pd.DataFrame(rows)
    return out.sort_values("n_prs", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Low-touch merge vs follow-up correlation
# ---------------------------------------------------------------------------


def low_touch_followup_correlation(per_pr: pd.DataFrame) -> pd.DataFrame:
    """2x2 contingency: human-touch (n_human_commits) vs follow-up at 30d.

    "High-touch" = ``n_human_commits`` strictly above the *per-agent
    median* of human-authored commits (per the author-class classifier),
    i.e. PRs that needed more human commits than the typical PR for that
    agent before merge. "Low-touch" is at-or-below the median.

    This replaces the retired ``n_by_other`` split, which conflated
    human reviewers with bot logins (e.g. the agent committing under a
    separate bot account). Each agent gets its own 2x2 table and
    chi-squared test, plus an aggregated ``__ALL__`` row. ``rate_diff_pp``
    is the follow-up rate in high-touch minus low-touch PRs.

    Cohort restriction (mirrors the paper's RQ1.1 contingency):

    * **Uncensored 30d only** (``censored_30d == 0``): the follow-up
      outcome is only observable for PRs whose full 30-day window fell
      inside AIDev's collection window, so the human-touch x follow-up
      contingency is computed on the same uncensored population that
      yields the headline 22.9\\% follow-up rate. Including censored PRs
      (whose follow-ups are unobservable) deflates both arms and the
      aggregate, and is not what the paper reports.
    * **Codex excluded**: OpenAI Codex commits under the developer's
      login with <2\\% tagged (RQ1.2), so its ``n_human_commits`` is
      unmeasurable and a high/low human-touch split is undefined for it.
      The ``__ALL__`` row is therefore the four attributable agents.
    """
    signals = per_pr_compute()[["pr_id", "agent", "n_human_commits"]]
    # ``per_pr_signals`` now populates ``n_human_commits`` for every
    # cohort (fillna 0), but guard against any stray NaN.
    signals = signals.dropna(subset=["n_human_commits"])
    signals["n_human_commits"] = signals["n_human_commits"].astype(int)

    merged = per_pr.merge(
        signals[["pr_id", "n_human_commits"]], on="pr_id", how="inner"
    )
    # Restrict to the paper's RQ1.1 contingency population: uncensored
    # 30-day window and the four agents with a measurable human-commit
    # split (Codex's is unmeasurable; see docstring).
    merged = merged[
        (merged["censored_30d"] == 0) & (merged["agent"] != "OpenAI_Codex")
    ].copy()
    # Per-agent median split on human-authored commits.
    med = merged.groupby("agent")["n_human_commits"].transform("median")
    merged["high_touch"] = (merged["n_human_commits"] > med).astype(int)

    rows = []
    for agent_label, sub in list(merged.groupby("agent")) + [("__ALL__", merged)]:
        if len(sub) == 0:
            continue
        # 2x2:                   followup=1  followup=0
        # high_touch=1           a           b
        # high_touch=0 (low)     c           d
        a = int(((sub["high_touch"] == 1) & (sub["has_followup_30d"] == 1)).sum())
        b = int(((sub["high_touch"] == 1) & (sub["has_followup_30d"] == 0)).sum())
        c = int(((sub["high_touch"] == 0) & (sub["has_followup_30d"] == 1)).sum())
        d = int(((sub["high_touch"] == 0) & (sub["has_followup_30d"] == 0)).sum())

        table = np.array([[a, b], [c, d]])
        if table.min() == 0 or table.sum(axis=0).min() == 0 or table.sum(axis=1).min() == 0:
            chi2, p, dof = float("nan"), float("nan"), 0
        else:
            chi2, p, dof, _ = chi2_contingency(table, correction=True)

        rate_high = a / max(a + b, 1)
        rate_low  = c / max(c + d, 1)
        rows.append({
            "agent":               agent_label,
            "n":                   int(table.sum()),
            "n_low_touch":         int(c + d),
            "n_high_touch":        int(a + b),
            "pct_followup_low":    100.0 * rate_low,
            "pct_followup_high":   100.0 * rate_high,
            "rate_diff_pp":        100.0 * (rate_high - rate_low),
            "chi2":                chi2,
            "p_value":             p,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _render_per_agent(summary: pd.DataFrame, n_total: int) -> Table:
    table = Table(
        title=f"RQ1 follow-up bug-fix rate per agent (N={n_total:,} merged PRs)"
    )
    table.add_column("Agent")
    table.add_column("N PRs", justify="right")
    table.add_column("% follow-up 14d", justify="right")
    table.add_column("% follow-up 30d", justify="right")
    table.add_column("% 14d (uncensored)", justify="right")
    table.add_column("% 30d (uncensored)", justify="right")
    table.add_column("mean # follow-ups 30d", justify="right")
    for r in summary.itertuples():
        unc14 = (f"{r.pct_followup_14d_unc:.1f}%"
                 if pd.notna(r.pct_followup_14d_unc) else "n/a")
        unc30 = (f"{r.pct_followup_30d_unc:.1f}%"
                 if pd.notna(r.pct_followup_30d_unc) else "n/a")
        table.add_row(
            str(r.agent),
            f"{int(r.n_prs):,}",
            f"{r.pct_followup_14d:.1f}%",
            f"{r.pct_followup_30d:.1f}%",
            unc14,
            unc30,
            f"{r.mean_n_followups_30d:.3f}",
        )
    return table


def _render_contingency(cont: pd.DataFrame) -> Table:
    table = Table(
        title="RQ1 low-touch merge x follow-up (chi-squared, 30d window)"
    )
    table.add_column("Agent")
    table.add_column("N", justify="right")
    table.add_column("# low-touch", justify="right")
    table.add_column("% follow-up | low-touch",  justify="right")
    table.add_column("% follow-up | high-touch", justify="right")
    table.add_column("Δ (pp)", justify="right")
    table.add_column("chi^2",  justify="right")
    table.add_column("p",      justify="right")
    for r in cont.itertuples():
        chi = "n/a" if not np.isfinite(r.chi2) else f"{r.chi2:.2f}"
        p   = "n/a" if not np.isfinite(r.p_value) else f"{r.p_value:.2e}"
        table.add_row(
            str(r.agent),
            f"{int(r.n):,}",
            f"{int(r.n_low_touch):,}",
            f"{r.pct_followup_low:.1f}%",
            f"{r.pct_followup_high:.1f}%",
            f"{r.rate_diff_pp:+.1f}",
            chi,
            p,
        )
    return table


def report() -> None:
    """Run the full RQ1 behavioural pipeline and persist artifacts."""
    ensure_dirs()
    console.log("[dim]computing per-PR follow-up flags...[/dim]")
    per_pr = compute()

    summary = per_agent_summary(per_pr)
    cont = low_touch_followup_correlation(per_pr)

    console.print(_render_per_agent(summary, n_total=len(per_pr)))
    console.print()
    console.print(_render_contingency(cont))

    per_pr.to_csv(PER_PR_CSV, index=False)
    summary.to_csv(PER_AGENT_CSV, index=False)
    cont_path = DIGESTS_DIR / "rq1_linker_contingency.csv"
    cont.to_csv(cont_path, index=False)
    console.log(f"[green]wrote[/green] {PER_PR_CSV}")
    console.log(f"[green]wrote[/green] {PER_AGENT_CSV}")
    console.log(f"[green]wrote[/green] {cont_path}")
