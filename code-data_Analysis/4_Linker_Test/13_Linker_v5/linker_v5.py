"""linker_v5 -- Mohamed's maintenance linker (branch feature/broader-maintenance-linker)
adapted to AIDev v5 (hao-li/AIDev-7.6M), AGENT follow-ups only.

AIDev v5 has no pr_task_type, human_pull_request or human_pr_task_type, so:
  * the follow-up pool (Q) is agent PRs only (pull_request);
  * the task type is read from the PR title (Conventional Commits prefix, e.g. "fix:");
  * file overlap uses v5 pr_commit_details (table pr_files, boilerplate flagged);
  * the lake is data/duckdb/aidev_v5.duckdb (built by step2_build_lake_v5.py).
Everything else (P population, all_changes / fix_only, windows, censoring, pair export)
is the branch code unchanged. See CHANGES_linker_v5.md.

Original docstring:

RQ1 behavioural follow-up: did a merged agent PR get a subsequent
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
* ``candidate_mode`` (see ``Q_FILTER``) sets the Q pool: ``fix_only`` is the
  fix/revert rule above and the default everywhere in this module;
  ``all_changes`` drops only that gate, for maintenance-candidate discovery.

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

import argparse
import json
import numbers
import re
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table
from scipy.stats import chi2_contingency

import os

import duckdb

from prfix.config import DATA_DIR, DIGESTS_DIR, PROCESSED_DIR, PROJECT_ROOT, ensure_dirs

# v5 lake (built by step2_build_lake_v5.py); override with the AIDEV_V5_DB variable.
V5_DB = Path(os.environ.get("AIDEV_V5_DB", DATA_DIR / "duckdb" / "aidev_v5.duckdb"))


def connect(read_only: bool = False) -> duckdb.DuckDBPyConnection:
    """Open the AIDev v5 lake (instead of the v3 lake prfix.duckdb)."""
    if not V5_DB.exists():
        raise FileNotFoundError(f"{V5_DB} not found - run step2_build_lake_v5.py first")
    return duckdb.connect(str(V5_DB), read_only=read_only)

console = Console()


# Per-PR CSVs are cached so the contingency-test step can be re-run
# without re-querying DuckDB.
PER_PR_CSV = DIGESTS_DIR / "rq1_linker_per_pr.csv"
PER_AGENT_CSV = DIGESTS_DIR / "rq1_linker_per_agent.csv"


# Follow-up windows, in days. ORIGINAL_WINDOWS is the paper's 14 / 30-day
# pair and the default of compute(); any other ascending set of positive
# integer day counts can be passed as ``windows``.
WINDOW_DAYS_SHORT = 14
WINDOW_DAYS_LONG = 30
ORIGINAL_WINDOWS = (WINDOW_DAYS_SHORT, WINDOW_DAYS_LONG)

# How a (P, Q) pair is assigned to each window.
#   original -- upstream behaviour and compute()'s default: the longest window
#               is the SQL range predicate (q_merged <= p_merged + N days,
#               exact), every shorter window is a calendar-day cut on
#               ``day_gap`` (DATE_DIFF('day')). So a Q merged 14 d 6 h after P
#               counts as "within 14 days" when it lands on calendar day 14.
#               Reproduces the paper's 14 / 30-day columns.
#   elapsed  -- every window is exact elapsed time (q_merged - p_merged <= N
#               days), the definition the longest window already uses; a
#               window's counts then do not depend on which other windows are
#               configured (30 d under [30, 60] equals the paper's 30 d).
WINDOW_RULES = ("original", "elapsed")


# Core SQL: produces one row per (P, Q) pair where Q is a candidate
# follow-up (fix/revert under ``fix_only``) within the longest window
# (MAX_WINDOW days; 30 in the paper) that shares at least one filename
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
}

SQL_FOLLOWUP_PAIRS = """
WITH p_pop AS (
{P_POP}
),
q_all AS (
    -- Every merged follow-up PR in the AIDev-pop repos, with its task type
    -- and revert-title flag kept as metadata; the candidate mode (Q_FILTER,
    -- applied in ``q_fix``) decides which of them may pair with a P.
    -- AIDev-pop agent PRs: task type from AIDev's Conventional-Commits
    -- classifier. We use ``pull_request`` (not ``all_pull_request``) because
    -- ``pr_commit_details`` and ``pr_task_type`` are only published for
    -- AIDev-pop, so we cannot check file overlap or task type outside
    -- that subset.
    SELECT
        q.id        AS q_id,
        q.repo_id   AS repo_id,
        TRY_CAST(q.merged_at AS TIMESTAMP) AS q_merged,
        q.title     AS q_title,
        COALESCE(t5.type, '(none)') AS q_type,  -- v5: type rebuilt by step2 (title prefix or classifier)
        (q.title LIKE 'Revert %' OR q.title LIKE 'revert %') AS q_is_revert
    FROM pull_request q
    JOIN repository   r ON r.id = q.repo_id
    LEFT JOIN pr_task_type_v5 t5 ON t5.id = q.id
    WHERE q.merged_at IS NOT NULL
      AND r.stars >= 500

),
q_fix AS (
    -- Candidate follow-up PRs under the selected candidate mode (see
    -- Q_FILTER): ``fix_only`` keeps the upstream rule -- typed ``fix`` OR a
    -- revert-style title -- and ``all_changes`` keeps every merged PR.
    SELECT * FROM q_all WHERE {Q_FILTER}
),
pair AS (
    -- Self-join: candidate follow-ups in the same repo, merged
    -- strictly after P but within the longest window. Skip self-matches.
    SELECT
        p.p_id,
        p.agent,
        p.repo_id,
        p.p_merged,
        q.q_id,
        q.q_merged,
        q.q_title,
        q.q_type,
        q.q_is_revert,
        DATE_DIFF('day', p.p_merged, q.q_merged) AS day_gap
    FROM p_pop p
    JOIN q_fix q
      ON q.repo_id = p.repo_id
     AND q.q_merged >  p.p_merged
     AND q.q_merged <= p.p_merged + INTERVAL {MAX_WINDOW} DAY
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
    -- File overlap: keep (P, Q) pairs that share at least one filename, and
    -- list the shared filenames (sorted, distinct) for the pair export.
    SELECT
        pair.p_id,
        pair.agent,
        pair.q_id,
        pair.day_gap,
        pair.q_title,
        pair.q_type,
        pair.q_is_revert,
        pair.repo_id,
        pair.p_merged,
        pair.q_merged,
        list_sort(list_distinct(list(pf.filename))) AS shared_files
    FROM pair
    JOIN all_commit_files pf
      ON pf.pr_id = pair.p_id
    JOIN all_commit_files qf
      ON qf.pr_id = pair.q_id
     AND qf.filename = pf.filename
    GROUP BY pair.p_id, pair.agent, pair.q_id, pair.day_gap,
             pair.q_title, pair.q_type, pair.q_is_revert,
             pair.repo_id, pair.p_merged, pair.q_merged
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
-- v5: agent PRs only, so the shared cut-off is the agent table's last merge.
SELECT MAX(TRY_CAST(q.merged_at AS TIMESTAMP)) AS shared_cutoff
FROM pull_request q JOIN repository r ON r.id = q.repo_id
WHERE q.merged_at IS NOT NULL AND r.stars >= 500
"""


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------


# File-set source per linker mode (substituted into the all_commit_files CTE).
#   net_sub  -- net-diff base->head co-edit, boilerplate excluded (DEFAULT/primary)
#   net_any  -- net-diff co-edit, boilerplate included
#   generic  -- per-commit union (legacy; over-links; kept for sensitivity)
FILES_SOURCE = {
    # v5: per-PR files from pr_commit_details (table pr_files, step2), boilerplate flagged
    # with prfix.rq1_fix_rates.boilerplate.is_boilerplate. v5 has no net diffs.
    "commit_sub": "SELECT pr_id, filename FROM pr_files WHERE NOT is_boilerplate",
    "commit_any": "SELECT pr_id, filename FROM pr_files",
}


# Follow-up (Q) selection per candidate mode (substituted into the q_fix CTE).
#   fix_only     -- upstream rule: AIDev task type ``fix`` OR a revert-style
#                   title. Reproduces the paper's candidate set.
#   all_changes  -- every later merged PR that passes the same repository,
#                   temporal and file-overlap criteria, whatever its task type
#                   (including PRs with no task-type row). Task type and the
#                   revert flag stay on each pair as metadata; an all_changes
#                   pair is a candidate, never evidence of a corrective change.
Q_FILTER = {
    "fix_only": "q_type = 'fix' OR q_is_revert",
    "all_changes": "TRUE",
}
CANDIDATE_MODES = tuple(Q_FILTER)


def _check_candidate_mode(candidate_mode: str) -> None:
    if candidate_mode not in Q_FILTER:
        raise ValueError(f"unknown candidate_mode {candidate_mode!r}; "
                         f"choose {list(CANDIDATE_MODES)}")


def _check_windows(windows) -> tuple[int, ...]:
    """Window lengths in days, validated, de-duplicated and ascending."""
    ws = tuple(windows)
    if not ws or any(isinstance(w, bool) or not isinstance(w, numbers.Integral) or w <= 0
                     for w in ws):
        raise ValueError(f"windows must be one or more positive integer day counts, "
                         f"got {windows!r}")
    return tuple(sorted({int(w) for w in ws}))


def _check_window_rule(window_rule: str) -> None:
    if window_rule not in WINDOW_RULES:
        raise ValueError(f"unknown window_rule {window_rule!r}; choose {list(WINDOW_RULES)}")


def _windows_of(per_pr: pd.DataFrame) -> list[int]:
    """The windows a per-PR frame carries, read off its has_followup_<N>d columns."""
    found = (re.fullmatch(r"has_followup_(\d+)d", c) for c in per_pr.columns)
    return sorted(int(m.group(1)) for m in found if m)


def _within_window(pairs: pd.DataFrame, window: int, max_window: int,
                   window_rule: str) -> pd.Series:
    """Which (P, Q) pairs fall inside a ``window``-day window (see WINDOW_RULES)."""
    if window_rule == "elapsed":
        return (pairs["q_merged"] - pairs["p_merged"]) <= pd.Timedelta(days=window)
    if window == max_window:
        # The SQL range predicate already bounds every pair by the longest window.
        return pd.Series(True, index=pairs.index)
    return pairs["day_gap"] <= window


def pairs_prelude_sql(p_side: str = "agent", candidate_mode: str = "fix_only",
                      max_window: int = WINDOW_DAYS_LONG) -> str:
    """The pair-CTE prelude (p_pop + q_fix + pair), substituted for a P side,
    WITHOUT the file-overlap stage -- for callers that enumerate time/repo
    candidate pairs regardless of linker mode (netdiff fetch, sensitivity)."""
    if p_side not in P_POP:
        raise ValueError(f"unknown p_side {p_side!r}; choose {list(P_POP)}")
    _check_candidate_mode(candidate_mode)
    (max_window,) = _check_windows((max_window,))
    full = (SQL_FOLLOWUP_PAIRS
            .replace("{P_POP}", P_POP[p_side])
            .replace("{Q_FILTER}", Q_FILTER[candidate_mode])
            .replace("{MAX_WINDOW}", str(max_window)))
    return full.split("all_commit_files AS")[0].rstrip().rstrip(",")


def pairs_sql(linker: str = "commit_sub", p_side: str = "agent",
              candidate_mode: str = "fix_only", max_window: int = WINDOW_DAYS_LONG) -> str:
    """The fully substituted follow-up pair SQL for a linker + P-side cohort
    + candidate mode (see Q_FILTER), pairing Q merged at most ``max_window``
    days after P. The defaults reproduce the paper."""
    if linker not in FILES_SOURCE:
        raise ValueError(f"unknown linker {linker!r}; choose {list(FILES_SOURCE)}")
    if p_side not in P_POP:
        raise ValueError(f"unknown p_side {p_side!r}; choose {list(P_POP)}")
    _check_candidate_mode(candidate_mode)
    (max_window,) = _check_windows((max_window,))
    return (SQL_FOLLOWUP_PAIRS
            .replace("{P_POP}", P_POP[p_side])
            .replace("{Q_FILTER}", Q_FILTER[candidate_mode])
            .replace("{MAX_WINDOW}", str(max_window))
            .replace("{FILES_SOURCE}", FILES_SOURCE[linker]))


def file_source_coverage(linker: str = "commit_sub", p_side: str = "agent",
                         candidate_mode: str = "fix_only",
                         max_window: int = WINDOW_DAYS_LONG) -> dict:
    """How many PRs in the time/repo candidate pairs have no file rows at all.

    Such PRs can never pass the file-overlap test, so their pairs vanish
    silently. For the net-diff linkers this is the PRs missing from
    ``pr_net_files`` (never fetched, unavailable, or an empty diff) -- upstream
    fetches net diffs only for fix_only / 30-day pairs, so a wider candidate
    mode or window needs ``fetch_netdiff --candidate-mode ... --max-window ...``
    first. Boilerplate-only PRs count as covered: they have rows.
    """
    source = FILES_SOURCE["commit_any"] if linker.startswith("commit") else FILES_SOURCE[linker]
    sql = pairs_prelude_sql(p_side, candidate_mode, max_window) + f"""
    , ids AS (SELECT p_id AS id FROM pair UNION SELECT q_id FROM pair)
    , known AS (SELECT DISTINCT pr_id FROM ({source}))
    SELECT ids.id FROM ids LEFT JOIN known ON known.pr_id = ids.id
    WHERE known.pr_id IS NULL
    ORDER BY ids.id
    """
    count_sql = pairs_prelude_sql(p_side, candidate_mode, max_window) + """
    SELECT COUNT(*) FROM (SELECT p_id FROM pair UNION SELECT q_id FROM pair)
    """
    con = connect(read_only=True)
    try:
        missing = [int(i) for (i,) in con.execute(sql).fetchall()]
        n_pair_prs = int(con.execute(count_sql).fetchone()[0])
    finally:
        con.close()
    return {"n_pair_prs": n_pair_prs, "n_without_files": len(missing),
            "ids_without_files_sample": missing[:20]}


def compute(linker: str = "commit_sub", p_side: str = "agent",
            cutoff: str = "own", candidate_mode: str = "fix_only",
            windows=ORIGINAL_WINDOWS, window_rule: str = "original") -> pd.DataFrame:
    """Return one row per merged AIDev-pop PR with follow-up flags.

    ``candidate_mode`` selects which later PRs count as follow-ups (see
    Q_FILTER): "fix_only" (default, the paper's fix/revert rule) or
    "all_changes" (any task type).
    ``windows`` are the observation windows in days (default 14 and 30, the
    paper's); each window N yields has_followup_Nd, n_followups_Nd and
    censored_Nd. ``window_rule`` decides how a pair is assigned to a window
    (see WINDOW_RULES; "original" reproduces the paper).
    The selected mode, windows, window rule, linker, P side, cut-off and the
    observation horizon used for censoring are recorded in the returned
    frame's ``attrs``.
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

    Columns (default windows):
        pr_id, agent, merged_at,
        has_followup_14d, has_followup_30d,
        n_followups_30d, n_followups_14d,
        censored_14d, censored_30d
    """
    if cutoff not in ("own", "shared"):
        raise ValueError(f"unknown cutoff {cutoff!r}; choose 'own' or 'shared'")
    windows = _check_windows(windows)
    _check_window_rule(window_rule)
    max_window = windows[-1]
    sql = pairs_sql(linker, p_side, candidate_mode, max_window)
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
    out = cohort.rename(columns={"pr_id": "pr_id"}).copy()
    for w in windows:
        if len(pairs):
            agg = pairs[_within_window(pairs, w, max_window, window_rule)] \
                      .groupby("p_id").size().rename(f"n_followups_{w}d")
        else:
            agg = pd.Series(dtype=int, name=f"n_followups_{w}d")
        out = out.merge(agg, left_on="pr_id", right_index=True, how="left")
        out[f"n_followups_{w}d"] = out[f"n_followups_{w}d"].fillna(0).astype(int)
        out[f"has_followup_{w}d"] = (out[f"n_followups_{w}d"] > 0).astype(int)

    # Censoring: PRs whose N-day window extends past the cut-off date
    # can't possibly observe later follow-ups. cutoff="shared" uses the same
    # date for every p_side (see SQL_SHARED_CUTOFF) so cross-cohort frames are
    # comparable; cutoff="own" (default) uses this cohort's own last merge,
    # keeping the full frame for single-cohort analyses. Either way the cut-off
    # is the data's observation horizon, never an assumed full window.
    cut_ts = None
    if len(out):
        merged_ts = pd.to_datetime(out["merged_at"])
        cut_ts = shared_cut if cutoff == "shared" else merged_ts.max()
        for w in windows:
            out[f"censored_{w}d"] = (merged_ts + pd.Timedelta(days=w) > cut_ts).astype(int)
    else:
        for w in windows:
            out[f"censored_{w}d"] = []

    out = out[["pr_id", "agent", "repo_id", "merged_at",
               *(f"has_followup_{w}d" for w in windows),
               *(f"n_followups_{w}d" for w in windows),
               *(f"censored_{w}d" for w in windows)]]
    out.attrs.update(candidate_mode=candidate_mode, file_match_mode=linker,
                     p_side=p_side, cutoff=cutoff, windows=list(windows),
                     window_rule=window_rule,
                     observation_horizon=None if cut_ts is None else str(cut_ts))
    return out


# Per-window follow-up status (see followup_status).
FOLLOWUP_STATUSES = ("observed", "none_observed", "censored")


def followup_status(per_pr: pd.DataFrame) -> pd.DataFrame:
    """Add a ``followup_status_<N>d`` column per window of a compute() frame.

    ``has_followup_Nd == 0`` alone cannot tell "no follow-up" from "window not
    fully observed". The status separates them:
      * observed      -- at least one follow-up seen (even if the window is
                         cut short by the observation horizon);
      * none_observed -- whole window observed, no follow-up seen;
      * censored      -- window runs past the observation horizon and no
                         follow-up was seen: unknown, NOT "no follow-up".
    """
    out = per_pr.copy()
    for w in _windows_of(per_pr):
        out[f"followup_status_{w}d"] = np.select(
            [out[f"has_followup_{w}d"] == 1, out[f"censored_{w}d"] == 1],
            ["observed", "censored"], default="none_observed")
    return out


# ---------------------------------------------------------------------------
# Candidate-pair export
# ---------------------------------------------------------------------------


# Which PR table(s) list each PR id -- the only evidence the lake carries on
# who authored a follow-up. ``pull_request`` holds AIDev's agent PRs and
# ``human_pull_request`` its human cohort; an id in both (or neither) is
# "unknown", and an id with several agent labels gets no agent.
SQL_PR_SOURCE = """
SELECT
    id AS q_id,
    BOOL_OR(src = 'agent') AS in_agent_table,
    BOOL_OR(src = 'human') AS in_human_table,
    CASE WHEN COUNT(DISTINCT agent) = 1 THEN MAX(agent) END AS q_agent
FROM (
    SELECT id, 'agent' AS src, agent FROM pull_request
)
GROUP BY id
"""

# Every exported pair carries this label: temporal + file overlap makes a
# candidate for code-level analysis, not a verified corrective relationship.
RELATIONSHIP_STATUS = "unverified_candidate"


def candidate_pairs(linker: str = "commit_sub", p_side: str = "agent",
                    candidate_mode: str = "fix_only", windows=ORIGINAL_WINDOWS,
                    window_rule: str = "original") -> pd.DataFrame:
    """One row per (original PR, follow-up PR) candidate pair.

    Takes the same selection arguments as compute() and returns the pairs it
    counts, longest window included, with ``within_<N>d`` flags per window,
    the shared filenames, the follow-up's task type / revert flag / agent /
    author type as metadata, and the run's candidate mode, file-match mode and
    window rule. Missing metadata stays missing (NA); no value is imputed.
    Rows are sorted by (original_pr_id, followup_pr_id). Two rows for one pair
    (e.g. a follow-up with two task-type rows) raise ValueError instead of one
    being dropped silently.
    """
    windows = _check_windows(windows)
    _check_window_rule(window_rule)
    sql = pairs_sql(linker, p_side, candidate_mode, windows[-1])
    con = connect(read_only=True)
    try:
        pairs = con.execute(sql).df()
        sources = con.execute(SQL_PR_SOURCE).df()
    finally:
        con.close()

    dup = pairs.duplicated(["p_id", "q_id"], keep=False)
    if dup.any():
        bad = sorted(set(zip(pairs.p_id[dup], pairs.q_id[dup], strict=True)))
        raise ValueError(
            f"{len(bad)} duplicate (original_pr_id, followup_pr_id) pairs, e.g. {bad[:5]}: "
            "their rows disagree on follow-up metadata (task type, title, ...). "
            "Resolve the source rows before exporting.")

    pairs = pairs.merge(sources, on="q_id", how="left")
    in_agent = pairs["in_agent_table"].eq(True)
    in_human = pairs["in_human_table"].eq(True)
    out = pd.DataFrame({
        "original_pr_id": pairs["p_id"],
        "followup_pr_id": pairs["q_id"],
        "repository_id": pairs["repo_id"],
        "original_agent": pairs["agent"],
        "original_merged_at": pairs["p_merged"],
        "followup_merged_at": pairs["q_merged"],
        "days_after_merge": (pairs["q_merged"] - pairs["p_merged"]) / pd.Timedelta(days=1),
        "day_gap": pairs["day_gap"],
    })
    for w in windows:
        out[f"within_{w}d"] = _within_window(pairs, w, windows[-1], window_rule)
    out["shared_files"] = [[str(f) for f in files] for files in pairs["shared_files"]]
    out["shared_file_count"] = out["shared_files"].map(len).astype(int)
    out["followup_task_type"] = pairs["q_type"].where(pairs["q_type"] != "(none)") \
                                               .astype("string")
    out["is_revert"] = pairs["q_is_revert"].astype("boolean")
    out["followup_agent"] = pairs["q_agent"].astype("string")
    out["followup_author_type"] = np.select(
        [in_agent & ~in_human, in_human & ~in_agent], ["agent", "human"], default="unknown")
    out["candidate_mode"] = candidate_mode
    out["file_match_mode"] = linker
    out["window_rule"] = window_rule
    out["relationship_status"] = RELATIONSHIP_STATUS
    return out.sort_values(["original_pr_id", "followup_pr_id"]).reset_index(drop=True)


def write_candidate_pairs(pairs: pd.DataFrame, path) -> Path:
    """Write a candidate_pairs() frame to ``path``; the suffix picks the format.

    ``.parquet`` keeps ``shared_files`` as a list column; ``.csv`` stores it as
    a JSON array string and writes missing values as ``NA``.
    """
    path = Path(path)
    if path.suffix == ".parquet":
        pairs.to_parquet(path, index=False)
    elif path.suffix == ".csv":
        flat = pairs.assign(shared_files=pairs["shared_files"].map(json.dumps))
        flat.to_csv(path, index=False, na_rep="NA")
    else:
        raise ValueError(f"candidate pairs export needs a .csv or .parquet path, got {path}")
    return path


def per_agent_summary(per_pr: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-PR follow-up flags to per-agent rates.

    Reports both the *naive* rate (all PRs) and the *uncensored* rate
    (PRs whose N-day window fell inside AIDev's collection window), for every
    window the per-PR frame carries; the mean follow-up count is taken over
    the longest window.
    """
    windows = _windows_of(per_pr)
    longest = windows[-1]
    rows = []
    for agent, sub in per_pr.groupby("agent"):
        n = len(sub)
        unc = {w: sub[sub[f"censored_{w}d"] == 0] for w in windows}
        row = {"agent": agent, "n_prs": n}
        row.update({f"n_uncensored_{w}d": len(unc[w]) for w in windows})
        row.update({f"pct_followup_{w}d":
                    100.0 * sub[f"has_followup_{w}d"].mean() if n else 0.0
                    for w in windows})
        row.update({f"pct_followup_{w}d_unc":
                    100.0 * unc[w][f"has_followup_{w}d"].mean() if len(unc[w])
                    else float("nan")
                    for w in windows})
        row[f"mean_n_followups_{longest}d"] = (
            float(sub[f"n_followups_{longest}d"].mean()) if n else 0.0)
        rows.append(row)
    out = pd.DataFrame(rows)
    return out.sort_values("n_prs", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _render_per_agent(summary: pd.DataFrame, n_total: int) -> Table:
    windows = sorted(int(m.group(1)) for m in
                     (re.fullmatch(r"pct_followup_(\d+)d", c) for c in summary.columns) if m)
    longest = windows[-1]
    table = Table(
        title=f"RQ1 follow-up bug-fix rate per agent (N={n_total:,} merged PRs)"
    )
    table.add_column("Agent")
    table.add_column("N PRs", justify="right")
    for w in windows:
        table.add_column(f"% follow-up {w}d", justify="right")
    for w in windows:
        table.add_column(f"% {w}d (uncensored)", justify="right")
    table.add_column(f"mean # follow-ups {longest}d", justify="right")
    for r in summary.to_dict("records"):
        unc = [f"{r[f'pct_followup_{w}d_unc']:.1f}%"
               if pd.notna(r[f"pct_followup_{w}d_unc"]) else "n/a" for w in windows]
        table.add_row(
            str(r["agent"]),
            f"{int(r['n_prs']):,}",
            *(f"{r[f'pct_followup_{w}d']:.1f}%" for w in windows),
            *unc,
            f"{r[f'mean_n_followups_{longest}d']:.3f}",
        )
    return table


# ---------------------------------------------------------------------------
# Command line: maintenance-candidate discovery
# ---------------------------------------------------------------------------


# Command-line defaults are the maintenance study's configuration. The library
# functions above keep the paper's defaults (fix_only, 14 / 30 days, original
# window rule), so report(), the consolidation gate and the annotation scripts
# reproduce the paper unchanged.
DEFAULT_CANDIDATE_MODE = "all_changes"
OBSERVATION_WINDOWS = [30, 60]
DEFAULT_WINDOW_RULE = "elapsed"
DEFAULT_OUTPUT_DIR = PROCESSED_DIR / "maintenance_linker_v5"

METHOD_WARNING = (
    "Candidate follow-up changes are identified through temporal and file-level "
    "relationships. These relationships do not establish corrective intent or "
    "demonstrate that the original AI-generated contribution introduced a "
    "deficiency. Such claims require subsequent code-level analysis and validation."
)


def _git_revision() -> str | None:
    """HEAD of the checkout running the linker (``+dirty`` if uncommitted
    changes), or None outside a Git checkout."""
    try:
        def git(*args: str) -> str:
            return subprocess.run(["git", *args], cwd=PROJECT_ROOT, capture_output=True,
                                  text=True, check=True).stdout.strip()
        sha, dirty = git("rev-parse", "HEAD"), git("status", "--porcelain")
    except (OSError, subprocess.CalledProcessError):
        return None
    return sha + ("+dirty" if dirty else "")


def main(argv: list[str] | None = None) -> int:
    """Write per-PR flags, per-agent rates and candidate pairs for one
    configuration, plus a run_metadata.json describing it."""
    ap = argparse.ArgumentParser(
        prog="python linker_v5.py",
        description="Follow-up candidate discovery for merged PRs. Defaults are the "
                    "maintenance study's (all_changes, 30/60 days, elapsed windows); "
                    "--candidate-mode fix_only --windows 14 30 --window-rule original "
                    "is the paper's baseline.")
    ap.add_argument("--candidate-mode", choices=CANDIDATE_MODES, default=DEFAULT_CANDIDATE_MODE,
                    help="which later PRs may be follow-ups (default: %(default)s)")
    ap.add_argument("--windows", type=int, nargs="+", default=OBSERVATION_WINDOWS,
                    metavar="DAYS", help="observation windows in days (default: 30 60)")
    ap.add_argument("--window-rule", choices=WINDOW_RULES, default=DEFAULT_WINDOW_RULE,
                    help="how pairs are assigned to windows (default: %(default)s)")
    ap.add_argument("--file-match-mode", choices=sorted(FILES_SOURCE), default="commit_sub",
                    help="file-overlap source, the upstream 'linker' (default: %(default)s)")
    ap.add_argument("--p-side", choices=sorted(P_POP), default="agent",
                    help="cohort of original PRs (default: %(default)s)")
    ap.add_argument("--cutoff", choices=["own"], default="own",
                    help="censoring cut-off (default: %(default)s)")
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
                    help="directory for the outputs (default: data/processed/maintenance_linker)")
    ap.add_argument("--pairs-format", choices=["csv", "parquet", "both"], default="csv",
                    help="candidate_pairs file format(s) (default: %(default)s)")
    args = ap.parse_args(argv)
    try:
        windows = list(_check_windows(args.windows))
    except ValueError as exc:
        ap.error(str(exc))

    selection = dict(linker=args.file_match_mode, p_side=args.p_side,
                     candidate_mode=args.candidate_mode, windows=windows,
                     window_rule=args.window_rule)
    per_pr = compute(cutoff=args.cutoff, **selection)
    horizon = per_pr.attrs["observation_horizon"]
    per_pr = followup_status(per_pr)
    summary = per_agent_summary(per_pr)
    pairs = candidate_pairs(**selection)
    coverage = file_source_coverage(args.file_match_mode, args.p_side,
                                    args.candidate_mode, windows[-1])

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    written = [out / "per_pr.csv", out / "per_agent.csv"]
    per_pr.to_csv(written[0], index=False)
    summary.to_csv(written[1], index=False)
    formats = ["csv", "parquet"] if args.pairs_format == "both" else [args.pairs_format]
    written += [write_candidate_pairs(pairs, out / f"candidate_pairs.{fmt}") for fmt in formats]

    meta = {
        "candidate_mode": args.candidate_mode,
        "windows": windows,
        "window_rule": args.window_rule,
        "file_match_mode": args.file_match_mode,
        "p_side": args.p_side,
        "cutoff": args.cutoff,
        "observation_horizon": horizon,
        "n_original_prs": len(per_pr),
        "n_censored": {f"{w}d": int(per_pr[f"censored_{w}d"].sum()) for w in windows},
        "n_candidate_pairs": len(pairs),
        "file_source_coverage": coverage,
        "relationship_status": RELATIONSHIP_STATUS,
        "methodological_warning": METHOD_WARNING,
        "outputs": [p.name for p in written],
        "code_revision": _git_revision(),
        "dataset": "AIDev v5 (hao-li/AIDev-7.6M), agent follow-ups only",
        "lake": str(V5_DB),
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    meta_path = out / "run_metadata.json"
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")

    console.print(_render_per_agent(summary, n_total=len(per_pr)))
    for p in [*written, meta_path]:
        console.log(f"[green]wrote[/green] {p}")
    if coverage["n_without_files"]:
        console.print(
            f"[yellow]{coverage['n_without_files']:,} of {coverage['n_pair_prs']:,} PRs in "
            f"the time/repo candidate pairs have no file rows ({args.file_match_mode}); "
            "their pairs cannot be file-matched (no rows in v5 pr_commit_details).[/yellow]")
    console.print(f"[yellow]{METHOD_WARNING}[/yellow]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
