"""Repair survival (KM / stratified Cox), per-PR commit-author metrics for
the author-class tables, and the cross-cohort Mantel-Haenszel helpers.

Everything here is deterministic (no RNG) and asserted by
human-annotation/consolidation_gate.py before any number is typeset.
Plot scripts (prfix.rq1_fix_rates.fig_survival,
prfix.rq3_commit_authors.commit_tables) import from this module so figures
and gate always agree.

Conventions:
  * Verified pair = judge Direct label (ERROR -> not-verified) on a
    substantive case, plus the two-annotator gold Direct-fix backfill --
    identical to consolidation_gate.py.
  * Commit classification runs CONTENT-ONLY (pr_agent=None, no Codex
    prior): the prior would label opener-authored Codex commits agent by
    construction, which is circular when the question is whether opener
    identity tracks commit authorship. Codex is reported unmeasured.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from statsmodels.discrete.conditional_models import ConditionalLogit
from statsmodels.duration.hazard_regression import PHReg
from statsmodels.stats.contingency_tables import StratifiedTable

from prfix.common.effect_sizes import cliffs_delta_from_u
from prfix.common.loaders.base import connect
from prfix.config import DIGESTS_DIR, JUDGE_DIR, PROJECT_ROOT
from prfix.rq1_fix_rates import linker as rb
from prfix.rq3_commit_authors import classify_commit
from prfix.rq4_signals.per_pr_signals import compute as pps_compute

AGENT_JUDGE = JUDGE_DIR / "rq1_llm_judge_pop_anthropic_claude-opus-4-8_low.csv"
AGENT_CASES = JUDGE_DIR / "rq1_judge_cases_population_12k.json"
HUMAN_JUDGE = JUDGE_DIR / "rq1_llm_judge_pop_humanP_anthropic_claude-opus-4-8_low.csv"
HUMAN_CASES = JUDGE_DIR / "rq1_judge_cases_population_12k_humanP.json"
AGREED = (PROJECT_ROOT / "human-annotation" / "labels" / "humans"
          / "rq1_followup_validation_agreed.csv")

WINDOW = 30


def _norm(u) -> str:
    return str(u).strip().rstrip("/").lower()


def _judged(judge_csv: Path, cases_json: Path) -> pd.DataFrame:
    j = pd.read_csv(judge_csv)[["pair_id", "label"]]
    j["label"] = j["label"].fillna("ERROR")
    c = pd.DataFrame(json.loads(cases_json.read_text()))
    cols = ["pair_id", "p_id", "q_id", "q_is_human", "n_substantive", "q_url"]
    for col in ("pair_id", "p_id", "q_id", "n_substantive"):
        c[col] = c[col].astype(int)
    return j.merge(c[cols], on="pair_id", validate="1:1")


def _gold_direct() -> pd.DataFrame:
    g = pd.read_csv(AGREED)
    return g[(g.label_1 == g.label_2) & (g.label_1 == "Direct fix")]


def gold_direct_resolved(agent_prs: pd.DataFrame,
                         human_prs: pd.DataFrame) -> pd.DataFrame:
    """The two-annotator gold Direct-fix pairs with the FIXING PR resolved
    against both cohort tables: `_gold_direct()` plus `q_id` and `q_is_human`.

    Single owner of that resolution -- every consumer (this module's
    verified_fix_prs, the fixer frame in consolidation_gate.py) must go
    through it, so the paper cannot typeset two different cohort assignments
    for the same gold pair. The two tables carry independent id spaces, so
    the lookup is by URL against each map separately: the agent table wins a
    URL collision, and a Q in neither table raises rather than being dropped
    or mislabelled human.

    Both frames need `id` and `html_url`.
    """
    a_map = {_norm(u): int(i)
             for i, u in zip(agent_prs.id, agent_prs.html_url, strict=True)}
    h_map = {_norm(u): int(i)
             for i, u in zip(human_prs.id, human_prs.html_url, strict=True)}
    g = _gold_direct().copy()
    resolved = []
    for u in g.q_url:
        k = _norm(u)
        if k in a_map:
            resolved.append((a_map[k], False))
        elif k in h_map:
            resolved.append((h_map[k], True))
        else:
            raise KeyError(f"gold q_url not in either PR table: {u}")
    g["q_id"] = [q for q, _ in resolved]
    g["q_is_human"] = [h for _, h in resolved]
    return g


def verified_pair_sets() -> tuple[set, set]:
    """(agent-side, human-side) sets of verified Direct (p_id, q_id) pairs."""
    va, vh = set(), set()
    for side, (judge, cases) in (("a", (AGENT_JUDGE, AGENT_CASES)),
                                 ("h", (HUMAN_JUDGE, HUMAN_CASES))):
        m = _judged(judge, cases)
        d = m[(m.n_substantive > 0) & m.label.str.startswith("Direct")]
        pairs = {(int(p), int(q)) for p, q
                 in d[["p_id", "q_id"]].itertuples(index=False)}
        (va if side == "a" else vh).update(pairs)

    con = connect(read_only=True)
    try:
        pr = con.execute("SELECT id, html_url FROM pull_request UNION ALL "
                         "SELECT id, html_url FROM human_pull_request").df()
    finally:
        con.close()
    url2id = {_norm(u): int(i) for i, u in zip(pr.id, pr.html_url, strict=True)}
    va |= {(url2id[_norm(p)], url2id[_norm(q)])
           for p, q in _gold_direct()[["p_url", "q_url"]].itertuples(index=False)}
    return va, vh


def verified_fix_prs() -> pd.DataFrame:
    """Unique verified fix PRs (q_id, q_is_human, q_agent)."""
    frames = []
    for judge, cases in ((AGENT_JUDGE, AGENT_CASES), (HUMAN_JUDGE, HUMAN_CASES)):
        m = _judged(judge, cases)
        frames.append(m[(m.n_substantive > 0) & m.label.str.startswith("Direct")]
                      [["q_id", "q_is_human"]])

    con = connect(read_only=True)
    try:
        qa = con.execute("SELECT id, html_url, agent FROM pull_request").df()
        qh = con.execute("SELECT id, html_url FROM human_pull_request").df()
    finally:
        con.close()
    frames.append(gold_direct_resolved(qa, qh)[["q_id", "q_is_human"]])

    q = (pd.concat(frames, ignore_index=True)
         .drop_duplicates(subset=["q_id", "q_is_human"]))
    q["q_agent"] = q.q_id.map(dict(zip(qa.id, qa.agent, strict=True)))
    q.loc[q.q_is_human, "q_agent"] = "Human"
    return q


# ------------------------------------------------------------------ survival
def survival_frame(p_side: str, verified: set) -> pd.DataFrame:
    """One row per merged PR on the SHARED cut-off frame: observation days
    (capped at WINDOW), candidate/verified first-event flags and times."""
    con = connect(read_only=True)
    try:
        pairs = con.execute(rb.pairs_sql("net_sub", p_side)).df()
        cohort = con.execute(rb.SQL_COHORT[p_side]).df()
        shared_cut = pd.to_datetime(
            con.execute(rb.SQL_SHARED_CUTOFF).fetchone()[0])
    finally:
        con.close()

    if pd.isna(shared_cut):
        raise ValueError(
            "the shared cut-off date is NULL: one of the two Q-source tables "
            "(pull_request, human_pull_request, stars >= 500) has no parseable "
            "merged_at. Load / rebuild the lake before running any "
            "cross-cohort comparison.")

    cohort["merged_at"] = pd.to_datetime(cohort["merged_at"])
    cohort = cohort[cohort.merged_at <= shared_cut].copy()
    cohort["obs_days"] = np.minimum(
        (shared_cut - cohort.merged_at).dt.total_seconds() / 86400.0, WINDOW)

    first_cand = pairs.groupby("p_id").day_gap.min().rename("cand_day")
    vp = pairs[[(p, q) in verified for p, q
                in zip(pairs.p_id, pairs.q_id, strict=True)]]
    first_ver = vp.groupby("p_id").day_gap.min().rename("ver_day")
    linked = {(int(p), int(q)) for p, q
              in vp[["p_id", "q_id"]].itertuples(index=False)}

    df = (cohort
          .merge(first_cand, left_on="pr_id", right_index=True, how="left")
          .merge(first_ver, left_on="pr_id", right_index=True, how="left"))
    for kind in ("cand", "ver"):
        day = df[f"{kind}_day"]
        event = day.notna() & (day <= df.obs_days)
        df[f"{kind}_event"] = event.astype(int)
        df[f"{kind}_time"] = np.where(event, day, df.obs_days).clip(1e-6, WINDOW)
    df["cohort"] = "Human" if p_side == "human" else "Agent"
    df.attrs["verified_pairs"] = len(verified)
    df.attrs["verified_pairs_linked"] = len(linked)
    df.attrs["verified_pairs_unlinked"] = len(verified) - len(linked)
    return df


def km_cuminc(time: np.ndarray, event: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Kaplan-Meier cumulative incidence (1 - S) as step-function points."""
    order = np.argsort(time)
    t, e = time[order], event[order]
    surv = 1.0
    xs, ys = [0.0], [0.0]
    for ti in np.unique(t[e == 1]):
        d = int(((t == ti) & (e == 1)).sum())
        at_risk = int((t >= ti).sum())
        surv *= 1.0 - d / at_risk
        xs.append(ti)
        ys.append(1.0 - surv)
    xs.append(WINDOW)
    ys.append(ys[-1])
    return np.array(xs), np.array(ys)


def cox_hr(df: pd.DataFrame, kind: str) -> tuple[float, float, float, float]:
    """Repository-stratified Cox HR (Agent vs Human): (hr, lo, hi, p)."""
    x = (df.cohort == "Agent").astype(float).to_numpy()[:, None]
    fit = PHReg(df[f"{kind}_time"].to_numpy(), x,
                status=df[f"{kind}_event"].to_numpy(),
                strata=df.repo_id.to_numpy()).fit()
    b, se = float(fit.params[0]), float(fit.bse[0])
    return (float(np.exp(b)), float(np.exp(b - 1.96 * se)),
            float(np.exp(b + 1.96 * se)), float(fit.pvalues[0]))


# ------------------------------------------------------------ commit authors
def cohort_commit_classes() -> pd.DataFrame:
    """One row per commit of every merged AIDev-pop PR (agent + human
    cohorts), classified CONTENT-ONLY: columns grp, pr_id, cls."""
    con = connect(read_only=True)
    try:
        c = con.execute("""
            SELECT c.pr_id, c.author, c.committer, c.message,
                   p.agent AS grp
            FROM pr_commits c JOIN pull_request p ON p.id = c.pr_id
            JOIN repository r ON r.id = p.repo_id
            WHERE p.merged_at IS NOT NULL AND r.stars >= 500
            UNION ALL
            SELECT hc.pr_id, hc.author, hc.committer, hc.message,
                   'Human' AS grp
            FROM human_pr_commits hc
            JOIN human_pull_request h ON h.id = hc.pr_id
            JOIN repository rh ON rh.url = h.repo_url
            WHERE h.merged_at IS NOT NULL AND rh.stars >= 500
        """).df()
    finally:
        con.close()
    return _classified(c)


def fix_pr_commit_classes(q: pd.DataFrame) -> pd.DataFrame:
    """Commits of the verified fix PRs, classified content-only:
    columns grp (opener class), pr_id, cls."""
    con = connect(read_only=True)
    try:
        con.register("fix_prs", q[["q_id", "q_is_human", "q_agent"]])
        both = con.execute("""
            SELECT c.pr_id, c.author, c.committer, c.message, f.q_agent AS grp
            FROM pr_commits c
            JOIN fix_prs f ON f.q_id = c.pr_id AND NOT f.q_is_human
            UNION ALL
            SELECT hc.pr_id, hc.author, hc.committer, hc.message, f.q_agent AS grp
            FROM human_pr_commits hc
            JOIN fix_prs f ON f.q_id = hc.pr_id AND f.q_is_human
        """).df()
    finally:
        con.close()
    return _classified(both)


def _classified(c: pd.DataFrame) -> pd.DataFrame:
    c = c.copy()
    c["cls"] = [classify_commit(a, co, m, pr_agent=None)
                for a, co, m in zip(c.author, c.committer, c.message,
                                    strict=True)]
    return c[["grp", "pr_id", "cls"]]


# --------------------------------------------------------- reviewer digests
# The DuckDB lake is gitignored, so on a fresh checkout nothing backs the
# survival estimates or the author-class table. These two CSVs are the
# lake-free inputs behind them: cox_hr / km_cuminc read the survival digest
# and per_pr_metrics reads the commit digest (side == "all" / "fix"), so every
# published cell is recomputable with pandas alone. A lake run of
# consolidation_gate.py writes them and re-checks the round-trip; both are
# committed and must be re-committed with any change to the commit-class rows.
DIGEST_SURVIVAL = DIGESTS_DIR / "rq1_survival_frame.csv"
DIGEST_COMMITS = DIGESTS_DIR / "rq3_commit_classes.csv"
_SURV_COLS = ["pr_id", "repo_id", "cohort",
              "cand_event", "cand_time", "ver_event", "ver_time"]


def write_digests(survival: pd.DataFrame, commits: pd.DataFrame) -> None:
    """Write the reviewer digests (survival frame, commit classes)."""
    DIGESTS_DIR.mkdir(parents=True, exist_ok=True)
    survival[_SURV_COLS].to_csv(DIGEST_SURVIVAL, index=False)
    commits[["side", "grp", "pr_id", "cls"]].to_csv(DIGEST_COMMITS, index=False)


# ------------------------------------------------- cross-cohort MH (rows 3/6/32)
# The Mantel-Haenszel rows are built from the shared-window frames, which only
# the lake can produce, so their strata get the same digest treatment: one row
# per repository per cohort with the merge count and the two outcome counts.
# crosscohort_frames() expands it back into per-merge rows, so repo_tables()
# below -- the real one the gate calls -- rebuilds the same strata with pandas
# alone. Committed; re-commit it with any change to the Mantel-Haenszel rows.
DIGEST_CROSSCOHORT = DIGESTS_DIR / "rq1_crosscohort_strata.csv"
_XC_COLS = ["repo_id", "cohort", "n_merges", "n_candidate", "n_verified"]


def mh(strata_tables: list[np.ndarray]) -> tuple[float, float, float, float, float]:
    """MH pooled odds ratio, its 95% CI, the CMH p-value and the pooled RR."""
    st = StratifiedTable(strata_tables)
    lo, hi = st.oddsratio_pooled_confint()
    return (st.oddsratio_pooled, lo, hi, st.test_null_odds().pvalue,
            st.riskratio_pooled)


def repo_tables(av: pd.DataFrame, hv: pd.DataFrame, col: str,
                min_n: int = 2) -> list[np.ndarray]:
    """2x2 (cohort x outcome) per shared repo with both cohorts present.

    min_n=2 is no filter beyond both-cohort presence (each side contributes
    at least one merge). The earlier min_n=6 sparse-strata heuristic was
    removed 2026-08-20 (Wannita): MH pools sparse tables by design, the
    sensitivity sweep showed the estimate stable from 218
    down to 56 strata, and the threshold only discarded signal."""
    tables = []
    for rid in set(av.repo_id) & set(hv.repo_id):
        ga, gh = av[av.repo_id == rid], hv[hv.repo_id == rid]
        if len(ga) and len(gh) and len(ga) + len(gh) >= min_n:
            tables.append(np.array(
                [[int(ga[col].sum()), int((1 - ga[col]).sum())],
                 [int(gh[col].sum()), int((1 - gh[col]).sum())]]))
    return tables


def crosscohort_digest(agent: pd.DataFrame, human: pd.DataFrame) -> pd.DataFrame:
    """Per-repo, per-cohort shared-window counts behind the MH strata."""
    def _side(df: pd.DataFrame, cohort: str) -> pd.DataFrame:
        return (df.groupby("repo_id")
                  .agg(n_merges=("pr_id", "size"),
                       n_candidate=("has_followup_30d", "sum"),
                       n_verified=("verified", "sum"))
                  .reset_index().assign(cohort=cohort))
    return pd.concat([_side(agent, "Agent"), _side(human, "Human")],
                     ignore_index=True)[_XC_COLS]


def write_crosscohort_digest(agent: pd.DataFrame, human: pd.DataFrame) -> None:
    """Write the cross-cohort strata digest (shared-window frames only)."""
    DIGESTS_DIR.mkdir(parents=True, exist_ok=True)
    crosscohort_digest(agent, human).to_csv(DIGEST_CROSSCOHORT, index=False)


def crosscohort_frames(digest: pd.DataFrame,
                       col: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Expand the digest counts back into one row per merge, outcome in `y`."""
    out = {}
    for cohort, g in digest.groupby("cohort"):
        rows = []
        for r in g.itertuples():
            hits = int(getattr(r, col))
            rows += [(r.repo_id, 1)] * hits
            rows += [(r.repo_id, 0)] * (int(r.n_merges) - hits)
        out[cohort] = pd.DataFrame(rows, columns=["repo_id", "y"])
    return out["Agent"], out["Human"]


def per_pr_metrics(c: pd.DataFrame) -> pd.DataFrame:
    """Per-group summary of per-PR agent-authorship metrics (percent, 1dp):
    n_prs, n_commits, med_commits, share_per_pr, pct_any, pct_all,
    pooled_agent, pooled_human."""
    c = c.assign(is_agent=(c.cls == "agent").astype(int),
                 is_human=(c.cls == "human").astype(int))
    per_pr = c.groupby(["grp", "pr_id"]).agg(
        n=("is_agent", "size"), sh=("is_agent", "mean"),
        any_a=("is_agent", "max"), all_a=("is_agent", "min"))
    s = per_pr.groupby("grp").agg(
        n_prs=("n", "size"), n_commits=("n", "sum"), med_commits=("n", "median"),
        share_per_pr=("sh", "mean"), pct_any=("any_a", "mean"),
        pct_all=("all_a", "mean"))
    s = (s.join(c.groupby("grp").is_agent.mean().rename("pooled_agent"))
          .join(c.groupby("grp").is_human.mean().rename("pooled_human")))
    for col in ("share_per_pr", "pct_any", "pct_all",
                "pooled_agent", "pooled_human"):
        s[col] = (100 * s[col]).round(1)
    return s.sort_values("n_prs", ascending=False)


# ------------------------------------------------------- RQ4 signal stats
RQ4_SIGNALS = ["hours_under_review", "n_commits", "lines_post_first",
               "n_review_items"]


def signal_frame(verified: set | None = None) -> pd.DataFrame:
    """Own-cut-off agent frame, uncensored 30d, joined to the four RQ4
    signals, with a 0/1 ``verified`` column. Single owner of the population
    behind both the RQ4 table and the violin figure."""
    a = rb.compute()
    a = a[a.censored_30d == 0].copy()
    if verified is None:
        verified, _ = verified_pair_sets()
    ver_p = {p for p, _ in verified}
    a["verified"] = a.pr_id.isin(ver_p).astype(int)
    sig = pps_compute()
    return a.merge(sig[["pr_id"] + RQ4_SIGNALS], on="pr_id", how="left")


def signal_clogit_frame() -> pd.DataFrame:
    """``signal_frame()`` joined to ``n_files``, the count of shipped
    non-boilerplate files (the linker's exposure unit) that controls the RQ4
    conditional logit. Single owner of that regression population."""
    df = signal_frame()
    con = connect(read_only=True)
    try:
        nf = con.execute(
            "SELECT pr_id, COUNT(*) AS n_files FROM pr_net_files "
            "WHERE NOT is_boilerplate GROUP BY pr_id").df()
    finally:
        con.close()
    df = df.merge(nf, on="pr_id", how="left")
    df["n_files"] = df.n_files.fillna(0)
    return df


def signal_clogit(df: pd.DataFrame, col: str, outcome: str):
    """Repository-conditional logit of ``outcome`` on the signal ``col``,
    controlling for the shipped non-boilerplate file count. Returns the
    complete-case frame it fitted on and the fit.

    Both regressors enter as log1p, so the first slope is the log odds ratio
    per tenfold increase in ``1 + signal``, not per 10x the raw count. Single
    owner of the RQ4 model spec: the gate table (ledger row 30) and
    fig:rq4-forest both read it, so their odds ratios cannot drift apart."""
    d = df[[col, outcome, "repo_id", "n_files"]].dropna()
    x = np.column_stack([np.log1p(d[col].clip(lower=0)),
                         np.log1p(d.n_files.clip(lower=0))])
    fit = ConditionalLogit(d[outcome].astype(float).to_numpy(), x,
                           groups=d.repo_id.to_numpy()).fit(disp=False)
    return d, fit


def signal_or_10x(slope: float) -> float:
    """Log-odds slope on log1p(signal) -> odds ratio per tenfold increase in
    ``1 + signal``. The typeset RQ4 quantity; used by the table and figure."""
    return float(np.exp(slope * np.log(10)))


def signal_distribution_stats() -> pd.DataFrame:
    """Per-signal distribution comparison of agent merges with and without a
    follow-up fix (own-window uncensored frame), for the candidate and
    verified outcomes: pooled Mann-Whitney U p and Cliff's delta (matches the
    violin figure), plus the ``signal_clogit`` odds ratio as
    ``clogit_or_10x`` -- see ``signal_clogit`` for what that ratio compares.
    Deterministic; asserted by consolidation_gate.py (ledger row 30)."""
    df = signal_clogit_frame()

    rows = []
    for outcome in ("has_followup_30d", "verified"):
        for col in RQ4_SIGNALS:
            d, fit = signal_clogit(df, col, outcome)
            x1 = d[d[outcome] == 1][col]
            x0 = d[d[outcome] == 0][col]
            u = mannwhitneyu(x1, x0, alternative="two-sided")
            delta = cliffs_delta_from_u(u.statistic, len(x1), len(x0))
            b = float(np.asarray(fit.params)[0])
            rows.append({
                "outcome": outcome, "signal": col,
                "mwu_p": float(u.pvalue), "cliffs_delta": float(delta),
                "clogit_or_10x": signal_or_10x(b),
                "clogit_p": float(np.asarray(fit.pvalues)[0]),
            })
    return pd.DataFrame(rows)
