"""Revise-and-Repair paper numbers: the consolidation gate.

Reproduces the paper rows it covers from the pinned artifacts and ASSERTS each
headline value -- including the sub-numbers those rows quote (RR, strata counts,
CI bounds, fixed-effects ORs, the joint task-type LR test, the gradient p-value
and the language split) -- so the paper can only typeset numbers this script
vouches for. The `ledger(<row>, ...)` calls below are the definitive
list of rows this file REPORTS. One row is asserted without one: row 8's
own-frame repository structure (606 repositories, 39 multi-agent) is a frame
invariant, so it is a bare `check()` in the frames block rather than a printed
ledger value. Rows 4, 12 and 18 are owned by rq1_verified_rates.py.

HYBRID CUT-OFF DESIGN (user-locked 2026-08-05): every CROSS-COHORT comparison
(agent vs human: the MH rows) runs on frames censored at the SHARED cut-off
date (the earlier of the two PR tables' last merge, currently the human
table's), so both cohorts observe fixes over the same calendar period.
SINGLE-COHORT analyses (incidence descriptives, fixer identity, the RQ4
visibility probes) run on each cohort's OWN cut-off, keeping the full frame.
The human frame is identical under both (the shared date IS its own end).
Within that own-cut-off frame, both fixer rows are further restricted to the
cohort's FULLY OBSERVED merges -- the same merges behind the RQ1 rates
(Wannita 2026-08-20) -- because a censored merge contributes a pair only when
its fix arrived inside the short observable stub, and those fast fixes skew
toward agent fixers. The unrestricted variants stay asserted as robustness.

Run:  uv run python human-annotation/consolidation_gate.py
Outputs: data/digests/consolidation_gate.csv (+ stdout report), plus the
committed digest pins it regenerates.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2, chi2_contingency, mannwhitneyu, spearmanr, wilcoxon
from statsmodels.discrete.conditional_models import ConditionalLogit

from prfix.common.loaders.base import connect
from prfix.common.metrics import (
    DIGEST_CROSSCOHORT,
    crosscohort_frames,
    gold_direct_resolved,
    mh,
    repo_tables,
    write_crosscohort_digest,
)
from prfix.rq1_fix_rates import linker as rb
from prfix.rq1_fix_rates.boilerplate import is_boilerplate
from prfix.rq2_fix_authors.fixer import (
    DIGEST_FIXER,
    add_fixer,
    per_agent_split,
    write_fixer_digest,
)
from prfix.rq4_signals.per_pr_signals import compute as per_pr_compute

HERE = Path(__file__).resolve().parent
CODE = HERE.parent
OUT = CODE / "data" / "digests"
JUDGE = CODE / "data" / "judge"

AGENT_JUDGE = JUDGE / "rq1_llm_judge_pop_anthropic_claude-opus-4-8_low.csv"
AGENT_CASES = CODE / "data" / "judge" / "rq1_judge_cases_population_12k.json"
HUMAN_JUDGE = JUDGE / "rq1_llm_judge_pop_humanP_anthropic_claude-opus-4-8_low.csv"
HUMAN_CASES = CODE / "data" / "judge" / "rq1_judge_cases_population_12k_humanP.json"
AGREED = HERE / "labels" / "humans" / "rq1_followup_validation_agreed.csv"

LEDGER: list[dict] = []  # accumulated rows for the output CSV


def ledger(row: int, claim: str, value: str) -> None:
    LEDGER.append({"ledger_row": row, "claim": claim, "value": value})
    print(f"  [{row:>2}] {claim}: {value}")


def check(cond: bool, msg: str) -> None:
    """Hard gate. Raises rather than asserting so `python -O` cannot turn the
    whole consolidation gate into a no-op that still prints PASSED.
    RRS_REPIN=1 downgrades failures to printed REPIN lines (collect mode for
    re-pinning after a deliberate frame/method change); the final banner then
    reports the run as NOT passed."""
    if not cond:
        if os.environ.get("RRS_REPIN"):
            print(f"REPIN: {msg}")
            _repin_misses.append(msg)
        else:
            raise SystemExit(f"LEDGER MISMATCH: {msg}")


_repin_misses: list[str] = []


def fe_logit(y: pd.Series, x: pd.DataFrame, groups: pd.Series):
    """Repository fixed-effects (conditional) logit. Groups with no outcome
    variation drop out of the conditional likelihood, which is what makes this
    the within-repo estimate the ledger quotes."""
    keep = y.notna() & groups.notna() & x.notna().all(axis=1)
    return ConditionalLogit(y[keep].astype(float), x[keep].astype(float),
                            groups=groups[keep]).fit(disp=False)


def fe_lr_test(res, k: int) -> tuple[float, int, float]:
    """Joint LR test of all k coefficients in a fitted conditional logit."""
    stat = 2 * (res.llf - res.model.loglike(np.zeros(k)))
    return stat, k, float(chi2.sf(stat, k))


def load_judged(judge_csv: Path, cases_json: Path) -> pd.DataFrame:
    """Judge verdicts joined to case metadata (p_id, q_id, n_substantive).
    A missing label is normalised to "ERROR" here -- the one place -- so every
    downstream label filter is NaN-safe and the ERROR tolerance below sees the
    failed calls whatever shape they arrive in.

    ``n_substantive`` is the count shipped in the case file, computed with the
    boilerplate rule of the day; ``n_sub_fixed`` recounts the same shared-file
    list through the CURRENT `is_boilerplate`. Row 34 asserts the two agree
    wherever a verdict depends on them."""
    j = pd.read_csv(judge_csv)
    j["label"] = j["label"].fillna("ERROR")
    c = pd.DataFrame(json.loads(cases_json.read_text()))
    for col in ("pair_id", "p_id", "q_id", "n_substantive"):
        c[col] = c[col].astype(int)
    c["n_sub_fixed"] = [sum(not is_boilerplate(f["name"]) for f in shared)
                        for shared in c["shared"]]
    return j.merge(c[["pair_id", "p_id", "q_id", "n_substantive", "n_sub_fixed"]],
                   on="pair_id", validate="1:1")


def cliffs(a, b) -> float:
    u = mannwhitneyu(a, b, alternative="two-sided")
    return 2 * u.statistic / (len(a) * len(b)) - 1


# ---------------------------------------------------------------- frames
print("=== frames ===")
a = rb.compute()                                # own cut-off: single-cohort frame
a = a[a.censored_30d == 0].copy()
h = rb.compute(p_side="human")                  # own cut-off: single-cohort frame
h = h[h.censored_30d == 0].copy()
a_sh = rb.compute(cutoff="shared")              # shared cut-off: cross-cohort frame
a_sh = a_sh[a_sh.censored_30d == 0].copy()
h_sh = rb.compute(p_side="human", cutoff="shared")   # cross-cohort frame
h_sh = h_sh[h_sh.censored_30d == 0].copy()
check(len(a) == 4505 and int(a.has_followup_30d.sum()) == 1031,
      "agent own frame drifted (expect 4,505 / 1,031)")
check(len(h) == 4063 and int(h.has_followup_30d.sum()) == 596,
      "human frame drifted (expect 4,063 / 596)")
_a_repos = a.repo_id.nunique()
_a_multi = int((a.groupby("repo_id").agent.nunique() > 1).sum())
check((_a_repos, _a_multi) == (606, 39),
      f"agent own frame hosts {_a_multi} of {_a_repos} multi-agent repositories, "
      f"not 39 of 606 -- results.tex and threats.tex typeset both counts as the "
      f"reason per-agent orderings are not repository-controlled")
check(len(a_sh) == 2012 and int(a_sh.has_followup_30d.sum()) == 421,
      "agent shared frame drifted (expect 2,012 / 421)")
# The human frame is expected to be identical under both cut-offs (the shared
# date IS the human table's own end). If a lake rebuild ever makes the human
# table run later this stops holding; the cross-cohort rows below use h_sh
# either way, so the comparison stays on one observation window.
check(set(h_sh.pr_id) == set(h.pr_id),
      f"human shared frame ({len(h_sh)}) != human own frame ({len(h)}): the human "
      f"table no longer ends first, so single-cohort human rows (2, 5, 11) are on "
      f"a wider window than the cross-cohort rows (3, 6) -- re-pin rows 2/5/11")
ledger(1, "agent candidate incidence (own cut-off)",
       f"{a.has_followup_30d.mean():.1%} (1,031/4,505)")
ledger(2, "human candidate incidence",
       f"{h.has_followup_30d.mean():.1%} (596/4,063)")
ledger(1, "agent candidate incidence (shared cut-off)",
       f"{a_sh.has_followup_30d.mean():.1%} (421/2,012)")

# ---------------------------------------------------------------- row 3: candidate MH
print("\n=== repair incidence, repo-stratified ===")
cand_tables = repo_tables(a_sh.assign(y=a_sh.has_followup_30d),
                          h_sh.assign(y=h_sh.has_followup_30d), "y")
or_, lo, hi, p, rr = mh(cand_tables)
check(round(or_, 2) == 1.41 and round(lo, 2) == 1.19 and round(hi, 2) == 1.68,
      f"candidate MH OR {or_:.3f} [{lo:.3f},{hi:.3f}] != 1.41 [1.19,1.68]")
check(round(p * 1e5, 1) == 4.3, f"candidate MH p {p:.2e} != 4.3e-05")
check(round(rr, 2) == 1.27, f"candidate MH RR {rr:.3f} != 1.27")
check(len(cand_tables) == 218, f"candidate MH strata {len(cand_tables)} != 218")
ledger(3, "candidate gap MH (shared cut-off)",
       f"OR {or_:.2f} [{lo:.2f},{hi:.2f}] p={p:.1e} RR {rr:.2f} "
       f"({len(cand_tables)} strata)")

# ---------------------------------------------------------------- verified flags
ja = load_judged(AGENT_JUDGE, AGENT_CASES)
jh = load_judged(HUMAN_JUDGE, HUMAN_CASES)

# Failed judge calls land as label="ERROR" and are counted as NOT-verified by the
# startswith("Direct") filters below, which deflates that side's verified count.
# Report both sides so the asymmetry is visible, and cap it so it can never grow
# large enough to matter without failing the gate.
for _side, _j in (("agent", ja), ("human", jh)):
    _n_err = int((_j.label == "ERROR").sum())
    _n_ind = int((_j.label == "Indeterminate").sum())
    ledger(14, f"{_side} judge rows not usable as evidence",
           f"{_n_err} ERROR + {_n_ind} Indeterminate of {len(_j)}")
    check(_n_err / max(len(_j), 1) < 0.005,
          f"{_side} judge ERROR rows {_n_err}/{len(_j)} exceed 0.5%: they are "
          f"counted as not-verified, so this side's verified rate is deflated "
          f"relative to the other -- re-judge with --retry-errors before quoting")

con = connect(read_only=True)
_norm = lambda u: str(u).strip().rstrip("/").lower()  # noqa: E731
agreed = pd.read_csv(AGREED)
_pr = con.execute("SELECT id, html_url, agent FROM pull_request").df()
_url2id = {_norm(u): int(i) for i, u in zip(_pr.id, _pr.html_url, strict=True)}
agreed["p_id"] = agreed.p_url.map(lambda u: _url2id[_norm(u)])
gold_direct = set(agreed[(agreed.label_1 == agreed.label_2)
                         & (agreed.label_1 == "Direct fix")].p_id)

ver_a = set(ja[(ja.n_substantive > 0)
               & ja.label.str.startswith("Direct")].p_id) | gold_direct
ver_h = set(jh[(jh.n_substantive > 0)
               & jh.label.str.startswith("Direct")].p_id)
a["verified"] = a.pr_id.isin(ver_a).astype(int)
h["verified"] = h.pr_id.isin(ver_h).astype(int)
a_sh["verified"] = a_sh.pr_id.isin(ver_a).astype(int)
h_sh["verified"] = h_sh.pr_id.isin(ver_h).astype(int)

check(int(a.verified.sum()) == 203, f"agent verified {int(a.verified.sum())} != 203")
check(int(h.verified.sum()) == 95, f"human verified {int(h.verified.sum())} != 95")
check(int(a_sh.verified.sum()) == 74,
      f"agent shared-frame verified {int(a_sh.verified.sum())} != 74")
ledger(4, "agent verified incidence (own cut-off)",
       f"{a.verified.mean():.2%} (203/4,505)")
ledger(5, "human verified incidence", f"{h.verified.mean():.2%} (95/4,063)")
ledger(4, "agent verified incidence (shared cut-off)",
       f"{a_sh.verified.mean():.2%} (74/2,012)")

# ---------------------------------------------------------------- row 6: verified MH
ver_tables = repo_tables(a_sh, h_sh, "verified")
or_, lo, hi, p, rr = mh(ver_tables)
check(round(or_, 2) == 1.62 and round(p, 3) == 0.015,
      f"verified MH OR {or_:.3f} p {p:.3f} != 1.62 / 0.015")
check(round(lo, 2) == 1.10 and round(hi, 2) == 2.39,
      f"verified MH CI [{lo:.3f},{hi:.3f}] != [1.10,2.39]")
check(len(ver_tables) == 218, f"verified MH strata {len(ver_tables)} != 218")
ledger(6, "verified gap MH (shared cut-off, SIGNIFICANT)",
       f"OR {or_:.2f} [{lo:.2f},{hi:.2f}] p={p:.3f} ({len(ver_tables)} strata)")

# Lake-free provenance for rows 3/6/32: the
# shared-window per-repo counts both MH strata sets are built from. A digest
# that does not rebuild both odds ratios through the same repo_tables()/mh()
# is not evidence, so the gate refuses it -- and once committed,
# tests/test_pinned_ledger_rows.py re-derives the typeset ORs without a lake.
write_crosscohort_digest(a_sh, h_sh)
_xc = pd.read_csv(DIGEST_CROSSCOHORT)
for _col, _want in (("n_candidate", mh(cand_tables)), ("n_verified", mh(ver_tables))):
    _xa, _xh = crosscohort_frames(_xc, _col)
    _xt = repo_tables(_xa, _xh, "y")
    check(len(_xt) == len(cand_tables) and np.allclose(mh(_xt), _want),
          f"cross-cohort strata digest does not reproduce the {_col} MH row "
          f"({len(_xt)} strata, {mh(_xt)} vs {_want})")
ledger(28, "cross-cohort strata digest reproduces rows 3/6/32 without the lake",
       f"{len(_xc)} repo x cohort rows -> {len(cand_tables)} strata, both MH "
       "odds ratios identical (data/digests/rq1_crosscohort_strata.csv)")

# ---------------------------------------------------------------- rows 9-11: fixer identity
# The fixer frame is every verified direct-fix pair (judge Direct + the 14
# human-consensus gold pairs) RESTRICTED to the agent cohort's fully
# observed frame (Wannita 2026-08-20, second decision of the day): a
# censored merge contributes a pair only when its fix arrived in the short
# observable stub, and those fast fixes skew toward agent fixers, so the
# unrestricted frame inflates the agent share on both sides -- the two
# unrestricted values asserted below (0.767 agent self-fix against 0.696;
# 0.763 human-fixed against 0.891) are that inflation, measured. The human
# side (rows below) was already restricted; this makes the two sides
# symmetric. Unrestricted values stay asserted as the robustness variant.
print("\n=== fixer identity ===")
qag = _pr.set_index("id").agent
# Q resolution (and with it the cohort each gold fix is attributed to) is
# owned by metrics.gold_direct_resolved, the same helper behind row 29's
# verified fix PRs, so the two rows cannot disagree about a gold pair.
_hpr_urls = con.execute("SELECT id, html_url FROM human_pull_request").df()
_gold = gold_direct_resolved(_pr, _hpr_urls)
_gold["p_id"] = _gold.p_url.map(lambda u: _url2id[_norm(u)])
check(len(_gold) == 14 and _gold.p_id.nunique() == 14,
      f"gold consensus Direct pairs {len(_gold)} != 14")
_jd = ja[(ja.label.str.startswith("Direct")) & (ja.n_substantive > 0)]
da_all = add_fixer(pd.concat(
    [_jd[["p_id", "q_id", "q_is_human"]],
     _gold[["p_id", "q_id", "q_is_human"]]], ignore_index=True), qag)
da = da_all[da_all.p_id.isin(set(a.pr_id))]

pair_share = da.fixer.value_counts(normalize=True)
check(round(pair_share["self"], 3) == 0.696 and len(da) == 263,
      f"pair-level self {pair_share['self']:.3f} (n={len(da)}) != 0.696 (n=263)")
ledger(9, "fixer of agent-merge fixes (pairs, n=263, judge + gold, uncensored frame)",
       f"self {pair_share['self']:.1%} / human {pair_share['human']:.1%} / "
       f"other-agent {pair_share['other-agent']:.1%}")
_all_share = da_all.fixer.value_counts(normalize=True)
check(round(_all_share["self"], 3) == 0.767 and len(da_all) == 347,
      f"unrestricted self {_all_share['self']:.3f} (n={len(da_all)}) != 0.767 (n=347)")
ledger(9, "fixer, unrestricted robustness variant (n=347)",
       f"self {_all_share['self']:.1%} -- typeset in RQ2's mirror paragraph "
       f"beside the human-side 76.3% to show the stub bias direction")

# Lake-free provenance for rows 9-10: frame
# membership is the only lake-bound part of the restriction, so the digest is
# one row per unrestricted pair with its fixer class and an `in_frame` flag.
# A digest that does not rebuild both frames through the same
# per_agent_split() is not evidence, so the gate refuses it -- and once
# committed, tests/test_rq2_per_agent_fixer.py re-derives the typeset
# restricted split without a lake.
write_fixer_digest(da_all, set(a.pr_id))
_fx = pd.read_csv(DIGEST_FIXER)
check(len(_fx) == len(da_all)
      and per_agent_split(_fx[_fx.in_frame == 1]).equals(per_agent_split(da))
      and per_agent_split(_fx).equals(per_agent_split(da_all)),
      f"fixer frame digest does not reproduce rows 9-10 ({len(_fx)} rows, "
      f"{int(_fx.in_frame.sum())} in frame)")
ledger(28, "fixer frame digest reproduces rows 9-10 without the lake",
       f"{len(_fx)} pairs, {int(_fx.in_frame.sum())} in the fully observed "
       "frame, per-agent split identical "
       "(data/digests/rq2_fixer_frame.csv)")

pr_fix = da.groupby("p_id").fixer.agg(
    lambda x: "human-involved" if "human" in set(x)
    else ("self-only" if set(x) == {"self"} else "agent-mixed"))
pr_share = pr_fix.value_counts(normalize=True)
# On the restricted frame the PR-level population IS the 203 verified merges
# of row 4 -- one population, two views (typeset in RQ2's results).
check(round(pr_share["self-only"], 3) == 0.626 and len(pr_fix) == 203,
      f"PR-level self-only {pr_share['self-only']:.3f} (n={len(pr_fix)}) != 0.626 (n=203)")
check(round(pr_share["human-involved"], 3) == 0.340
      and round(pr_share.get("agent-mixed", 0), 3) == 0.034,
      f"PR-level human-involved/agent-mixed "
      f"{pr_share['human-involved']:.3f}/{pr_share.get('agent-mixed', 0):.3f} "
      f"!= 0.340/0.034")
ledger(9, "fixer of agent-merge fixes (PRs = the 203 verified merges of row 4)",
       f"self-only {pr_share['self-only']:.1%} / human-involved "
       f"{pr_share['human-involved']:.1%} / agent-mixed {pr_share.get('agent-mixed', 0):.1%}")

# per-agent fixer split (verified direct-fix pairs, judge + gold, on the
# uncensored frame). Claude Code's two frame pairs are the gold pairs, fixed
# by a human and by another agent; its judged self-fix sits on a censored
# merge and is out of frame.
per_agent = per_agent_split(da)
_expect = {  # agent: (n, self, human, other-agent)
    "OpenAI_Codex": (125, 74, 48, 3),
    "Devin":        (88,  64, 21, 3),
    "Copilot":      (40,  38,  2, 0),
    "Cursor":       (8,   7,   0, 1),
    "Claude_Code":  (2,   0,   1, 1),
}
for ag, (n_, s_, h_, o_) in _expect.items():
    if ag not in per_agent.index:
        check(False, f"per-agent fixer split: {ag} has no judged direct-fix "
                     f"pairs, expected n={n_}")
        continue
    row = per_agent.loc[ag]
    n, s, hu, o = (int(row["n"]), int(row["self"]),
                   int(row["human"]), int(row["other-agent"]))
    check((n, s, hu, o) == (n_, s_, h_, o_),
          f"per-agent fixer split drifted for {ag}: "
          f"{n}/{s}/{hu}/{o} != {n_}/{s_}/{h_}/{o_}")
    ledger(9, f"fixer split {ag} (n={n})",
           f"self {100 * s / n:.1f}% / human {100 * hu / n:.1f}% / "
           f"other-agent {100 * o / n:.1f}%")

noncx = da[da.p_agent != "OpenAI_Codex"]
check(round(noncx.fixer.value_counts(normalize=True)["self"], 3) == 0.79
      and len(noncx) == 138,
      f"excl-Codex self share {noncx.fixer.value_counts(normalize=True)['self']:.3f} "
      f"(n={len(noncx)}) != 0.790 (n=138)")
ledger(10, "excl-Codex (pairs, n=138, uncensored frame)",
       f"self {noncx.fixer.value_counts(normalize=True)['self']:.1%}")

dh_all = jh[(jh.label.str.startswith("Direct")) & (jh.n_substantive > 0)].copy()
dh = dh_all[dh_all.p_id.isin(set(h.pr_id))].copy()
hshare, hshare_all = dh.q_is_human.mean(), dh_all.q_is_human.mean()
check(len(dh) == 110 and round(hshare, 3) == 0.891,
      f"human-merge fixer human-share {hshare:.3f} (n={len(dh)}) != 0.891 (n=110)")
check(len(dh_all) == 156 and round(hshare_all, 3) == 0.763,
      f"unrestricted human-merge fixer human-share {hshare_all:.3f} "
      f"(n={len(dh_all)}) != 0.763 (n=156)")
ledger(11, "fixer of human-merge fixes (pairs, n=110)",
       f"human {hshare:.1%} / agent {1 - hshare:.1%}")
# INCLUSION RULE: this row and rows 9-10 are both restricted to their cohort's
# uncensored frame (2026-08-20), so the two sides are quoted under the SAME
# rule. The restriction is not cosmetic -- dropping it moves this row from
# 0.891 to 0.763 human-fixed and rows 9-10 from 0.696 to 0.767 self-fix, both
# toward agent fixers, because a censored P can only be repaired inside the
# still-observable stub. Both unrestricted values stay asserted as the
# robustness variant of their row.
ledger(11, "fixer of human-merge fixes, unrestricted judged population",
       f"human {hshare_all:.1%} / agent {1 - hshare_all:.1%} (n={len(dh_all)}) "
       f"-- inclusion-rule sensitivity; rows 9-10 are restricted to the "
       f"uncensored frame (2026-08-20), so both sides now use the SAME "
       f"rule and this row is the human-side robustness variant")

# ---------------------------------------------------------------- RQ4 signals
print("\n=== visibility (RQ4) ===")
sig = per_pr_compute()
df = a.merge(sig, on="pr_id", suffixes=("", "_s"))

# row 15: docs (MH over repos, docs vs rest)
# The len(g) >= 6 filter below is the same sparse-strata heuristic that was
# removed from repo_tables() on 2026-08-20 (see its docstring: MH pools sparse
# tables by design). It survives here only because this row is retired from the
# paper -- nothing typeset depends on it, and re-pinning 0.50/101 strata without
# the filter needs a lake run. Do not read its strata count as comparable to the
# 218 strata of rows 3/6.
df["docs"] = (df.task_type == "docs").astype(int)
tables = []
for _rid, g in df.groupby("repo_id"):
    if g.docs.nunique() == 2 and len(g) >= 6:
        tables.append(np.array(
            [[int(g[g.docs == 1].has_followup_30d.sum()),
              int((1 - g[g.docs == 1].has_followup_30d).sum())],
             [int(g[g.docs == 0].has_followup_30d.sum()),
              int((1 - g[g.docs == 0].has_followup_30d).sum())]]))
or_, lo, hi, p, _ = mh(tables)
check(round(or_, 2) == 0.50 and p < 1e-3,
      f"docs MH OR {or_:.3f} p {p:.2e} != 0.50 / <1e-3")
check(len(tables) == 101, f"docs MH strata {len(tables)} != 101")

# same claim under a repo fixed-effects logit, plus the joint task-type LR test
fe_docs = fe_logit(df.has_followup_30d, df[["docs"]], df.repo_id)
or_fe, p_fe = float(np.exp(fe_docs.params.iloc[0])), float(fe_docs.pvalues.iloc[0])
check(round(or_fe, 2) == 0.56 and p_fe < 1e-3,
      f"docs FE logit OR {or_fe:.3f} p {p_fe:.2e} != 0.56 / <.001")
tt = pd.get_dummies(df.task_type, prefix="tt", drop_first=True)
fe_tt = fe_logit(df.has_followup_30d, tt, df.repo_id)
lr, lr_df, p_lr = fe_lr_test(fe_tt, tt.shape[1])
check(round(lr, 1) == 25.1 and lr_df == 11 and p_lr < 0.01,
      f"joint task-type LR chi2={lr:.2f} df={lr_df} p={p_lr:.2e} != 25.1 / 11 / <.01")
ledger(15, "docs merges (lone visible marker)",
       f"MH OR {or_:.2f} [{lo:.2f},{hi:.2f}] p={p:.1e} ({len(tables)} strata); "
       f"FE logit OR {or_fe:.2f} p={p_fe:.1e}; joint task-type LR chi2={lr:.1f} "
       f"df={lr_df} p={p_lr:.1e}; "
       f"pooled docs rate {df[df.docs == 1].has_followup_30d.mean():.1%} (Simpson-reversed)")

# row 16: size null within-repo
ds = []
for _rid, g in df.groupby("repo_id"):
    if g.has_followup_30d.nunique() == 2 and len(g) >= 20:
        ds.append(cliffs(g[g.has_followup_30d == 1].lines_first.dropna(),
                         g[g.has_followup_30d == 0].lines_first.dropna()))
w, p_size = wilcoxon(ds)
n_pos = int(sum(d > 0 for d in ds))
check(round(p_size, 2) == 0.38, f"size within-repo Wilcoxon p {p_size:.3f} != 0.38")
check((n_pos, len(ds)) == (26, 46),
      f"size within-repo positive repos {n_pos}/{len(ds)} != 26/46")

# same null under a repo fixed-effects logit; OR is per natural-log line, so the
# "10x size" figure the ledger quotes is OR**ln(10)
df["ln_lines"] = np.log1p(df.lines_first)
fe_size = fe_logit(df.has_followup_30d, df[["ln_lines"]], df.repo_id)
or_sz, p_sz = float(np.exp(fe_size.params.iloc[0])), float(fe_size.pvalues.iloc[0])
or_10x = or_sz ** np.log(10)
check(round(or_sz, 3) == 1.053 and round(p_sz, 3) == 0.045,
      f"size FE logit OR {or_sz:.4f} p {p_sz:.4f} != 1.053 / 0.045")
check(round(or_10x, 2) == 1.13, f"size FE 10x odds factor {or_10x:.3f} != 1.13")
ledger(16, "size within-repo", f"AT MOST MARGINAL: per-repo Wilcoxon p={p_size:.2f} "
       f"(median d {np.median(ds):+.3f}, {n_pos}/{len(ds)} repos +) vs conditional-logit "
       f"OR {or_sz:.3f} p={p_sz:.3f}; magnitude negligible either way "
       f"(10x size -> odds x{or_10x:.2f}); pooled d=+0.15 is exposure")

# row 17: Codex review inversion = composition
cx = df[df.agent == "OpenAI_Codex"].copy()
cx["zero_rev"] = (cx.n_review_items == 0).astype(int)
zr = cx.groupby("zero_rev").has_followup_30d.mean()
chi2_, p_zr = chi2_contingency(pd.crosstab(cx.zero_rev, cx.has_followup_30d))[:2]
wr, nr = [], []
for _rid, g in cx.groupby("repo_id"):
    if g.has_followup_30d.nunique() == 2 and len(g) >= 20:
        wr.append(cliffs(g[g.has_followup_30d == 1].hours_under_review.dropna(),
                         g[g.has_followup_30d == 0].hours_under_review.dropna()))
        nr.append(cliffs(g[g.has_followup_30d == 1].n_review_items.dropna(),
                         g[g.has_followup_30d == 0].n_review_items.dropna()))
check(round(100 * zr[1], 1) == 32.2 and round(100 * zr[0], 1) == 18.4,
      f"Codex zero-review split {100 * zr[1]:.1f}/{100 * zr[0]:.1f} != 32.2/18.4")
check(np.median(wr) > -0.05, "Codex within-repo hours d unexpectedly negative")
check(round(np.median(wr), 2) == 0.07 and round(np.median(nr), 2) == 0.00,
      f"Codex within-repo median d hours {np.median(wr):+.3f} / items "
      f"{np.median(nr):+.3f} != +0.07 / 0.00")
check(len(wr) == 20, f"Codex within-repo strata {len(wr)} != 20")
ledger(17, "Codex review inversion", f"pooled zero-review {zr[1]:.1%} vs reviewed "
       f"{zr[0]:.1%} (p={p_zr:.1e}) BUT within-repo median d hours "
       f"{np.median(wr):+.2f} / items {np.median(nr):+.2f} ({len(wr)} repos) "
       f"-> repo composition, exposure framing only")

# row 21: repo exposure gradient
repo = df.groupby("repo_id").agg(n=("pr_id", "size"), rate=("has_followup_30d", "mean"))
big = repo[repo.n >= 10]
rho, p_rho = spearmanr(big.n, big.rate)
check(round(rho, 2) == 0.27, f"repo gradient rho {rho:.3f} != 0.27")
check(round(p_rho, 3) == 0.006 or round(p_rho, 4) == 0.0055,
      f"repo gradient p {p_rho:.4f} != ~0.006")
lang = df.groupby("language").has_followup_30d.mean()
check(round(100 * lang["Java"], 1) == 50.2 and round(100 * lang["Ruby"], 1) == 1.9,
      f"pooled language split Java {100 * lang['Java']:.1f}% / "
      f"Ruby {100 * lang['Ruby']:.1f}% != 50.2 / 1.9")
ledger(21, "repo exposure gradient",
       f"rho={rho:.2f} p={p_rho:.3f} ({len(big)} repos >=10 PRs); pooled Java "
       f"{lang['Java']:.1%} vs Ruby {lang['Ruby']:.1%}")

# ============================================================ digest additions
# Repair survival (KM / stratified Cox) and the combined
# per-PR author-class table (version 2). Logic lives in prfix.common.metrics so the
# figure scripts and this gate share one implementation.
from prfix.common import metrics  # noqa: E402

print("\n=== v3: repair survival (shared cut-off) ===")
va_pairs, vh_pairs = metrics.verified_pair_sets()
check(len(va_pairs) == 347 and len(vh_pairs) == 156,
      f"verified pair sets {len(va_pairs)}/{len(vh_pairs)} != 347/156")
sv_a = metrics.survival_frame("agent", va_pairs)
sv_h = metrics.survival_frame("human", vh_pairs)
check(len(sv_a) == 4396 and len(sv_h) == 5044,
      f"survival frames {len(sv_a)}/{len(sv_h)} != 4396/5044")
# Verified pairs come from the judge case pool; survival events can only fire
# for pairs the net_sub linker also carries. Record the drop per cohort so a
# silent, asymmetric loss of verified events cannot pass unnoticed.
drop_a = sv_a.attrs["verified_pairs_unlinked"]
drop_h = sv_h.attrs["verified_pairs_unlinked"]
check(sv_a.attrs["verified_pairs_linked"] > 0
      and sv_h.attrs["verified_pairs_linked"] > 0,
      f"a cohort lost ALL verified pairs to the net_sub linker: agent "
      f"{drop_a}/{len(va_pairs)}, human {drop_h}/{len(vh_pairs)} dropped")
ledger(25, "verified pairs not carried by the net_sub linker (survival frame)",
       f"agent {drop_a}/{len(va_pairs)}, human {drop_h}/{len(vh_pairs)} "
       "-- judge/gold pairs whose co-edit is boilerplate-only or whose PR is "
       "missing from pr_net_files; these contribute no verified event")
sv_both = pd.concat([sv_a, sv_h], ignore_index=True)

hr_c, lo_c, hi_c, p_c = metrics.cox_hr(sv_both, "cand")
check(round(hr_c, 2) == 1.39 and round(lo_c, 2) == 1.25 and round(hi_c, 2) == 1.55,
      f"cand Cox HR {hr_c:.3f} [{lo_c:.3f},{hi_c:.3f}] != 1.39 [1.25,1.55]")
check(p_c < 1e-8, f"cand Cox p {p_c:.2e} not < 1e-8")
hr_v, lo_v, hi_v, p_v = metrics.cox_hr(sv_both, "ver")
check(round(hr_v, 2) == 1.29 and round(lo_v, 2) == 1.00 and round(hi_v, 2) == 1.67,
      f"ver Cox HR {hr_v:.3f} [{lo_v:.3f},{hi_v:.3f}] != 1.29 [1.00,1.67]")
check(0.049 < p_v < 0.052, f"ver Cox p {p_v:.4f} not in (0.049, 0.052)")

cum = {}
for kind in ("cand", "ver"):
    for sub, side in ((sv_a, "agent"), (sv_h, "human")):
        cum[(kind, side)] = 100 * metrics.km_cuminc(
            sub[f"{kind}_time"].to_numpy(), sub[f"{kind}_event"].to_numpy())[1][-1]
check(round(cum[("cand", "agent")], 1) == 23.3
      and round(cum[("cand", "human")], 1) == 15.5,
      f"cand 30d cuminc {cum[('cand', 'agent')]:.2f}/{cum[('cand', 'human')]:.2f}"
      " != 23.3/15.5")
check(round(cum[("ver", "agent")], 1) == 4.5
      and round(cum[("ver", "human")], 1) == 2.6,
      f"ver 30d cuminc {cum[('ver', 'agent')]:.2f}/{cum[('ver', 'human')]:.2f}"
      " != 4.5/2.6")
ledger(25, "repair survival, repo-stratified Cox (shared cut-off)",
       f"candidate HR {hr_c:.2f} [{lo_c:.2f},{hi_c:.2f}] p={p_c:.1e} "
       f"(30d cuminc {cum[('cand', 'agent')]:.1f}% vs {cum[('cand', 'human')]:.1f}%); "
       f"verified HR {hr_v:.2f} [{lo_v:.2f},{hi_v:.2f}] p={p_v:.3f} "
       f"(30d cuminc {cum[('ver', 'agent')]:.1f}% vs {cum[('ver', 'human')]:.1f}%); "
       f"frames {len(sv_a)}/{len(sv_h)}")

print("\n=== v3: combined author-class table (per-PR, content-only) ===")
all_c = metrics.cohort_commit_classes()
all_m = metrics.per_pr_metrics(all_c)
_exp_all = {  # grp: (n_prs, med, share/PR, any, all)
    "Human":        (4900, 2.0, 0.9, 1.7, 0.6),
    "OpenAI_Codex": (2834, 1.0, None, None, None),   # unmeasured (no marker)
    "Devin":        (1813, 3.0, 80.1, 99.5, 56.9),
    "Copilot":      (1428, 4.0, 81.6, 98.7, 51.8),
    "Cursor":       (569, 2.0, 72.9, 95.6, 54.1),
    "Claude_Code":  (130, 2.5, 57.1, 79.2, 40.0),
}
for grp, (n, med, sh, any_, all_) in _exp_all.items():
    r = all_m.loc[grp]
    check(int(r.n_prs) == n and float(r.med_commits) == med,
          f"all-merged {grp} n/med {int(r.n_prs)}/{r.med_commits} != {n}/{med}")
    if sh is not None:
        check((r.share_per_pr, r.pct_any, r.pct_all) == (sh, any_, all_),
              f"all-merged {grp} per-PR metrics "
              f"{r.share_per_pr}/{r.pct_any}/{r.pct_all} != {sh}/{any_}/{all_}")
check(all_m.loc["Human", "pooled_agent"] == 0.8
      and all_m.loc["Human", "pooled_human"] == 94.9
      and int(all_m.loc["Human", "n_commits"]) == 19009,
      "human all-merged pooled row drifted")
_shares = [all_m.loc[g, "share_per_pr"]
           for g in ("Claude_Code", "Cursor", "Devin", "Copilot")]
check(min(_shares) == 57.1 and max(_shares) == 81.6,
      f"per-PR revision range {min(_shares)}-{max(_shares)} != 57.1-81.6")
ledger(26, "combined table v2: all-merged side (per-PR, content-only)",
       "share/PR % Claude 57.1 Cursor 72.9 Devin 80.1 Copilot 81.6 "
       "(range 57-82, floors, Codex unmeasured); Human 0.9 share / 1.7 any / "
       "0.6 all (4,900 PRs, 144 commit-fetch gaps); paper range replaces "
       "pooled 35-73")

qfix = metrics.verified_fix_prs()
check(len(qfix) == 350 and int(qfix.q_is_human.sum()) == 124,
      f"verified fix PRs {len(qfix)}/{int(qfix.q_is_human.sum())} != 350/124")
fix_c = metrics.fix_pr_commit_classes(qfix)
fix_m = metrics.per_pr_metrics(fix_c)
_exp_fix = {  # grp: (n_prs, n_commits, med, share/PR, any, all)
    "Human":        (124, 554, 2.0, 0.8, 0.8, 0.8),
    "OpenAI_Codex": (103, 185, 1.0, None, None, None),
    "Devin":        (59, 151, 1.0, 92.0, 98.3, 84.7),
    "Copilot":      (40, 177, 3.0, 86.6, 100.0, 70.0),
    "Cursor":       (23, 72, 1.0, 79.1, 91.3, 69.6),
    "Claude_Code":  (1, 8, 8.0, None, None, None),   # n=1: counts only
}
for grp, (n, nc, med, sh, any_, all_) in _exp_fix.items():
    r = fix_m.loc[grp]
    check(int(r.n_prs) == n and int(r.n_commits) == nc
          and float(r.med_commits) == med,
          f"fix-side {grp} n/commits/med {int(r.n_prs)}/{int(r.n_commits)}/"
          f"{r.med_commits} != {n}/{nc}/{med}")
    if sh is not None:
        check((r.share_per_pr, r.pct_any, r.pct_all) == (sh, any_, all_),
              f"fix-side {grp} per-PR metrics "
              f"{r.share_per_pr}/{r.pct_any}/{r.pct_all} != {sh}/{any_}/{all_}")
check(fix_m.loc["Human", "pooled_agent"] == 0.2
      and fix_m.loc["Human", "pooled_human"] == 99.3,
      "fix-side human pooled proxy drifted")
ledger(27, "combined table v2: fix-PR side + opener proxy",
       "human-opened fixes: 0.8% per-PR / 0.2% pooled agent share (124 PRs, "
       "554 commits) -> opener identity is a faithful fixer proxy; "
       "agent-opened share/PR 79.1-92.0 (floors); Claude n=1 counts only")

# row 29: overall (all measurable agents, excl. Codex) rows for the combined
# table; the all-merged share is the union reference (app:authorclass, quoted
# once in RQ3.1's prose -- the main-text RQ3.1 numbers are row 35's), the
# fix-side share is the abstract's third finding and RQ3.2's headline.
_MEASURABLE = ["Copilot", "Devin", "Cursor", "Claude_Code"]
ov_all = metrics.per_pr_metrics(
    all_c[all_c.grp.isin(_MEASURABLE)].assign(grp="All")).loc["All"]
ov_fix = metrics.per_pr_metrics(
    fix_c[fix_c.grp.isin(_MEASURABLE)].assign(grp="All")).loc["All"]
check(int(ov_all.n_prs) == 3940 and float(ov_all.med_commits) == 3.0
      and (ov_all.share_per_pr, ov_all.pct_all) == (78.8, 54.1),
      f"overall all-merged row drifted: {ov_all.to_dict()}")
check(int(ov_fix.n_prs) == 123 and float(ov_fix.med_commits) == 2.0
      and (ov_fix.share_per_pr, ov_fix.pct_all) == (87.4, 76.4),
      f"overall fix-side row drifted: {ov_fix.to_dict()}")
ledger(29, "combined table: overall rows (measurable agents, excl. Codex)",
       "all-merged: 3,940 PRs, med 3, share/PR 78.8%, all-agent 54.1%; "
       "fix side: 123 PRs, med 2, share/PR 87.4%, all-agent 76.4%; "
       "union reference = 78.8% (RQ3.1 typesets row 35's 78.6%), abstract "
       "finding (3) = 87.4% (fix side); "
       "both pool the four signing agents only, Codex excluded")

# row 30: RQ4 per-signal distribution stats (pooled MWU/delta matching the
# violin figure + file-count-controlled conditional logit).
sig_stats = metrics.signal_distribution_stats()
sig_stats.to_csv(OUT / "rq4_signal_stats.csv", index=False)
_ss = sig_stats.set_index(["outcome", "signal"])
def _pstr(p):  # the paper's p-value format: <0.001, else 2 significant digits
    return "<0.001" if float(p) < 1e-3 else f"{float(p):.2g}"


# Every typeset RQ4 number is pinned here: the deltas and ORs of the table and
# the RQ4 prose, plus the p-values printed alongside them. The candidate-outcome
# clogit p-values are not typeset, so they carry no pin (None).
_exp_delta = {  # (outcome, signal): (delta 3dp, or10x 2dp, MWU p, clogit p)
    ("has_followup_30d", "hours_under_review"): (-0.100, 1.14, "<0.001", None),
    ("has_followup_30d", "n_commits"):          (0.028, 1.71, "0.15", None),
    ("has_followup_30d", "lines_post_first"):   (0.024, 1.05, "0.21", None),
    ("has_followup_30d", "n_review_items"):     (-0.101, 0.99, "<0.001", None),
    ("verified", "hours_under_review"):         (-0.221, 1.23, "<0.001", "0.18"),
    ("verified", "n_commits"):                  (0.013, 6.06, "0.75", "<0.001"),
    ("verified", "lines_post_first"):           (0.032, 1.34, "0.41", "<0.001"),
    ("verified", "n_review_items"):             (-0.239, 2.41, "<0.001", "0.022"),
}
for key, (dlt, or10, mwu_p, clog_p) in _exp_delta.items():
    r = _ss.loc[key]
    check(round(float(r.cliffs_delta), 3) == dlt and
          round(float(r.clogit_or_10x), 2) == or10,
          f"rq4 signal stats drifted at {key}: "
          f"delta {r.cliffs_delta:.3f} or10x {r.clogit_or_10x:.2f}")
    check(_pstr(r.mwu_p) == mwu_p,
          f"rq4 MWU p drifted at {key}: {_pstr(r.mwu_p)} != {mwu_p}")
    check(clog_p is None or _pstr(r.clogit_p) == clog_p,
          f"rq4 clogit p drifted at {key}: {_pstr(r.clogit_p)} != {clog_p}")
ledger(30, "RQ4 per-signal distribution stats",
       "pooled (violin view): all deltas negligible-to-small, review signals "
       "NEGATIVE (merges with a follow-up fix sit in fast-review repos); "
       "within-repo + shipped-files control: commits/PR OR 6.06 per 10x "
       "(p<0.001), review items OR 2.41 (p=0.022), churn OR 1.34, hours n.s. "
       "-> revision effort marks fix-prone merges beyond file surface area")

# combined-table CSV for typesetting
comb = all_m.add_prefix("all_").join(fix_m.add_prefix("fix_"), how="outer")
comb.to_csv(OUT / "combined_authorclass_v2.csv")

# Reviewer digests: the lake is gitignored, so rows 25-27 would rest on data
# only the author can load. Write the lake-free inputs and re-derive every
# published estimate from them before they are committed.
metrics.write_digests(sv_both, pd.concat([all_c.assign(side="all"),
                                     fix_c.assign(side="fix")],
                                    ignore_index=True))
_d_sv = pd.read_csv(metrics.DIGEST_SURVIVAL)
_d_c = pd.read_csv(metrics.DIGEST_COMMITS)
check(round(metrics.cox_hr(_d_sv, "cand")[0], 4) == round(hr_c, 4)
      and round(metrics.cox_hr(_d_sv, "ver")[0], 4) == round(hr_v, 4)
      and metrics.per_pr_metrics(_d_c[_d_c.side == "all"]).equals(all_m)
      and metrics.per_pr_metrics(_d_c[_d_c.side == "fix"]).equals(fix_m),
      "the reviewer digests do not reproduce rows 25-27 -- do not commit them")
ledger(28, "reviewer digests reproduce rows 25-27 without the lake",
       f"{metrics.DIGEST_SURVIVAL.name} ({len(_d_sv)} PRs) + {metrics.DIGEST_COMMITS.name} "
       f"({len(_d_c)} commits): Cox HRs and both per-PR tables re-derived exactly")

# ---------------------------------------------------------------- row 31
# Judge Direct-fix precision on HUMAN-merge pairs: the blind spot-check
# delivered 2026-08-17 (annotator nhat, hosted-labeler export). Lake-free:
# both inputs are committed under labels/judge/. The paper typesets 27/30 =
# 90% at three sites (RQ1 approach, RQ1.2 results, threats) as the transfer
# check for the agent-side precision constant in rq1_verified_rates.py.
_hp = pd.read_csv(HERE / "labels" / "judge"
                  / "rq1_judge_sample50_enriched_humanP_annotator_nhat3.csv")[
    ["pair_id", "label"]].merge(
    pd.read_csv(HERE / "labels" / "judge"
                / "rq1_judge_sample50_enriched_humanP_model_key.csv")[
        ["pair_id", "label"]].rename(columns={"label": "llm"}), on="pair_id")
_hp_d = _hp[_hp.llm == "Direct fix"]
_hp_ok = int((_hp_d.label == "Direct fix").sum())
check(len(_hp) == 50 and len(_hp_d) == 30 and _hp_ok == 27,
      f"humanP spot-check precision {_hp_ok}/{len(_hp_d)} on {len(_hp)} "
      f"pairs != 27/30 on 50")
# The paper scales adjusted rates by the POOLED precision (54/60, both
# spot-checks; user-locked 2026-08-17) and typesets its Wilson bounds 80-95%.
# Recompute the pooled interval here so the typeset bounds cannot drift from
# the committed label files (formula matches rq1_verified_rates.wilson).
_pk, _pn = _hp_ok + 27, len(_hp_d) + 30
_pp, _pz = _pk / _pn, 1.96
_pd = 1 + _pz ** 2 / _pn
_pc = (_pp + _pz ** 2 / (2 * _pn)) / _pd
_ph = _pz * np.sqrt(_pp * (1 - _pp) / _pn + _pz ** 2 / (4 * _pn ** 2)) / _pd
check((_pk, _pn) == (54, 60)
      and (round(100 * (_pc - _ph)), round(100 * (_pc + _ph))) == (80, 95),
      f"pooled precision {_pk}/{_pn}, Wilson "
      f"[{100 * (_pc - _ph):.1f}, {100 * (_pc + _ph):.1f}] != 54/60 [80, 95]")
ledger(31, "judge Direct-fix precision on human-merge pairs (blind spot-check)",
       f"{_hp_ok}/{len(_hp_d)} = {100 * _hp_ok / len(_hp_d):.1f}%, identical "
       f"to the agent-side 27/30; pooled {_pk}/{_pn} = {100 * _pp:.1f}% "
       f"(Wilson 80-95%) is the scaling estimate; the paper's transfer "
       f"caveat is retired")

# ---------------------------------------------------------------- row 32
# tab:rq1-crosscohort: both cohorts on the SHARED window, with the adjusted
# column scaling each side's judge-verified merges by the pooled precision
# (_pp, row 31). Human-consensus verdicts take no haircut, which is why the
# agent row carries 6 unscaled gold merges and the human row none. The paper
# typesets both the six rate cells and that split (74 = 68 + 6, 95 = 95 + 0).
# The human row has no gold merges only because ver_h, unlike ver_a, does not
# union gold_direct: a human-merge PR in the gold set would drop out of
# h_sh.verified and then be scaled as judge-verified, so the two rows would
# follow different rules while every rate cell still matched.
check(not (gold_direct & set(h_sh.pr_id)),
      f"{len(gold_direct & set(h_sh.pr_id))} human-merge PRs are in the "
      f"human-consensus gold set: ver_h does not union gold_direct, so "
      f"tab:rq1-crosscohort would scale the two rows by different rules")
for _side, _f in (("agent", a_sh), ("human", h_sh)):
    _hv = int(_f.pr_id.isin(gold_direct).sum())
    _jv = int(_f.verified.sum()) - _hv
    _adj = 100 * (_jv * _pp + _hv) / len(_f)
    _alo = 100 * (_jv * (_pc - _ph) + _hv) / len(_f)
    _ahi = 100 * (_jv * (_pc + _ph) + _hv) / len(_f)
    _want = {"agent": (2012, 20.9, 3.68, 3.34, 3.00, 3.52, 68, 6),
             "human": (4063, 14.7, 2.34, 2.10, 1.87, 2.23, 95, 0)}[_side]
    _got = (len(_f), round(100 * _f.has_followup_30d.mean(), 1),
            round(100 * _f.verified.mean(), 2), round(_adj, 2),
            round(_alo, 2), round(_ahi, 2), _jv, _hv)
    check(_got == _want,
          f"tab:rq1-crosscohort {_side} row {_got} != typeset {_want}")
    ledger(32, f"cross-cohort table, {_side} row (shared window)",
           f"n={_got[0]}, candidate {_got[1]}%, verified {_got[2]}% "
           f"({_jv} judge-verified + {_hv} human-consensus), "
           f"adjusted {_got[3]}% [{_got[4]}, {_got[5]}]")

# ---------------------------------------------------------------- row 33
# Why the single-cohort analyses keep their OWN window (typeset in threats).
# The agents ramped at different dates, so the shared cut-off does not shrink
# the cohort evenly: it removes almost all of Cursor and every Copilot
# verified fix, which is what makes it unusable for the per-agent table and
# for RQ4. Same shared frame as rows 3/6/32; a_sh already carries `verified`.
_sh_by_agent = a_sh.groupby("agent").agg(
    n=("pr_id", "size"), ver=("verified", "sum"))
_sh_want = {"Cursor": (4, 0), "Copilot": (136, 0), "Claude_Code": (31, 1),
            "Devin": (1204, 49), "OpenAI_Codex": (637, 24)}
_sh_by_agent = _sh_by_agent.reindex(list(_sh_want)).fillna(0)
for _ag, (_n, _v) in _sh_want.items():
    _got = (int(_sh_by_agent.loc[_ag, "n"]), int(_sh_by_agent.loc[_ag, "ver"]))
    check(_got == (_n, _v),
          f"shared-window frame for {_ag} is {_got}, not {(_n, _v)} -- the "
          f"threats justification for the hybrid window quotes these")
ledger(33, "shared-window per-agent frame (why single-cohort rows keep their own window)",
       "Cursor 4 merges / 0 verified, Copilot 136 / 0, Claude Code 31 / 1, "
       "Codex 637 / 24, Devin 1,204 / 49 -- against own-window 153/8, 721/25, "
       "63/2, 2,049/112, 1,519/56 (row 4). A shared cut-off everywhere would "
       "cost two agents their verified cells and drop RQ4 positives 203 -> 74")

# ---------------------------------------------------------------- row 34
# The shipped `pr_net_files.is_boilerplate` column was precomputed with an
# earlier version of `is_boilerplate` whose `lstrip("./")` ate EVERY leading
# "." and "/" of a path, not just a "./" prefix, so two rule classes never
# fired at repository root: the four tool-config directory prefixes (.cursor/
# .vscode/ .idea/ .devcontainer/) and the dotted basenames (.gitignore,
# .gitattributes, .DS_Store, .release-please-manifest.json), which the old
# rule reduced to `gitignore` and friends -- deliberately EVIDENTIAL names
# (tests/test_boilerplate.py). The pipeline fixed the function (PR #26); the
# stored column still reflects the old rule, and rebuilding it needs the lake
# rebuild. This row proves the difference is inert: 49 filenames change class,
# but no candidate count and no judged Direct-fix pair moves, so every
# published incidence figure holds under either rule.
_stored = {f: b for f, b in con.execute(
    "SELECT filename, any_value(is_boilerplate) FROM pr_net_files "
    "GROUP BY 1").fetchall()}
_reclassified = [f for f, b in _stored.items() if is_boilerplate(f) and not b]
_unclassified = [f for f, b in _stored.items() if b and not is_boilerplate(f)]
check(not _unclassified,
      f"{len(_unclassified)} stored-boilerplate filenames are EVIDENTIAL under "
      f"the corrected rule (e.g. {_unclassified[:5]}): net_fixed below only "
      f"subtracts filenames from the stored column, so it can no longer stand "
      f"in for the corrected predicate and this row must re-link from scratch")
check(len(_reclassified) == 49,
      f"{len(_reclassified)} filenames change boilerplate class, expected 49 "
      f"(0 means the stored column has been rebuilt under the corrected rule, "
      f"in which case row 34 has served its purpose and can be retired)")
_lit = ",".join("'" + f.replace("'", "''") + "'" for f in _reclassified)
_rows = con.execute(
    f"SELECT count(*) FROM pr_net_files WHERE filename IN ({_lit})"
).fetchone()[0] if _reclassified else 0
check(_rows == 167,
      f"{_rows} file-rows change boilerplate class, expected 167")
# tests/test_row34_boilerplate_equivalence.py drives this same substitution
# through rb.compute() on a synthetic lake, so the mechanism this row rests on
# is checked on a fresh checkout even though the counts above need the lake.
rb.FILES_SOURCE["net_fixed"] = rb.net_sub_excluding(_reclassified)
for _label, _kw, _want in (("agent own", {}, (4505, 1031)),
                           ("agent shared", {"cutoff": "shared"}, (2012, 421)),
                           ("human", {"p_side": "human"}, (4063, 596))):
    _f = rb.compute(linker="net_fixed", **_kw)
    _f = _f[_f.censored_30d == 0]
    _got = (len(_f), int(_f.has_followup_30d.sum()))
    check(_got == _want,
          f"under the corrected boilerplate rule the {_label} frame is {_got}, "
          f"not {_want}: the published incidence figures DO depend on the "
          f"stored column and must be recomputed")
# The VERIFIED counts and the judged half of the fixer frame come from the
# judged cases, whose `n_substantive` was counted under the old rule: a pair
# losing its last substantive shared file would leave the candidate frames
# untouched and still move 203 / 95 / 74 and the 333 judged pairs of the
# fixer frame (n=347; the 14 gold pairs never pass through
# `n_substantive`). `n_sub_fixed` recounts the same shared-file
# lists through the corrected function, so this pins the pair level too.
for _side, _j in (("agent", ja), ("human", jh)):
    _d = _j[_j.label.str.startswith("Direct")]
    _lost = set(_d[_d.n_substantive > 0].pair_id) - set(_d[_d.n_sub_fixed > 0].pair_id)
    check(not _lost,
          f"{len(_lost)} judged Direct-fix {_side} pairs lose their last "
          f"substantive shared file under the corrected boilerplate rule "
          f"(pair_ids {sorted(_lost)[:5]}): the verified counts and the fixer "
          f"split are computed from n_substantive and must be recomputed")
ledger(34, "corrected boilerplate rule leaves every candidate count unchanged",
       "49 filenames / 167 file-rows change class (the four tool-config "
       "directory prefixes AND the dotted root basenames, both inert in the "
       "shipped column); frames stay 4,505/1,031, 2,012/421 and 4,063/596, "
       "and no judged Direct-fix pair loses its last substantive shared file")

# ---------------------------------------------------------------- row 35
# tab:rq3-authorclass, LEFT block: the merged PRs that are NOT verified fix
# PRs (RQ3.1 under the disjoint layout, Wannita 2026-08-20). The two blocks
# partition the merged PRs; the union (row 29's all-merged values) moved to
# appendix app:authorclass. Cells: (n_prs, med_commits, share_per_pr,
# pct_all); None = unmeasured (Codex) or not typeset.
_excl_c = all_c[~all_c.pr_id.isin(set(fix_c.pr_id))]
_excl_m = metrics.per_pr_metrics(_excl_c)
_ROW35 = {"Copilot": (1388, 4.0, 81.5, 51.3), "Devin": (1754, 3.0, 79.7, 55.9),
          "Cursor": (546, 2.0, 72.6, 53.5), "Claude_Code": (129, 2.0, 57.2, 40.3),
          "OpenAI_Codex": (2731, 1.0, None, None),
          "Human": (4776, 2.0, 1.0, 0.6)}
for _g, (_n, _med, _sh, _all) in _ROW35.items():
    _r = _excl_m.loc[_g]
    check((int(_r.n_prs), float(_r.med_commits)) == (_n, _med),
          f"row 35 {_g}: n/med {(int(_r.n_prs), float(_r.med_commits))} != {(_n, _med)}")
    if _sh is not None:
        check((_r.share_per_pr, _r.pct_all) == (_sh, _all),
              f"row 35 {_g}: share/all {(_r.share_per_pr, _r.pct_all)} != {(_sh, _all)}")
_excl_all = metrics.per_pr_metrics(
    _excl_c[_excl_c.grp.isin(_MEASURABLE)].assign(grp="All")).loc["All"]
check((int(_excl_all.n_prs), float(_excl_all.med_commits),
       _excl_all.share_per_pr, _excl_all.pct_all) == (3817, 3.0, 78.6, 53.4),
      f"row 35 pooled excl-fix row drifted: "
      f"{(int(_excl_all.n_prs), _excl_all.share_per_pr, _excl_all.pct_all)} "
      f"!= (3,817, 78.6, 53.4)")
ledger(35, "RQ3.1 disjoint block (merged PRs excluding verified fix PRs)",
       "pooled 3,817 PRs, med 3, share/PR 78.6%, all-agent 53.4% "
       "(vs union 78.8/54.1, row 29); per-agent Copilot 81.5 / Devin 79.7 / "
       "Cursor 72.6 / Claude 57.2; Human 4,776 PRs share 1.0%")

# ---------------------------------------------------------------- write
OUT.mkdir(parents=True, exist_ok=True)
pd.DataFrame(LEDGER).to_csv(OUT / "consolidation_gate.csv", index=False)
if _repin_misses:
    print(f"\nREPIN MODE: {len(_repin_misses)} mismatches above -- NOT a passing run")
else:
    print(f"\nALL LEDGER ASSERTS PASSED -> {OUT / 'consolidation_gate.csv'}")
