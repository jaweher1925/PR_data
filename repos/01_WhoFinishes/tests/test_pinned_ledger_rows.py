"""Lake-free re-derivation of the ledger rows the paper typesets.

consolidation_gate.py is the authoritative gate, but it opens the DuckDB lake
(gitignored), so on a fresh checkout it cannot verify anything. The committed
provenance pins -- data/digests/rq3_commit_classes.csv (the reviewer commit
digest, ledger row 28) and data/digests/rq4_signal_stats.csv -- exist so the
digest-backed rows can be re-derived without it. These tests re-derive them:
rows 26/27/29 through the same prfix.common.metrics.per_pr_metrics the gate calls,
row 30 by re-applying the gate's pin block (values plus the paper's p-value
format) to the pinned signal-stats CSV, row 31 -- both spot-check arms and
the pooled estimate the paper scales by -- by re-joining the committed
agent-merge and human-merge labels against their judge model keys, and row 25
(the verified-fix day-30 cumulative incidence) from a fourth committed digest,
data/digests/rq1_survival_frame.csv, through the same prfix.common.metrics.km_cuminc
the figure uses.

Contract: these CSVs are committed generated artifacts (the reviewer digests,
the signal-stats table, and both hosted-labeler exports plus their judge model
keys under human-annotation/labels/judge/), and every number here is typeset in
the paper (tab:rq3-authorclass and its appendix union table, tab:rq4-signals,
the union reference 78.8% quoted in RQ3.1's prose, the abstract's third
finding 87.4%, the RQ4 prose odds ratios, the
per-side 27/30 = 90% spot-checks, and the pooled 54/60 = 90% with its Wilson
80-95% bounds). A mismatch means the paper quotes numbers the pinned artifacts
no longer carry.

Row 32 (tab:rq1-crosscohort) adds a second contract: paper-emse/sections/
results.tex is the typeset table itself, parsed into row -> cells here so its
six rate cells can be checked against the rates this file derives. That row's
four shared-window counts per cohort stay pinned -- they need the lake, and the
gate asserts them there. The two Mantel-Haenszel odds ratios (rows 3/6) are
pinned for the typeset check but also re-derived from the third committed
digest, data/digests/rq1_crosscohort_strata.csv (the shared-window per-repo
counts, ledger row 28), through the same repo_tables()/mh() the gate calls.

Row 33 (the hybrid-window justification) is guarded the same way: its
per-agent shared-window counts need the lake, so they stay pinned, but they are
checked against row 32's agent row, against the own-window PR counts
tab:rq3-verified typesets, and against the threats sentence that quotes two of
them.

Row 8 (the per-agent candidate rates) is guarded the same way as row 33: its
own-window repository structure (606 repositories, 39 of them hosting more than
one agent) needs the lake and stays pinned, but both sentences that quote it --
results.tex and threats.tex -- must carry the pair together with the merge count
that identifies the frame, and the committed shared-window digest must stay a
subset of it.

Row 34 (the corrected boilerplate rule is inert) splits the same way: the
candidate-frame half needs the linker and stays in the gate -- its mechanism is
exercised on a synthetic lake by tests/test_row34_boilerplate_equivalence.py --
while the pair level is re-derived here from the committed judged-case files,
whose per-file `boilerplate` flag carries the shipped (old-rule) classification
the published counts were computed with.

The adjusted rates in tab:rq3-verified are not pinned here: they need the
gitignored lake, so rq1_verified_rates.py asserts them against the typeset
table instead.
"""

from __future__ import annotations

import json
import math
import re

import pandas as pd
import pytest

from prfix.common import metrics
from prfix.rq1_fix_rates.boilerplate import is_boilerplate
from tests._texguard import read_tex

MEASURABLE = ["Copilot", "Devin", "Cursor", "Claude_Code"]  # signing agents


@pytest.fixture(scope="module")
def commits() -> pd.DataFrame:
    return pd.read_csv(metrics.DIGEST_COMMITS)


@pytest.fixture(scope="module")
def sig_stats() -> pd.DataFrame:
    return pd.read_csv(metrics.DIGESTS_DIR / "rq4_signal_stats.csv").set_index(
        ["outcome", "signal"])


def _pstr(p: float) -> str:
    """The paper's p-value format: <0.001, else 2 significant digits."""
    return "<0.001" if float(p) < 1e-3 else f"{float(p):.2g}"


# grp: (n_prs, med_commits, share_per_pr, pct_any, pct_all); None = unmeasured
ROW26_ALL = {
    "Human":        (4900, 2.0, 0.9, 1.7, 0.6),
    "OpenAI_Codex": (2834, 1.0, None, None, None),
    "Devin":        (1813, 3.0, 80.1, 99.5, 56.9),
    "Copilot":      (1428, 4.0, 81.6, 98.7, 51.8),
    "Cursor":       (569, 2.0, 72.9, 95.6, 54.1),
    "Claude_Code":  (130, 2.5, 57.1, 79.2, 40.0),
}
ROW27_FIX = {
    "Human":        (124, 2.0, 0.8, 0.8, 0.8),
    "OpenAI_Codex": (103, 1.0, None, None, None),
    "Devin":        (59, 1.0, 92.0, 98.3, 84.7),
    "Copilot":      (40, 3.0, 86.6, 100.0, 70.0),
    "Cursor":       (23, 1.0, 79.1, 91.3, 69.6),
    "Claude_Code":  (1, 8.0, None, None, None),   # n=1: counts only
}


@pytest.mark.parametrize("side,expect", [("all", ROW26_ALL), ("fix", ROW27_FIX)])
def test_per_group_rows_from_commit_digest(commits, side, expect):
    """Ledger rows 26/27: the per-agent rows of tab:combined-authorclass."""
    m = metrics.per_pr_metrics(commits[commits.side == side])
    for grp, (n, med, sh, any_, all_) in expect.items():
        r = m.loc[grp]
        assert (int(r.n_prs), float(r.med_commits)) == (n, med), grp
        if sh is not None:
            assert (r.share_per_pr, r.pct_any, r.pct_all) == (sh, any_, all_), grp


@pytest.mark.parametrize("side,expect", [
    ("all", (3940, 3.0, 78.8, 54.1)),   # union reference (app:authorclass)
    ("fix", (123, 2.0, 87.4, 76.4)),    # abstract finding (3) = 87.4%
])
def test_overall_rows_from_commit_digest(commits, side, expect):
    """Ledger row 29: the All (excl. Codex) rows, pooling the four signing
    agents only."""
    c = commits[(commits.side == side) & commits.grp.isin(MEASURABLE)]
    r = metrics.per_pr_metrics(c.assign(grp="All")).loc["All"]
    assert (int(r.n_prs), float(r.med_commits),
            r.share_per_pr, r.pct_all) == expect


def test_the_two_blocks_partition_the_merged_prs(commits):
    """tab:rq3-authorclass's caption states that its two blocks PARTITION the
    merged PRs: RQ3.1 is the merged PRs that are not verified fix PRs, RQ3.2 is
    the verified fix PRs. That holds only if every fix-side PR is itself an
    all-side PR, so the subset relation and the resulting split of the pooled
    3,940 into 3,817 + 123 are re-derived from the commit digest here."""
    allp, fixp = commits[commits.side == "all"], commits[commits.side == "fix"]
    assert set(fixp.pr_id) <= set(allp.pr_id)          # all 350 fix PRs

    base = set(allp[allp.grp.isin(MEASURABLE)].pr_id)
    overlap = set(fixp[fixp.grp.isin(MEASURABLE)].pr_id)
    assert (len(base), len(overlap)) == (3940, 123)
    assert len(base - overlap) == 3817


# (outcome, signal): (cliffs_delta 3dp, or_10x 2dp, MWU p, clogit p as typeset;
# None = not typeset, so unpinned)
ROW30_SIGNALS = {
    ("has_followup_30d", "hours_under_review"): (-0.100, 1.14, "<0.001", None),
    ("has_followup_30d", "n_commits"):          (0.028, 1.71, "0.15", None),
    ("has_followup_30d", "lines_post_first"):   (0.024, 1.05, "0.21", None),
    ("has_followup_30d", "n_review_items"):     (-0.101, 0.99, "<0.001", None),
    ("verified", "hours_under_review"):         (-0.221, 1.23, "<0.001", "0.18"),
    ("verified", "n_commits"):                  (0.013, 6.06, "0.75", "<0.001"),
    ("verified", "lines_post_first"):           (0.032, 1.34, "0.41", "<0.001"),
    ("verified", "n_review_items"):             (-0.239, 2.41, "<0.001", "0.022"),
}


@pytest.mark.parametrize("key,expect", list(ROW30_SIGNALS.items()))
def test_signal_stats_pin(sig_stats, key, expect):
    """Ledger row 30: every cell of tab:rq4-signals and every odds ratio the
    RQ4 prose quotes, in the format the paper prints them."""
    dlt, or10, mwu_p, clog_p = expect
    r = sig_stats.loc[key]
    assert round(float(r.cliffs_delta), 3) == dlt
    assert round(float(r.clogit_or_10x), 2) == or10
    assert _pstr(r.mwu_p) == mwu_p
    if clog_p is not None:
        assert _pstr(r.clogit_p) == clog_p


def _spotcheck_direct(stem: str, annotator: str) -> tuple[int, int, int]:
    """(pairs, judge Direct calls, calls the annotator confirmed) for one
    enriched blind spot-check sheet."""
    d = metrics.PROJECT_ROOT / "human-annotation" / "labels" / "judge"
    df = pd.read_csv(d / f"{stem}_annotator_{annotator}.csv")[
        ["pair_id", "label"]].merge(
        pd.read_csv(d / f"{stem}_model_key.csv")[["pair_id", "label"]].rename(
            columns={"label": "llm"}), on="pair_id", validate="1:1")
    direct = df[df.llm == "Direct fix"]
    return len(df), len(direct), int((direct.label == "Direct fix").sum())


def test_humanp_spotcheck_precision():
    """Ledger row 31: the judge's Direct-fix precision on human-merge pairs,
    typeset at three sites (RQ1 approach, RQ1.2 results, threats)."""
    n, nd, ok = _spotcheck_direct("rq1_judge_sample50_enriched_humanP", "nhat3")
    assert (n, nd, ok) == (50, 30, 27)
    assert round(100 * ok / nd, 1) == 90.0


def test_pooled_spotcheck_precision_and_wilson():
    """Ledger row 31, pooled arm: both blind spot-checks return 27/30, and the
    paper scales the adjusted rates in tab:rq3-verified by their pooled
    estimate (54/60 = 90%, Wilson 80-95%). The bounds are typeset, so both arms
    and the interval are re-derived from the label files here rather than
    trusted to rq1_verified_rates.py, which needs the lake to run."""
    agent = _spotcheck_direct("rq1_judge_sample50_enriched",
                              "anon_NHATDUONG")
    human = _spotcheck_direct("rq1_judge_sample50_enriched_humanP", "nhat3")
    assert agent[1:] == (30, 27) and human[1:] == (30, 27)

    ok, nd = agent[2] + human[2], agent[1] + human[1]
    assert (ok, nd) == (54, 60)
    p, z = ok / nd, 1.96
    denom = 1 + z ** 2 / nd
    centre = (p + z ** 2 / (2 * nd)) / denom
    half = z * math.sqrt(p * (1 - p) / nd + z ** 2 / (4 * nd ** 2)) / denom
    assert round(100 * p, 1) == 90.0
    assert (round(100 * (centre - half)), round(100 * (centre + half))) == (80, 95)



# tab:rq3-authorclass LEFT block (RQ3.1 disjoint: merged PRs that are not
# verified fix PRs): (n_prs, med_commits, share_per_pr, pct_all); None =
# unmeasured. The union values stay pinned in ROW26_ALL (appendix table).
ROW35_EXCL = {
    "Human":        (4776, 2.0, 1.0, 0.6),
    "OpenAI_Codex": (2731, 1.0, None, None),
    "Devin":        (1754, 3.0, 79.7, 55.9),
    "Copilot":      (1388, 4.0, 81.5, 51.3),
    "Cursor":       (546, 2.0, 72.6, 53.5),
    "Claude_Code":  (129, 2.0, 57.2, 40.3),
}


def test_rq31_disjoint_block_from_commit_digest(commits):
    """Ledger row 35: the RQ3.1 block of tab:rq3-authorclass -- the all-side
    digest minus the fix-side PR ids. Pooled row is the paper's RQ3.1 numbers
    (78.6 / 53.4)."""
    fix_ids = set(commits[commits.side == "fix"].pr_id)
    excl = commits[(commits.side == "all") & ~commits.pr_id.isin(fix_ids)]
    m = metrics.per_pr_metrics(excl)
    for grp, (n, med, sh, all_) in ROW35_EXCL.items():
        r = m.loc[grp]
        assert (int(r.n_prs), float(r.med_commits)) == (n, med), grp
        if sh is not None:
            assert (r.share_per_pr, r.pct_all) == (sh, all_), grp
    pooled = metrics.per_pr_metrics(
        excl[excl.grp.isin(MEASURABLE)].assign(grp="All")).loc["All"]
    assert (int(pooled.n_prs), float(pooled.med_commits),
            pooled.share_per_pr, pooled.pct_all) == (3817, 3.0, 78.6, 53.4)



def test_verified_km_day30_from_survival_digest():
    """Ledger row 25: the verified-fix KM cumulative incidence at day 30
    (4.5% agent vs 2.6% human), re-derived lake-free from the committed
    survival digest through the same km_cuminc the figure uses, and checked
    against the two values the RQ1 survival sentence typesets -- results.tex
    is the contract on that side, as it is for row 32's table cells."""
    sv = pd.read_csv(metrics.DIGEST_SURVIVAL)
    want = {"Agent": 4.5, "Human": 2.6}
    for cohort, val in want.items():
        d = sv[sv.cohort == cohort]
        _, ci = metrics.km_cuminc(d.ver_time.to_numpy(), d.ver_event.to_numpy())
        assert round(100 * float(ci[-1]), 1) == val, cohort

    sentence = re.search(r"verified-fix curves reach\s+"
                         r"\$([\d.]+)\\,\\%\$ vs\.\\ \$([\d.]+)\\,\\%\$"
                         r" by day 30", read_tex(RESULTS_TEX))
    assert sentence, "the RQ1 survival sentence no longer typesets both day-30 rates"
    assert [float(g) for g in sentence.groups()] == [want["Agent"], want["Human"]], \
        "the typeset day-30 rates drifted from the survival digest"


def test_review_signal_deltas_are_negative(sig_stats):
    """The RQ4 framing rests on the sign: pooled review-signal deltas are
    NEGATIVE (merges with a follow-up fix sit in fast-review repos) while
    the within-repo conditional logit is positive."""
    for outcome in ("has_followup_30d", "verified"):
        for signal in ("hours_under_review", "n_review_items"):
            assert float(sig_stats.loc[(outcome, signal)].cliffs_delta) < 0
    assert float(sig_stats.loc[("verified", "n_commits")].clogit_or_10x) > 1


# Shared-window counts for tab:rq1-crosscohort, per cohort row as the table
# labels it: (merges, candidate fixes, judge-verified merges, human-consensus
# verified merges). These four come from the frames only the lake can build
# (consolidation_gate.py ledger rows 3/6/32 asserts them there); every rate the
# table typesets is derived from them below, so nothing else is pinned here.
ROW32_COUNTS = {"AI agents": (2012, 421, 68, 6), "Humans": (4063, 596, 95, 0)}

# The odds-ratio row: (candidate OR + CI, verified OR + CI, strata). Pinned for
# the typeset check below, and re-derived from the committed strata digest by
# test_crosscohort_mh_from_strata_digest.
ROW32_MH = ((1.41, 1.19, 1.68), (1.62, 1.10, 2.39), 218)
# The two CMH p-values rows 3 and 6 quote, per outcome column of the digest.
# As typeset: the RQ1.1 paragraph prints the candidate p in scientific
# notation, Table 2's companion prose prints the verified p to two significant
# figures. Pinned as the printed strings so the test tracks the paper's own
# precision (0.015317 rounds to 0.015; a tighter relative tolerance failed on
# exactly that gap).
ROW32_MH_P = {"n_candidate": "4.3e-05", "n_verified": "0.015"}

RESULTS_TEX = metrics.PROJECT_ROOT.parent / "paper-emse" / "sections" / "results.tex"


def _pooled_precision() -> tuple[float, float, float]:
    """(point, Wilson low, Wilson high) Direct-fix precision over both blind
    spot-checks -- the factor tab:rq1-crosscohort's Adjusted column scales by."""
    agent = _spotcheck_direct("rq1_judge_sample50_enriched", "anon_NHATDUONG")
    human = _spotcheck_direct("rq1_judge_sample50_enriched_humanP", "nhat3")
    ok, nd = agent[2] + human[2], agent[1] + human[1]
    p, z = ok / nd, 1.96
    denom = 1 + z ** 2 / nd
    centre = (p + z ** 2 / (2 * nd)) / denom
    half = z * math.sqrt(p * (1 - p) / nd + z ** 2 / (4 * nd ** 2)) / denom
    return p, centre - half, centre + half


def _typeset_rows(label: str) -> dict[str, list[str]]:
    r"""The tabular carrying `label` as {first cell: remaining cells}, with LaTeX
    markup normalised away (\shortstack line breaks become spaces, 2{,}012
    becomes 2012, en-dash ranges become hyphens). The .tex table is the
    contract: it is what the reader sees, and the gate cannot check it here."""
    env = re.search(r"\\begin\{table\}(?:(?!\\end\{table\}).)*?"
                    r"\\label\{" + re.escape(label) + r"\}.*?\\end\{table\}",
                    read_tex(RESULTS_TEX), re.S)
    assert env, f"{label} is no longer typeset in {RESULTS_TEX.name}"
    body = re.search(r"\\begin\{tabular\}\{[^}]*\}(.*?)\\end\{tabular\}",
                     env.group(0), re.S).group(1)
    rows = {}
    for raw in re.split(r"\\\\\s*\n", body):
        line = re.sub(r"\\(?:top|mid|bottom)rule", "", raw)
        line = line.replace("\\\\", " ")                        # in-cell \shortstack breaks
        line = re.sub(r"\\[a-zA-Z]+(?:\[[a-z]\])?", " ", line)    # \textbf, \shortstack[r], ...
        line = line.replace("{,}", "").replace("---", "-").replace("--", "-")
        cells = [" ".join(c.replace("$", "").replace("{", " ").replace("}", " ").split())
                 for c in line.split("&")]
        if any(cells) and len(cells) > 1:
            rows[cells[0]] = cells[1:]
    return rows


def test_crosscohort_table_matches_typeset():
    """Ledger row 32: every rate cell tab:rq1-crosscohort typesets, derived from
    the shared-window counts rather than pinned. Candidate and verified are
    those counts as percentages; Adjusted scales the judge-verified merges by
    the pooled spot-check precision and leaves human-consensus verdicts
    unscaled, so its bracketed range comes from the precision interval only."""
    typeset = _typeset_rows("tab:rq1-crosscohort")
    p, lo, hi = _pooled_precision()

    for row, (n, cand, jv, hv) in ROW32_COUNTS.items():
        adj = tuple(100 * (jv * f + hv) / n for f in (p, lo, hi))
        assert typeset[row] == [
            str(n),
            f"{100 * cand / n:.1f}",
            f"{100 * (jv + hv) / n:.2f}",
            f"{adj[0]:.2f} ({adj[1]:.2f}-{adj[2]:.2f})",
        ], f"{row} row of tab:rq1-crosscohort drifted from the derived rates"
        # The adjusted rate is a haircut on the verified rate, never a raise.
        assert adj[1] <= adj[0] <= 100 * (jv + hv) / n and adj[0] <= adj[2]
        # Only the agent side carries gold pairs, which is what makes the two
        # rows follow the same scaling rule (the gate asserts the set-level
        # version: no human-merge PR is in the gold set).
        assert (hv > 0) == (row == "AI agents")

    (c_or, c_lo, c_hi), (v_or, v_lo, v_hi), strata = ROW32_MH
    assert typeset[f"Odds ratio, within repositories ({strata} strata)"] == [
        "", f"{c_or:.2f} [{c_lo:.2f}-{c_hi:.2f}]",
        f"{v_or:.2f} [{v_lo:.2f}-{v_hi:.2f}]", "-"]


def test_crosscohort_mh_from_strata_digest():
    """Ledger rows 3/6: both Mantel-Haenszel odds ratios, their CIs and their
    p-values re-derived from the committed cross-cohort strata digest through
    the same repo_tables()/mh() the gate calls -- no lake, no re-implementation.
    The digest is a committed generated artifact (row 28's contract), so this
    runs on a fresh checkout. Only a lake run of consolidation_gate.py can
    rebuild it, so a tree that has dropped it skips rather than passes
    silently."""
    if not metrics.DIGEST_CROSSCOHORT.exists():
        pytest.skip(f"{metrics.DIGEST_CROSSCOHORT.name} not committed: rows 3/6 stay "
                    "pinned until a lake run of consolidation_gate.py writes it")
    digest = pd.read_csv(metrics.DIGEST_CROSSCOHORT)
    cand, ver, strata = ROW32_MH

    for col, want in (("n_candidate", cand), ("n_verified", ver)):
        tables = metrics.repo_tables(*metrics.crosscohort_frames(digest, col), "y")
        assert len(tables) == strata, f"{col} strata drifted from {strata}"
        or_, lo, hi, pval, _ = metrics.mh(tables)
        assert (round(or_, 2), round(lo, 2), round(hi, 2)) == want
        # p-values are typeset to the precision the paper prints them at.
        assert f"{pval:.2g}" == ROW32_MH_P[col]


# ---------------------------------------------------------------- row 33
# The shared-window agent frame per agent, keyed as tab:rq3-verified labels the
# agents: (merges, verified merges). Lake-derived, so pinned -- but not free
# floating: it must add up to row 32's agent row, every agent must observe more
# under its own window (the own-window frame IS tab:rq3-verified), and it must
# strand exactly the agents the hybrid-window justification in threats names.
ROW33_SHARED = {"OpenAI Codex": (637, 24), "Cursor": (4, 0), "Devin": (1204, 49),
                "Claude Code": (31, 1), "GitHub Copilot": (136, 0)}

THREATS_TEX = metrics.PROJECT_ROOT.parent / "paper-emse" / "sections" / "threats.tex"
NUM_WORD = {0: "no", 1: "one", 2: "two", 3: "three", 4: "four", 5: "five"}


def test_shared_window_frame_adds_up_to_the_crosscohort_agent_row():
    """Ledger row 33 against rows 32 and 4: the per-agent shared-window frame is
    a partition of the cross-cohort agent row, and it is narrower for every
    agent than the own-window frame tab:rq3-verified typesets."""
    n = sum(v[0] for v in ROW33_SHARED.values())
    ver = sum(v[1] for v in ROW33_SHARED.values())
    agent_n, _, jv, hv = ROW32_COUNTS["AI agents"]
    assert (n, ver) == (agent_n, jv + hv), \
        "the per-agent shared-window frame does not partition the agent row"

    typeset = _typeset_rows("tab:rq3-verified")
    own = {a: int(typeset[a][0]) for a in ROW33_SHARED}
    for agent, (shared_n, _) in ROW33_SHARED.items():
        assert shared_n < own[agent], \
            f"{agent}: shared window {shared_n} is not narrower than own {own[agent]}"
    assert sum(own.values()) == int(typeset["Overall"][0])


def test_hybrid_window_justification_names_the_stranded_agents():
    """threats.tex is the contract: the sentence that keeps the single-cohort
    rates on their own window quotes the shared-window frame, so its two figures
    must be row 33's, and no agent that still has a verified merge there may be
    cited as stranded."""
    text = " ".join(read_tex(THREATS_TEX).replace("---", " ").split())
    match = re.search(r"The single-cohort rates keep each cohort's own window[^.]+\.", text)
    assert match, "threats no longer justifies the hybrid observation window"
    sentence = match.group(0)

    # Cursor is quoted by merge count, Copilot by the absence of a verified fix;
    # both words come from the pinned frame, not from the prose.
    assert (f"Cursor with {NUM_WORD[ROW33_SHARED['Cursor'][0]]} observable merges"
            in sentence)
    assert (f"GitHub Copilot with {NUM_WORD[ROW33_SHARED['GitHub Copilot'][1]]} "
            f"verified fix" in sentence)

    for agent, (_, shared_ver) in ROW33_SHARED.items():
        assert (agent in sentence) == (shared_ver == 0), \
            f"{agent} keeps {shared_ver} verified merges under the shared window"


# ---------------------------------------------------------------- row 8
# The own-window agent frame's repository structure, as (repositories,
# repositories hosting more than one agent). Lake-derived like ROW33_SHARED --
# the gate asserts it on the frame itself -- but not free floating: both
# typeset sites quote the pair as the reason the per-agent ordering is read
# qualitatively, they must name the frame tab:rq3-verified totals, and the
# committed shared-window digest's agent repositories are a subset of these
# (the shared cut-off only censors more merges).
ROW8_OWN_REPOS = (606, 39)

_MULTI_AGENT_RE = (r"(\d+) of the (\d+) repositories in (?:this|the) (\d+)-merge"
                   r"(?: agent)? frame host more than one agent")


@pytest.mark.parametrize("tex", ["results", "threats"])
def test_multi_agent_repo_counts_match_both_typeset_sites(tex):
    """Ledger row 8: the two sentences that disclaim the per-agent ordering
    (results.tex RQ3.1 and threats.tex external validity) must quote the pinned
    own-window repository structure, and must identify the frame by the merge
    count tab:rq3-verified totals -- 606 is otherwise checkable against neither
    the 891 merged-PR repositories nor the 1,479 population."""
    repos, multi = ROW8_OWN_REPOS
    path = {"results": RESULTS_TEX, "threats": THREATS_TEX}[tex]
    text = " ".join(read_tex(path).replace("{,}", "").split())
    match = re.search(_MULTI_AGENT_RE, text)
    assert match, f"{path.name} no longer qualifies the 606 by its frame"
    own_n = int(_typeset_rows("tab:rq3-verified")["Overall"][0])
    assert (int(match.group(1)), int(match.group(2)), int(match.group(3))) \
        == (multi, repos, own_n), \
        f"{path.name} drifted from the pinned own-window frame structure"


def test_shared_window_repositories_are_a_subset_of_the_own_window_frame():
    """Row 8 against row 28's digest: the shared cut-off censors a superset of
    the merges the own cut-off censors, so the committed cross-cohort digest
    cannot carry more agent repositories than the own-window frame has, and
    only repositories in that frame can host two agents."""
    if not metrics.DIGEST_CROSSCOHORT.exists():
        pytest.skip(f"{metrics.DIGEST_CROSSCOHORT.name} not committed: row 8's frame "
                    "check stays pinned until a lake run writes it")
    digest = pd.read_csv(metrics.DIGEST_CROSSCOHORT)
    shared_repos = digest[digest.cohort == "Agent"].repo_id.nunique()
    repos, multi = ROW8_OWN_REPOS
    assert shared_repos <= repos, \
        f"shared-window agent repositories ({shared_repos}) exceed the own-window {repos}"
    assert multi <= repos

# ---------------------------------------------------------------- row 34
# The judged-case files carry, per shared file, the `boilerplate` flag as it was
# SHIPPED -- the same old-rule classification the lake column pr_net_files
# .is_boilerplate holds -- so the pair level of row 34 can be re-derived without
# the lake. The candidate-frame half (4,505/1,031, 2,012/421, 4,063/596) needs
# the linker and stays in the gate.
JUDGED_POPULATIONS = [("agent", metrics.AGENT_CASES, metrics.AGENT_JUDGE),
                      ("human", metrics.HUMAN_CASES, metrics.HUMAN_JUDGE)]


def _judged(cases_json, judge_csv) -> pd.DataFrame:
    """Judge verdicts joined to the shared-file lists, with the substantive
    count recomputed through the CURRENT boilerplate rule (`n_sub_fixed`)
    alongside the count shipped under the old one (`n_substantive`)."""
    c = pd.DataFrame(json.loads(cases_json.read_text()))
    c["pair_id"] = c["pair_id"].astype(int)
    c["n_substantive"] = c["n_substantive"].astype(int)
    c["n_sub_fixed"] = [sum(not is_boilerplate(f["name"]) for f in shared)
                        for shared in c["shared"]]
    j = pd.read_csv(judge_csv)
    j["label"] = j["label"].fillna("ERROR")
    return j.merge(c[["pair_id", "shared", "n_substantive", "n_sub_fixed"]],
                   on="pair_id", validate="1:1")


@pytest.mark.parametrize("side,cases_json,judge_csv", JUDGED_POPULATIONS)
def test_shipped_boilerplate_flag_is_the_old_rules_classification(
        side, cases_json, judge_csv):
    """The premise of row 34: `n_substantive` is exactly the count of shared
    files the shipped per-file flag calls evidential, so that flag -- not the
    current function -- is what the published counts were computed with."""
    c = pd.DataFrame(json.loads(cases_json.read_text()))
    shipped = [sum(not f["boilerplate"] for f in shared) for shared in c["shared"]]
    assert list(c["n_substantive"].astype(int)) == shipped


@pytest.mark.parametrize("side,cases_json,judge_csv", JUDGED_POPULATIONS)
def test_corrected_boilerplate_rule_only_ever_adds_dotted_root_files(
        side, cases_json, judge_csv):
    """The bug the PR #26 review fixed, on real data: paths whose dotted root
    component the old `lstrip("./")` ate are boilerplate under the corrected
    rule and evidential in the shipped classification. The reclassification must
    be non-empty (or the rest of row 34 is vacuous), must be confined to the
    dotted classes the appendix table lists, and must never run the other way --
    the gate subtracts filenames from the stored column, which only works while
    the corrected rule is a superset of the old one."""
    c = pd.DataFrame(json.loads(cases_json.read_text()))
    files = {f["name"]: f["boilerplate"] for shared in c["shared"] for f in shared}
    added = {f for f, was in files.items() if is_boilerplate(f) and not was}
    dropped = {f for f, was in files.items() if was and not is_boilerplate(f)}

    assert added, "no file changes class: the reclassification claim is vacuous"
    assert not dropped, f"{side}: corrected rule frees {sorted(dropped)[:5]}"
    for f in added:
        assert f.startswith("."), f"{f} is not a dotted root path"


@pytest.mark.parametrize("side,cases_json,judge_csv", JUDGED_POPULATIONS)
def test_no_judged_direct_fix_pair_loses_its_last_substantive_file(
        side, cases_json, judge_csv):
    """Row 34's pair level: the verified counts (203 / 95 / 74) and the RQ3
    fixer split are computed from `n_substantive`, so a Direct-fix pair dropping
    to zero substantive shared files under the corrected rule would move them.
    None does -- every reclassified file sits in a pair that keeps other
    evidence, or in a pair the judge did not call a direct fix."""
    m = _judged(cases_json, judge_csv)
    direct = m[m.label.str.startswith("Direct")]
    lost = set(direct[direct.n_substantive > 0].pair_id) - \
        set(direct[direct.n_sub_fixed > 0].pair_id)
    assert not lost, f"{side}: pairs {sorted(lost)[:5]} lose their last file"
    assert (direct.n_substantive > 0).any(), "no Direct-fix pair to check"


def test_fixpr_total_and_opener_split_match_the_digest_and_the_rq2_frame(commits):
    """The two sentences that bridge RQ2 and RQ3.2 quote both grouping axes at
    once, so each of their numbers is bound to its source here: the 350 fix PRs
    and their opener split come from the commit digest's fix side (the same
    artifact rows 27/29 use), and RQ2's pair counts from tab:rq3-fixer's two
    frame rows. Neither sentence can survive a digest or frame change."""
    text = " ".join(read_tex(RESULTS_TEX).split()).replace("{,}", "").replace("$", "")
    fix = commits[commits.side == "fix"]
    by_opener = fix.groupby("grp").pr_id.nunique()
    agents, codex, human = (int(by_opener[MEASURABLE].sum()),
                            int(by_opener["OpenAI_Codex"]), int(by_opener["Human"]))
    total = int(fix.pr_id.nunique())
    assert (agents, codex, human) == (123, 103, 124)
    assert total == agents + codex + human           # the openers partition the 350

    split = re.search(r"Those (\d+) are the fix PRs of every verified pair on "
                      r"both cohorts under the unrestricted rule, split by "
                      r"opener into (\d+) attributable agent-opened, (\d+) "
                      r"Codex-opened, and (\d+) human-opened", text)
    assert split, "RQ2 no longer derives the fix-PR total in place"
    assert [int(g) for g in split.groups()] == [total, agents, codex, human]

    fixer = _typeset_rows("tab:rq3-fixer")
    bridge = re.search(r"\(n=(\d+) agent-merge, n=(\d+) human-merge", text)
    assert bridge, "RQ3.2 no longer states the axis RQ2 grouped by"
    assert list(bridge.groups()) == [fixer["Overall (agent merges)"][0],
                                     fixer["Human"][0]]
    assert re.search(rf"counts the {total} unique fix PRs themselves, grouped "
                     r"by opener", text), \
        "RQ3.2's bridge no longer names the fix-PR total as its own axis"
