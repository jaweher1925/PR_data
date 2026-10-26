"""RQ2 per-agent fixer split (paper table `tab:rq3-fixer`).

consolidation_gate.py pins the per-agent split but needs the full DuckDB lake
(gitignored) to run, so it cannot gate the numbers on a fresh checkout.

Two frames matter. The UNRESTRICTED frame is every verified direct-fix pair:
the judge's Direct verdicts plus the 14 two-annotator gold pairs. It is fully
recoverable lake-free for the two columns this file derives -- pair count and
human-fixed count per merging agent -- the judged half from the shipped judge
CSV plus its case file, the gold half from the committed agreed-labels CSV
(whose `agent` column is the merging agent) plus the pinned fixer split of
those 14 pairs. Since 2026-08-20 the paper typesets the RESTRICTED frame
instead: the same pairs limited to the agent cohort's uncensored merges. Which
merges those are needs the lake, so the gate writes that membership out as the
committed digest data/digests/rq2_fixer_frame.csv (one row per unrestricted
pair, `in_frame` flag; ledger row 28's contract). The digest is committed, so
the whole restricted split is re-derived here through the same
per_agent_split() the gate calls, on a fresh checkout. Only a lake run of
consolidation_gate.py can rebuild it, so a tree that has dropped it falls back
to FRAME below as a pin (the gate's `_expect`) and skips the digest-backed
tests rather than passing silently.

The pin is not free floating. Every cell Table tab:rq3-fixer typesets is
checked against it, and the pin is checked against the unrestricted frame
derived here: restricting merges can only remove pairs, so no restricted count
may exceed its unrestricted counterpart. The judge-only sub-frame is asserted
separately so a drift in either half of the unrestricted frame is attributable.

The self vs other-agent split needs the PR table's agent column for the
FIXING PR, which the artifacts only carry for fixing PRs that are themselves
judged: that subset is classified end-to-end here and bounded against the
paper's other-agent counts, and the classification and aggregation
(fixer.add_fixer / per_agent_split) are additionally exercised on a
synthetic frame.

Skips when the (gitignored) judge artifacts are absent.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd
import pytest

from prfix.common.metrics import AGREED
from prfix.config import JUDGE_DIR
from prfix.rq2_fix_authors import fixer
from prfix.rq2_fix_authors.fixer import (
    DIGEST_FIXER,
    add_fixer,
    fixer_digest,
    per_agent_split,
    write_fixer_digest,
)
from tests._texguard import read_tex
from tests.test_pinned_ledger_rows import _typeset_rows

JUDGE_CSV = JUDGE_DIR / "rq1_llm_judge_pop_anthropic_claude-opus-4-8_low.csv"
CASES_JSON = JUDGE_DIR / "rq1_judge_cases_population_12k.json"

# agent -> (direct-fix pairs, human-fixed pairs) on the JUDGE-ONLY sub-frame:
# the judge's Direct verdicts alone. NOT the paper's frame -- Table
# tab:rq3-fixer is typeset on the judge verdicts PLUS the gold pairs,
# restricted to the uncensored merges (see UNRESTRICTED and FRAME below).
# Held separately so a drift is attributable to one half.
JUDGED = {
    "OpenAI_Codex": (167, 48),
    "Devin":        (87, 17),
    "Copilot":      (56, 2),
    "Cursor":       (22, 0),
    "Claude_Code":  (1, 0),
}
JUDGED_TOTAL = (333, 67)   # 20.1 % human-fixed

# Fixer split of the 14 gold pairs by merging agent, (self, human,
# other-agent), pinned from consolidation_gate.py's `_expect`. The pair
# COUNTS are re-derived from the committed agreed-labels CSV below; which
# cohort opened each gold fixing PR needs the (gitignored) lake, so that part
# is a pin.
GOLD_SPLIT = {
    "Devin":       (2, 4, 0),
    "Cursor":      (5, 0, 0),
    "Copilot":     (1, 0, 0),
    "Claude_Code": (0, 1, 1),
}

# The UNRESTRICTED frame (judge + gold, no frame filter): agent -> (pairs,
# human-fixed %). Asserted by the gate as row 9's robustness variant and quoted
# once in RQ2's mirror paragraph (76.7 % self / 20.7 % human overall); NOT the
# frame Table tab:rq3-fixer typesets, see FRAME.
UNRESTRICTED = {
    "OpenAI_Codex": (167, 28.7),
    "Devin":        (93, 22.6),
    "Copilot":      (57, 3.5),
    "Cursor":       (27, 0.0),
    "Claude_Code":  (3, 33.3),
}
UNRESTRICTED_OVERALL = (347, 20.7)

# The RESTRICTED frame Table tab:rq3-fixer typesets: agent -> (pairs, self,
# human, other-agent) counts, pinned from consolidation_gate.py's `_expect`
# (the frame filter needs the lake). Keyed as the table labels the agents.
FRAME = {
    "OpenAI_Codex": (125, 74, 48, 3),
    "Devin":        (88,  64, 21, 3),
    "Copilot":      (40,  38,  2, 0),
    "Cursor":       (8,    7,  0, 1),
    "Claude_Code":  (2,    0,  1, 1),
}
FRAME_LABEL = {"OpenAI_Codex": "OpenAI Codex", "Devin": "Devin",
               "Copilot": "GitHub Copilot", "Cursor": "Cursor",
               "Claude_Code": "Claude Code"}
# The Human row of the same table (ledger row 11, uncensored human frame).
FRAME_HUMAN_ROW = (110, 89.1, 10.9)


@pytest.fixture(scope="module")
def cases() -> pd.DataFrame:
    for path in (JUDGE_CSV, CASES_JSON):
        if not Path(path).exists():
            pytest.skip(f"judge artifact missing: {path}")
    c = pd.DataFrame(json.loads(Path(CASES_JSON).read_text()))
    for col in ("pair_id", "p_id", "q_id", "n_substantive"):
        c[col] = c[col].astype(int)
    return c


@pytest.fixture(scope="module")
def direct_fix_pairs(cases) -> pd.DataFrame:
    judged = pd.read_csv(JUDGE_CSV)
    judged["label"] = judged["label"].fillna("ERROR")
    merged = judged.merge(cases[["pair_id", "p_id", "q_id", "n_substantive"]],
                          on="pair_id", validate="1:1")
    return merged[merged.label.str.startswith("Direct") & (merged.n_substantive > 0)]


@pytest.fixture(scope="module")
def gold_pairs() -> pd.DataFrame:
    if not Path(AGREED).exists():
        pytest.skip(f"agreed-labels artifact missing: {AGREED}")
    g = pd.read_csv(AGREED)
    return g[(g.label_1 == g.label_2) & (g.label_1 == "Direct fix")]


def test_per_agent_pair_and_human_counts_judge_only(direct_fix_pairs):
    got = direct_fix_pairs.groupby("agent").q_is_human.agg(["size", "sum"])
    assert set(got.index) == set(JUDGED), f"agent set drifted: {list(got.index)}"
    for agent, (n, human) in JUDGED.items():
        assert (int(got.loc[agent, "size"]), int(got.loc[agent, "sum"])) == (n, human), \
            f"{agent}: {got.loc[agent].to_dict()} != n={n}, human={human}"


def test_overall_row_judge_only(direct_fix_pairs):
    n, human = JUDGED_TOTAL
    assert len(direct_fix_pairs) == n
    assert int(direct_fix_pairs.q_is_human.sum()) == human
    assert round(100 * human / n, 1) == 20.1


def test_gold_pair_counts_match_the_pinned_split(gold_pairs):
    # The gold half of the paper's frame: 14 consensus Direct pairs on 14
    # distinct merged PRs, and one pinned fixer split per merging agent.
    assert len(gold_pairs) == 14
    assert gold_pairs.p_url.nunique() == 14
    assert gold_pairs.agent.value_counts().to_dict() == \
        {agent: sum(split) for agent, split in GOLD_SPLIT.items()}


def _unrestricted_counts(direct_fix_pairs, gold_pairs) -> dict[str, tuple[int, int]]:
    """agent -> (pairs, human-fixed pairs) on the unrestricted frame, derived
    from the shipped judge artifacts plus the committed gold labels."""
    per_agent = direct_fix_pairs.groupby("agent").q_is_human.agg(["size", "sum"])
    gold_n = gold_pairs.agent.value_counts()
    return {agent: (int(per_agent.loc[agent, "size"]) + int(gold_n.get(agent, 0)),
                    int(per_agent.loc[agent, "sum"])
                    + GOLD_SPLIT.get(agent, (0, 0, 0))[1])
            for agent in per_agent.index}


def test_unrestricted_frame_reconciles_lake_free(direct_fix_pairs, gold_pairs):
    per_agent = direct_fix_pairs.groupby("agent").q_is_human.agg(["size", "sum"])
    gold_n = gold_pairs.agent.value_counts()
    assert set(gold_n.index) <= set(UNRESTRICTED), \
        f"gold agent outside the frame: {list(gold_n.index)}"
    for agent, (fixes, human_pct) in UNRESTRICTED.items():
        n = int(per_agent.loc[agent, "size"]) + int(gold_n.get(agent, 0))
        human = int(per_agent.loc[agent, "sum"]) + GOLD_SPLIT.get(agent, (0, 0, 0))[1]
        assert n == fixes, f"{agent}: judge+gold n={n} != pinned {fixes}"
        assert round(100 * human / n, 1) == human_pct, \
            f"{agent}: judge+gold human {100 * human / n:.1f}% != pinned {human_pct}%"

    n = len(direct_fix_pairs) + len(gold_pairs)
    human = (int(direct_fix_pairs.q_is_human.sum())
             + sum(split[1] for split in GOLD_SPLIT.values()))
    assert (n, round(100 * human / n, 1)) == UNRESTRICTED_OVERALL


def test_typeset_table_matches_the_pinned_frame(direct_fix_pairs, gold_pairs):
    """Table tab:rq3-fixer in results.tex is the contract: every cell it
    typesets comes from FRAME, and FRAME is bounded by the unrestricted frame
    derived here (restricting to uncensored merges only ever drops pairs)."""
    typeset = _typeset_rows("tab:rq3-fixer")
    unrestricted = _unrestricted_counts(direct_fix_pairs, gold_pairs)

    for agent, (n, self_, human, other) in FRAME.items():
        assert self_ + human + other == n, f"{agent}: {self_}+{human}+{other} != {n}"
        assert typeset[FRAME_LABEL[agent]] == [
            str(n), f"{100 * self_ / n:.1f}", f"{100 * human / n:.1f}",
            f"{100 * other / n:.1f}",
        ], f"{FRAME_LABEL[agent]} row of tab:rq3-fixer drifted from the pinned frame"
        u_n, u_human = unrestricted[agent]
        assert n <= u_n, f"{agent}: restricted n={n} exceeds unrestricted {u_n}"
        assert human <= u_human, \
            f"{agent}: restricted human={human} exceeds unrestricted {u_human}"
        assert other <= UNRESTRICTED_OTHER[agent], \
            f"{agent}: restricted other-agent={other} exceeds unrestricted " \
            f"{UNRESTRICTED_OTHER[agent]}"

    n = sum(v[0] for v in FRAME.values())
    totals = [sum(v[i] for v in FRAME.values()) for i in (1, 2, 3)]
    assert typeset["Overall (agent merges)"] == \
        [str(n)] + [f"{100 * c / n:.1f}" for c in totals]
    # The restricted frame is a strict subset: the unrestricted total is larger.
    assert n < UNRESTRICTED_OVERALL[0]

    h_n, h_human, h_agent = FRAME_HUMAN_ROW
    assert round(h_human + h_agent, 1) == 100.0
    assert typeset["Human"] == [str(h_n), "-", f"{h_human:.1f}", f"{h_agent:.1f}"]


# --- digest-backed re-derivation of the restricted frame --------------------


@pytest.fixture(scope="module")
def digest() -> pd.DataFrame:
    """The committed fixer-frame digest, ledger row 28's contract: one row per
    unrestricted pair, `in_frame` marking the fully observed merges."""
    if not Path(DIGEST_FIXER).exists():
        pytest.skip(f"fixer frame digest missing: {DIGEST_FIXER} "
                    "(written by a lake run of consolidation_gate.py)")
    return pd.read_csv(DIGEST_FIXER)


def test_digest_unrestricted_half_matches_the_lake_free_derivation(
        digest, direct_fix_pairs, gold_pairs):
    """The digest may only differ from the shipped artifacts by the frame flag:
    its full row set must be the judge Direct pairs plus the gold pairs, with
    the same per-agent pair and human-fixed counts derived here."""
    unrestricted = _unrestricted_counts(direct_fix_pairs, gold_pairs)
    assert len(digest) == len(direct_fix_pairs) + len(gold_pairs)
    assert set(digest.p_id) >= set(direct_fix_pairs.p_id)
    per_agent = per_agent_split(digest)
    for agent, (n, human) in unrestricted.items():
        assert int(per_agent.loc[agent, "n"]) == n, \
            f"{agent}: digest n={per_agent.loc[agent, 'n']} != judge+gold {n}"
        assert int(per_agent.loc[agent, "human"]) == human, \
            f"{agent}: digest human={per_agent.loc[agent, 'human']} != {human}"


def test_digest_rebuilds_the_typeset_restricted_frame(digest):
    """The restriction is a row filter on the digest, so the paper's per-agent
    split, its Overall row and the merge-level view of RQ2 all come back
    through per_agent_split() with pandas alone."""
    restricted = digest[digest.in_frame == 1]
    per_agent = per_agent_split(restricted)
    for agent, (n, self_, human, other) in FRAME.items():
        # The gate script guards the same way: an expected agent with no pair
        # in the frame is a drift, not a lookup error.
        assert agent in per_agent.index, \
            f"{agent} has no pair in the digest's frame, expected n={n}"
        row = per_agent.loc[agent]
        assert (int(row["n"]), int(row["self"]), int(row["human"]),
                int(row["other-agent"])) == (n, self_, human, other), \
            f"digest disagrees with the pinned frame for {agent}: {row.to_dict()}"
    assert len(restricted) == sum(v[0] for v in FRAME.values())

    # Merge level: the same population viewed per PR is RQ1's verified merges.
    pr_fix = restricted.groupby("p_id").fixer.agg(
        lambda x: "human-involved" if "human" in set(x)
        else ("self-only" if set(x) == {"self"} else "agent-mixed"))
    share = pr_fix.value_counts(normalize=True)
    assert len(pr_fix) == 203
    assert [round(100 * share[k], 1) for k in
            ("self-only", "human-involved", "agent-mixed")] == [62.6, 34.0, 3.4]


def test_write_fixer_digest_round_trips_both_frames(tmp_path, monkeypatch):
    """The writer is what a lake run commits, so it is exercised without one:
    read back with pandas alone the digest must give the same per-agent split
    as the in-memory frames, restricted and unrestricted."""
    da_all = add_fixer(PAIRS, AGENT_OF)
    frame_ids = {1}                       # PR 4 is censored, PR 3 out of frame
    monkeypatch.setattr(fixer, "DIGEST_FIXER", tmp_path / "rq2_fixer_frame.csv")
    monkeypatch.setattr(fixer, "DIGESTS_DIR", tmp_path)

    write_fixer_digest(da_all, frame_ids)
    d = pd.read_csv(fixer.DIGEST_FIXER)

    assert list(d.columns) == ["p_id", "q_id", "p_agent", "fixer", "in_frame"]
    assert len(d) == len(da_all)
    # .equals, not approx: the gate commits the digest only if it round-trips.
    assert per_agent_split(d).equals(per_agent_split(da_all))
    assert per_agent_split(d[d.in_frame == 1]).equals(
        per_agent_split(da_all[da_all.p_id.isin(frame_ids)]))
    # The flag is the only lake-bound column, and it is a strict filter.
    assert set(d[d.in_frame == 1].p_id) == frame_ids & set(da_all.p_id)


def test_fixer_digest_drops_no_pair_and_keeps_the_frame_flag_out_of_the_split():
    """A pair outside the frame stays in the digest (the unrestricted frame is
    the robustness variant) and only the flag distinguishes it."""
    d = fixer_digest(add_fixer(PAIRS, AGENT_OF), set())
    assert len(d) == len(PAIRS) and int(d.in_frame.sum()) == 0
    assert per_agent_split(d[d.in_frame == 1]).empty


# Fixer counts on the subset of real pairs whose fixing PR Q is itself a judged
# PR (so its merging agent is recoverable from the case file, no lake needed),
# and the "Other agent" counts Table tab:rq3-fixer typesets for the full frame.
RESOLVABLE = {
    "OpenAI_Codex": {"self": 35, "human": 48, "other-agent": 1, "n": 84},
    "Devin":        {"self": 40, "human": 17, "other-agent": 3, "n": 60},
    "Copilot":      {"self": 25, "human": 2,  "other-agent": 1, "n": 28},
    "Cursor":       {"self": 8,  "human": 0,  "other-agent": 0, "n": 8},
}
# Other-agent counts on the UNRESTRICTED judge + gold frame (Claude Code's is
# its gold pair, see GOLD_SPLIT). Both the resolvable subset below and the
# restricted frame the paper typesets are bounded by these.
UNRESTRICTED_OTHER = {"OpenAI_Codex": 3, "Devin": 3, "Copilot": 1, "Cursor": 1,
                      "Claude_Code": 1}


def test_per_agent_split_on_resolvable_real_pairs(direct_fix_pairs, cases):
    # p_id -> merging agent, projected off the case file (the same column the
    # gate script reads from the lake's pull_request table).
    agent_of = pd.Series(dict(zip(cases.p_id, cases.agent, strict=True)))
    labelled = add_fixer(direct_fix_pairs, agent_of)
    # p_agent must agree with the judge CSV's own agent column for every pair.
    assert (labelled.p_agent == labelled.agent).all()
    known = labelled[labelled.q_is_human | labelled.q_id.isin(agent_of.index)]
    got = per_agent_split(known)
    assert {a: {k: int(v) for k, v in got.loc[a].items()} for a in got.index} == RESOLVABLE
    # Every other-agent fix found here is one of the unrestricted frame's, so
    # the resolvable count can never exceed it.
    for agent, other in UNRESTRICTED_OTHER.items():
        assert int(got["other-agent"].get(agent, 0)) <= other


# --- self vs other-agent: synthetic frame, no lake needed --------------------
# PR id -> merging agent, the mapping the gate script reads off the PR table.
AGENT_OF = pd.Series({1: "Devin", 2: "Devin", 3: "Copilot", 4: "Devin", 5: "Copilot"})
PAIRS = pd.DataFrame([
    # P merged by Devin, fixed by Devin -> self
    {"p_id": 1, "q_id": 2, "q_is_human": False},
    # P merged by Devin, fixed by Copilot -> other-agent
    {"p_id": 1, "q_id": 3, "q_is_human": False},
    # P merged by Devin, fixed by a human (q_id still maps to an agent row) -> human
    {"p_id": 4, "q_id": 2, "q_is_human": True},
    # P merged by Copilot, fixed by Copilot -> self
    {"p_id": 3, "q_id": 5, "q_is_human": False},
])


def test_add_fixer_classifies_self_human_and_other_agent():
    got = add_fixer(PAIRS, AGENT_OF)
    assert list(got.fixer) == ["self", "other-agent", "human", "self"]
    assert list(got.p_agent) == ["Devin", "Devin", "Devin", "Copilot"]
    # q_is_human wins even when Q maps to an agent row (human-authored fix in a
    # repo whose PR row carries an agent label).
    assert got.loc[2, "q_agent"] == "Devin"


def test_per_agent_split_counts_and_missing_categories():
    t = per_agent_split(add_fixer(PAIRS, AGENT_OF))
    assert list(t.columns) == ["self", "human", "other-agent", "n"]
    assert t.loc["Devin"].to_dict() == {"self": 1, "human": 1, "other-agent": 1, "n": 3}
    # Copilot has no human and no other-agent fix: the columns are still present
    # and zero, which is what the gate script's row lookup indexes.
    assert t.loc["Copilot"].to_dict() == {"self": 1, "human": 0, "other-agent": 0, "n": 1}
    assert int(t.n.sum()) == len(PAIRS)


def test_per_agent_split_omits_agent_with_no_judged_pair():
    # The gate script's `ag not in per_agent.index` guard fires on exactly this:
    # an expected agent that contributed no judged direct-fix pair.
    t = per_agent_split(add_fixer(PAIRS[PAIRS.p_id != 3], AGENT_OF))
    assert "Copilot" not in t.index
    assert list(t.index) == ["Devin"]


# ---------------------------------------------------------------- typeset prose
# The RQ2 self-fix share is typeset in seven places besides Table
# tab:rq3-fixer, each hand-written. The section .tex files are the contract
# (same convention as _typeset_rows): a frame change has to move every site,
# so these tests bind the prose back to the table's Overall row.
SECTIONS = Path(__file__).resolve().parents[2] / "paper-emse" / "sections"
HEADLINE_SITES = ["abstract.tex", "introduction.tex", "discussion.tex",
                  "conclusion.tex", "related-work.tex", "results.tex"]


def _prose(name: str) -> str:
    """A section's text with the LaTeX noise around numbers normalised away."""
    text = read_tex(SECTIONS / name)
    text = "\n".join(ln for ln in text.splitlines() if not ln.startswith("%"))
    for old, new in (("{,}", ""), ("$", ""), ("\\,", ""), ("\\%", "%"),
                     ("\\emph{", ""), ("}", ""), ("---", " - ")):
        text = text.replace(old, new)
    return " ".join(text.split())


def _headline_shares(name: str) -> list[float]:
    text = _prose(name)
    return [float(m) for m in
            re.findall(r"([\d.]+)\s*% of the verified fixes on agent merges "
                       r"come from the same agent", text)
            + re.findall(r"verified fixes on agent merges, ([\d.]+)\s*% come "
                         r"from the same agent", text)]


def test_headline_self_fix_share_agrees_across_every_typeset_site():
    overall = _typeset_rows("tab:rq3-fixer")["Overall (agent merges)"]
    for name in HEADLINE_SITES:
        quoted = _headline_shares(name)
        assert quoted, f"{name} no longer typesets the RQ2 self-fix share"
        assert quoted == [float(overall[1])] * len(quoted), \
            f"{name} quotes {quoted}, tab:rq3-fixer's Overall row says {overall[1]}"


def test_rq2_prose_numbers_follow_the_typeset_frame():
    """The RQ2 paragraphs quote four things derived from the same frame: the
    pair-level triple, the merge-level triple, the excl-Codex share, and
    Codex's human-fixed share. Each has to follow the table it sits under."""
    typeset = _typeset_rows("tab:rq3-fixer")
    text = _prose("results.tex")
    n, self_, human, other = typeset["Overall (agent merges)"]

    assert (f"{self_}% are self-fixes" in text
            and f"against {human}% fixed by a human and {other}%" in text
            and f"(n={n} verified fixes)" in text), \
        "the RQ2 pair-level sentence drifted from the Overall row"

    # Merge level: the same population viewed per PR, so it is RQ1's verified
    # agent merges -- the count RQ1 already typesets -- and never more PRs
    # than there are pairs.
    rq1 = re.search(r"of agent merges \((\d+) of \d+; \d+ judge-verified", text)
    merges = re.search(r"of the (\d+) verified-fixed merges of RQ1, ([\d.]+)% "
                       r"were fixed entirely by the merging agent, ([\d.]+)% "
                       r"had a human involved, and ([\d.]+)%", text)
    assert rq1 and merges, "RQ2 no longer states the merge-level view of RQ1's merges"
    assert merges.group(1) == rq1.group(1), \
        f"merge-level n={merges.group(1)} is not RQ1's {rq1.group(1)} verified merges"
    assert int(merges.group(1)) <= int(n), "more fixed merges than verified pairs"
    assert round(sum(float(g) for g in merges.groups()[1:]), 1) == 100.0

    noncodex = {a: v for a, v in FRAME.items() if a != "OpenAI_Codex"}
    nc_n = sum(v[0] for v in noncodex.values())
    nc_self = sum(v[1] for v in noncodex.values())
    assert f"rises to {100 * nc_self / nc_n:.1f}% (n={nc_n} verified fixes)" in text, \
        "the excl-Codex sentence drifted from the table's non-Codex rows"
    codex_human = typeset[FRAME_LABEL["OpenAI_Codex"]][2]
    assert f"humans fix most ({codex_human}%)" in text, \
        "the Codex human-fixed share in prose drifted from its table row"


def test_unrestricted_sensitivity_is_quoted_consistently_on_both_sides():
    """The inclusion-rule sensitivity quotes the unrestricted frame in results
    and again in threats; both must name the same agent-side value, its n must
    be the unrestricted total this file derives, and it must exceed the
    restricted share the table typesets (restricting drops the fast fixes)."""
    restricted = float(_typeset_rows("tab:rq3-fixer")["Overall (agent merges)"][1])
    results, threats = _prose("results.tex"), _prose("threats.tex")

    m = re.search(r"agent self-fix share is ([\d.]+)% \(n=(\d+)\)", results)
    assert m, "results no longer quotes the unrestricted agent self-fix share"
    assert int(m.group(2)) == UNRESTRICTED_OVERALL[0]
    assert float(m.group(1)) > restricted
    assert f"agent self-fix share is {m.group(1)}%" in threats, \
        "threats quotes a different unrestricted share than results"
