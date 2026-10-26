"""Metrics helpers.

consolidation_gate.py pins the published numbers but needs the full DuckDB
lake (gitignored), so it cannot gate anything on a fresh checkout. These tests
exercise the parts that are lake-independent on a synthetic in-memory lake:
the fix-PR commit filter (which must return commits of the verified fix PRs
only, per side), content-only classification, the per-PR aggregation, and the
Kaplan-Meier cumulative incidence.
"""

from __future__ import annotations

import duckdb
import numpy as np
import pandas as pd
import pytest

from prfix.common import metrics


class _Keep:
    """DuckDB connection whose close() is a no-op (the helpers own closing)."""

    def __init__(self, con):
        self._con = con

    def __getattr__(self, name):
        return getattr(self._con, name)

    def close(self):
        pass


def _lake(monkeypatch) -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    con.execute("CREATE TABLE pr_commits"
                "(pr_id BIGINT, author VARCHAR, committer VARCHAR, message VARCHAR)")
    con.execute("INSERT INTO pr_commits VALUES "
                "(1,'devin-ai-integration[bot]','x','a'),(1,'alice','x','b'),"
                "(9,'alice','x','not a verified fix')")
    con.execute("CREATE TABLE human_pr_commits"
                "(pr_id BIGINT, author VARCHAR, committer VARCHAR, message VARCHAR)")
    con.execute("INSERT INTO human_pr_commits VALUES "
                "(1,'bob','x','c'),(2,'bob','x','d'),(7,'bob','x','not a verified fix')")
    monkeypatch.setattr(metrics, "connect", lambda read_only=True: _Keep(con))
    return con


def _survival_lake(monkeypatch, human_merged: str | None = "2025-01-10") -> None:
    """Minimal lake for survival_frame: one >=500-star repo, one agent P with
    two candidate fix Qs -- only one of which the net_sub linker can carry."""
    con = duckdb.connect()
    con.execute("CREATE TABLE repository(id BIGINT, url VARCHAR, stars BIGINT)")
    con.execute("INSERT INTO repository VALUES (1,'https://github.com/o/r',900)")
    con.execute("CREATE TABLE pull_request"
                "(id BIGINT, repo_id BIGINT, agent VARCHAR, merged_at VARCHAR,"
                " title VARCHAR, html_url VARCHAR)")
    con.execute("INSERT INTO pull_request VALUES "
                "(10,1,'Devin','2025-01-01','feat p','u10'),"
                "(11,1,'Devin','2025-01-05','fix linked','u11'),"
                "(12,1,'Devin','2025-01-06','fix unlinked','u12')")
    con.execute("CREATE TABLE human_pull_request"
                "(id BIGINT, repo_url VARCHAR, merged_at VARCHAR,"
                " title VARCHAR, html_url VARCHAR)")
    merged = "NULL" if human_merged is None else repr(human_merged)
    con.execute("INSERT INTO human_pull_request VALUES "
                f"(20,'https://github.com/o/r',{merged},"
                "'human pr','u20')")
    con.execute("CREATE TABLE pr_task_type(id BIGINT, type VARCHAR)")
    con.execute("INSERT INTO pr_task_type VALUES (11,'fix'),(12,'fix')")
    con.execute("CREATE TABLE human_pr_task_type(id BIGINT, type VARCHAR)")
    # P and Q=11 co-edit a substantive file; Q=12 only touches boilerplate, so
    # the net_sub linker never emits the (10, 12) pair.
    con.execute("CREATE TABLE pr_net_files"
                "(pr_id BIGINT, filename VARCHAR, is_boilerplate BOOLEAN)")
    con.execute("INSERT INTO pr_net_files VALUES "
                "(10,'a.py',false),(11,'a.py',false),"
                "(12,'a.py',true),(12,'pnpm-lock.yaml',true)")
    monkeypatch.setattr(metrics, "connect", lambda read_only=True: _Keep(con))


def test_survival_frame_accounts_for_verified_pairs_the_linker_drops(monkeypatch):
    """A verified pair the net_sub linker does not carry produces no event, so
    the frame must report it as dropped rather than lose it silently."""
    _survival_lake(monkeypatch)

    df = metrics.survival_frame("agent", {(10, 11), (10, 12)})

    assert (df.attrs["verified_pairs"], df.attrs["verified_pairs_linked"],
            df.attrs["verified_pairs_unlinked"]) == (2, 1, 1)
    p = df[df.pr_id == 10].iloc[0]
    # Only the linked pair (day 4) becomes the verified event; the dropped
    # pair at day 5 never reaches the frame.
    assert (p.ver_event, p.ver_time) == (1, 4)


def test_survival_frame_rejects_a_null_shared_cutoff(monkeypatch):
    """With no parseable merge on one side the shared cut-off is NULL; the
    frame must refuse rather than censor every PR against NaT."""
    _survival_lake(monkeypatch, human_merged=None)

    with pytest.raises(ValueError, match="shared cut-off date is NULL"):
        metrics.survival_frame("agent", {(10, 11)})


def test_fix_pr_commits_keep_only_verified_fix_prs_per_side(monkeypatch):
    _lake(monkeypatch)
    q = pd.DataFrame({"q_id": [1, 1, 2],
                      "q_is_human": [False, True, True],
                      "q_agent": ["Devin", "Human", "Human"]})

    out = metrics.fix_pr_commit_classes(q)

    # pr 9 (agent table) and pr 7 (human table) are not verified fixes; pr_id 1
    # exists on both sides and must be attributed per side, not merged.
    assert sorted(zip(out.grp, out.pr_id, strict=True)) == [
        ("Devin", 1), ("Devin", 1), ("Human", 1), ("Human", 2)]
    assert out[out.grp == "Devin"].cls.tolist() == ["agent", "human"]


def test_classification_is_content_only(monkeypatch):
    """The opener login must not leak in as a prior: an opener-authored commit
    with no agent marker stays `human` (the Codex prior is circular
    when the question is whether opener identity tracks authorship)."""
    _lake(monkeypatch)
    q = pd.DataFrame({"q_id": [1], "q_is_human": [False],
                      "q_agent": ["OpenAI_Codex"]})

    out = metrics.fix_pr_commit_classes(q)

    assert out.cls.tolist() == ["agent", "human"]


def test_per_pr_metrics_averages_per_pr_not_pooled():
    c = pd.DataFrame({
        "grp": ["Devin"] * 3 + ["Human"] * 2,
        "pr_id": [1, 1, 2, 3, 3],
        "cls": ["agent", "human", "agent", "human", "human"],
    })

    m = metrics.per_pr_metrics(c)

    # Devin: PR 1 is 50 % agent, PR 2 is 100 % -> 75 % per-PR vs 66.7 % pooled.
    assert m.loc["Devin", "share_per_pr"] == 75.0
    assert m.loc["Devin", "pooled_agent"] == 66.7
    assert (m.loc["Devin", "pct_any"], m.loc["Devin", "pct_all"]) == (100.0, 50.0)
    assert (m.loc["Devin", "n_prs"], m.loc["Devin", "med_commits"]) == (2, 1.5)
    assert m.loc["Human", "pct_any"] == 0.0


def test_digests_reproduce_the_estimates_without_the_lake(tmp_path, monkeypatch):
    """The committed digests are the reviewer's only route to rows 25-27: read
    back with pandas alone they must give the same Cox HR, cumulative incidence
    and per-PR table as the in-memory lake frames."""
    rng = np.arange(12)
    sv = pd.DataFrame({
        "pr_id": rng, "repo_id": np.where(rng < 6, 1, 2),
        "cohort": np.where(rng % 2 == 0, "Agent", "Human"),
        "merged_at": pd.Timestamp("2025-01-01"),  # lake-only column, dropped
        "cand_event": [1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0],
        "cand_time": [2.0, 30.0, 5.0, 9.0, 30.0, 3.0,
                      4.0, 12.0, 30.0, 7.0, 6.0, 30.0],
        "ver_event": [1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0],
        "ver_time": [2.0, 30.0, 30.0, 9.0, 30.0, 30.0,
                     4.0, 30.0, 30.0, 7.0, 30.0, 30.0],
    })
    commits = pd.DataFrame({
        "side": ["all"] * 4 + ["fix"] * 3,
        "grp": ["Devin", "Devin", "Human", "Human", "Devin", "Devin", "Human"],
        "pr_id": [1, 1, 2, 3, 1, 2, 3],
        "cls": ["agent", "human", "human", "human", "agent", "agent", "human"],
    })
    monkeypatch.setattr(metrics, "DIGESTS_DIR", tmp_path)
    monkeypatch.setattr(metrics, "DIGEST_SURVIVAL", tmp_path / "rq1_survival_frame.csv")
    monkeypatch.setattr(metrics, "DIGEST_COMMITS", tmp_path / "rq3_commit_classes.csv")

    metrics.write_digests(sv, commits)
    d_sv = pd.read_csv(metrics.DIGEST_SURVIVAL)
    d_c = pd.read_csv(metrics.DIGEST_COMMITS)

    assert metrics.cox_hr(d_sv, "ver") == pytest.approx(metrics.cox_hr(sv, "ver"))
    assert metrics.cox_hr(d_sv, "cand") == pytest.approx(metrics.cox_hr(sv, "cand"))
    assert np.allclose(
        metrics.km_cuminc(d_sv.cand_time.to_numpy(), d_sv.cand_event.to_numpy())[1],
        metrics.km_cuminc(sv.cand_time.to_numpy(), sv.cand_event.to_numpy())[1])
    for side in ("all", "fix"):
        # .equals, not approx: the gate commits the digest only if it round-trips
        # to the published tables cell-for-cell.
        assert metrics.per_pr_metrics(d_c[d_c.side == side]).equals(
            metrics.per_pr_metrics(commits[commits.side == side]))


def test_crosscohort_digest_rebuilds_the_mh_strata(tmp_path, monkeypatch):
    """The strata digest is rows 3/6/32's only lake-free input: expanded back
    through the real repo_tables() it must give the same 2x2 tables and MH
    estimate as the per-merge shared-window frames, and repositories where one
    cohort is absent must contribute no stratum."""
    agent = pd.DataFrame({
        "pr_id": range(6), "repo_id": [1, 1, 1, 2, 2, 3],
        "has_followup_30d": [1, 0, 1, 1, 0, 1],
        "verified": [1, 0, 0, 0, 0, 1]})
    human = pd.DataFrame({
        "pr_id": range(10, 16), "repo_id": [1, 1, 1, 2, 2, 4],
        "has_followup_30d": [0, 0, 1, 1, 0, 1],
        "verified": [0, 0, 1, 0, 0, 1]})
    monkeypatch.setattr(metrics, "DIGESTS_DIR", tmp_path)
    monkeypatch.setattr(metrics, "DIGEST_CROSSCOHORT",
                        tmp_path / "rq1_crosscohort_strata.csv")

    metrics.write_crosscohort_digest(agent, human)
    d = pd.read_csv(metrics.DIGEST_CROSSCOHORT)

    for col, flag in (("n_candidate", "has_followup_30d"), ("n_verified", "verified")):
        want = metrics.repo_tables(agent.assign(y=agent[flag]),
                              human.assign(y=human[flag]), "y")
        got = metrics.repo_tables(*metrics.crosscohort_frames(d, col), "y")
        assert sorted(t.tolist() for t in got) == sorted(t.tolist() for t in want)
        assert metrics.mh(got) == pytest.approx(metrics.mh(want))

    a, h = metrics.crosscohort_frames(d, "n_candidate")
    # Repos 3 (agent only) and 4 (human only) drop out; min_n=2, the default
    # since the sparse-strata heuristic was retired, keeps both shared repos,
    # while the retired min_n=6 would discard repo 2's 2+2 merges.
    assert len(metrics.repo_tables(a, h, "y")) == 2
    assert len(metrics.repo_tables(a, h, "y", min_n=6)) == 1


def test_km_cuminc_ignores_censored_and_holds_to_window():
    xs, ys = metrics.km_cuminc(np.array([1.0, 2.0, 3.0]), np.array([1, 0, 1]))

    assert xs.tolist() == [0.0, 1.0, 3.0, metrics.WINDOW]
    # 3 at risk at t=1 (1 event), 1 at risk at t=3 (1 event) -> S drops to 0.
    assert np.allclose(ys, [0.0, 1 / 3, 1.0, 1.0])


def test_gold_direct_resolution_assigns_each_fix_to_its_own_cohort(monkeypatch):
    """The 14 gold pairs enter the RQ2 fixer frame through this helper, so the
    cohort of the FIXING PR decides whether a gold fix reads human or agent.
    A Q in the agent table must resolve to that table's id with
    q_is_human False, one only in the human table to human, and the URL
    normalisation (trailing slash, case) must not change the answer."""
    monkeypatch.setattr(metrics, "_gold_direct", lambda: pd.DataFrame({
        "p_url": ["https://github.com/o/r/pull/1",
                  "https://github.com/o/r/pull/2",
                  "https://github.com/o/r/pull/3"],
        "q_url": ["https://github.com/o/r/pull/11",
                  "https://GitHub.com/o/r/pull/22/",   # human side, denormalised
                  "https://github.com/o/r/pull/33"],
        "agent": ["Devin", "Claude_Code", "Cursor"]}))
    agent = pd.DataFrame({"id": [11, 33],
                          "html_url": ["https://github.com/o/r/pull/11",
                                       "https://github.com/o/r/pull/33"]})
    human = pd.DataFrame({"id": [22], "html_url": ["https://github.com/o/r/pull/22"]})

    g = metrics.gold_direct_resolved(agent, human)

    assert list(zip(g.q_id, g.q_is_human, strict=True)) == [
        (11, False), (22, True), (33, False)]
    # p_url/agent (the MERGING agent) survive: the fixer split groups on them.
    assert g.agent.tolist() == ["Devin", "Claude_Code", "Cursor"]


def test_gold_direct_resolution_prefers_the_agent_table_and_refuses_an_unknown_q(
        monkeypatch):
    """The two cohort tables have independent id spaces, so a shared URL must
    resolve agent-side (not silently pick the human id), and a Q in neither
    table must raise rather than be dropped or default to human."""
    gold = pd.DataFrame({"p_url": ["p"], "q_url": ["https://github.com/o/r/pull/11"],
                         "agent": ["Devin"]})
    monkeypatch.setattr(metrics, "_gold_direct", lambda: gold)
    both = pd.DataFrame({"id": [11], "html_url": ["https://github.com/o/r/pull/11"]})
    collide = pd.DataFrame({"id": [999], "html_url": ["https://github.com/o/r/pull/11"]})

    assert metrics.gold_direct_resolved(both, collide).q_id.tolist() == [11]

    empty = pd.DataFrame({"id": [], "html_url": []})
    with pytest.raises(KeyError, match="gold q_url not in either PR table"):
        metrics.gold_direct_resolved(empty, empty)
