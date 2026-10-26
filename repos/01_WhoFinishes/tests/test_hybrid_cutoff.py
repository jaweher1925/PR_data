"""Hybrid censoring cut-off in linker.compute().

Cross-cohort comparisons must run on the SHARED cut-off (the earlier of the two
PR tables' last merge) so agent and human PRs observe follow-ups over the same
calendar period; single-cohort analyses keep their own cut-off. These tests build
a tiny synthetic lake -- the agent table deliberately runs three months past the
human table -- and check both frames, rather than needing the real AIDev lake.
"""

from __future__ import annotations

import duckdb
import pytest

from prfix.rq1_fix_rates import linker as rb

AGENT_MERGES = ["2025-01-05", "2025-02-10", "2025-04-01", "2025-06-01"]
HUMAN_MERGES = ["2025-01-06", "2025-02-11", "2025-03-01"]
SHARED_CUT = "2025-03-01"  # = human table's last merge (the earlier of the two)


@pytest.fixture()
def lake(tmp_path, monkeypatch):
    """Synthetic one-repo lake with the tables the follow-up SQL touches."""
    path = tmp_path / "lake.duckdb"
    con = duckdb.connect(str(path))
    con.execute("CREATE TABLE repository(id INT, url VARCHAR, stars INT)")
    con.execute("INSERT INTO repository VALUES (1, 'https://github.com/o/r', 1000)")
    con.execute("CREATE TABLE pull_request"
                "(id INT, repo_id INT, agent VARCHAR, merged_at VARCHAR, title VARCHAR)")
    con.executemany("INSERT INTO pull_request VALUES (?, 1, 'Devin', ?, 'feat: x')",
                    [(100 + i, m) for i, m in enumerate(AGENT_MERGES)])
    # One agent fix Q, merged 3 days after the first agent P and after the first human P.
    con.execute("INSERT INTO pull_request VALUES (900, 1, 'Devin', '2025-01-08', 'fix: y')")
    con.execute("CREATE TABLE pr_task_type(id INT, type VARCHAR)")
    con.execute("INSERT INTO pr_task_type VALUES (900, 'fix')")
    con.execute("CREATE TABLE human_pull_request"
                "(id INT, repo_url VARCHAR, merged_at VARCHAR, title VARCHAR)")
    con.executemany("INSERT INTO human_pull_request "
                    "VALUES (?, 'https://github.com/o/r', ?, 'feat: h')",
                    [(200 + i, m) for i, m in enumerate(HUMAN_MERGES)])
    con.execute("CREATE TABLE human_pr_task_type(id INT, type VARCHAR)")
    con.execute("CREATE TABLE pr_net_files(pr_id INT, filename VARCHAR, is_boilerplate BOOL)")
    con.executemany("INSERT INTO pr_net_files VALUES (?, 'src/a.py', FALSE)",
                    [(100,), (200,), (900,)])
    con.close()
    monkeypatch.setattr(rb, "connect",
                        lambda read_only=False: duckdb.connect(str(path), read_only=read_only))
    return path


def test_agent_frame_narrows_under_shared_cutoff(lake):
    """Agent PRs merged after the human table ends are censored only under 'shared'."""
    own = rb.compute().set_index("pr_id")
    shared = rb.compute(cutoff="shared").set_index("pr_id")

    # own cut-off = the agent table's own last merge (2025-06-01): only the PR
    # merged on the last day sits inside the 30-day censoring band.
    assert dict(own.censored_30d) == {100: 0, 101: 0, 102: 0, 103: 1, 900: 0}
    # shared cut-off = 2025-03-01, so everything merged after 2025-01-30 is censored.
    assert dict(shared.censored_30d) == {100: 0, 101: 1, 102: 1, 103: 1, 900: 0}
    assert int((own.censored_30d == 0).sum()) == 4
    assert int((shared.censored_30d == 0).sum()) == 2

    # The follow-up flags themselves are cut-off independent -- only the frame moves.
    assert own.loc[100, "has_followup_30d"] == 1
    assert shared.loc[100, "has_followup_30d"] == 1


def test_human_frame_identical_when_human_table_ends_first(lake):
    """The shared cut-off IS the human table's own end, so its frame is unchanged."""
    own = rb.compute(p_side="human").set_index("pr_id")
    shared = rb.compute(p_side="human", cutoff="shared").set_index("pr_id")

    assert list(own.index) == list(shared.index)
    assert list(own.censored_30d) == list(shared.censored_30d)
    assert set(own.agent) == {"Human"}
    # Human P #200 is repaired by agent fix #900 -- the Q pool spans both cohorts.
    assert own.loc[200, "has_followup_30d"] == 1


def test_shared_cutoff_is_the_earlier_table_end(lake):
    """The cut-off date itself, straight from SQL, is the human table's last merge."""
    con = duckdb.connect(str(lake), read_only=True)
    got = con.execute(rb.SQL_SHARED_CUTOFF).fetchone()[0]
    con.close()
    assert str(got)[:10] == SHARED_CUT


def test_bad_cutoff_rejected(lake):
    with pytest.raises(ValueError, match="unknown cutoff"):
        rb.compute(cutoff="global")


def test_shared_cutoff_null_is_a_hard_error(tmp_path, monkeypatch):
    """An unloaded human table makes the shared cut-off NULL: refuse, never silently
    fall back to the agent table's own end (that is the mismatched-window bug)."""
    path = tmp_path / "empty.duckdb"
    con = duckdb.connect(str(path))
    con.execute("CREATE TABLE repository(id INT, url VARCHAR, stars INT)")
    con.execute("INSERT INTO repository VALUES (1, 'https://github.com/o/r', 1000)")
    con.execute("CREATE TABLE pull_request"
                "(id INT, repo_id INT, agent VARCHAR, merged_at VARCHAR, title VARCHAR)")
    con.execute("INSERT INTO pull_request VALUES (100, 1, 'Devin', '2025-01-05', 'feat: x')")
    con.execute("CREATE TABLE pr_task_type(id INT, type VARCHAR)")
    con.execute("CREATE TABLE human_pull_request"
                "(id INT, repo_url VARCHAR, merged_at VARCHAR, title VARCHAR)")
    con.execute("CREATE TABLE human_pr_task_type(id INT, type VARCHAR)")
    con.execute("CREATE TABLE pr_net_files(pr_id INT, filename VARCHAR, is_boilerplate BOOL)")
    con.close()
    monkeypatch.setattr(rb, "connect",
                        lambda read_only=False: duckdb.connect(str(path), read_only=read_only))

    with pytest.raises(ValueError, match="shared cut-off date is NULL"):
        rb.compute(cutoff="shared")
