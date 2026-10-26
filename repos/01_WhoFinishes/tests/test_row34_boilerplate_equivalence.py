"""Ledger row 34's candidate-frame equivalence, on a synthetic lake.

The shipped ``pr_net_files.is_boilerplate`` column was precomputed with an
earlier ``is_boilerplate`` whose ``lstrip("./")`` ate every leading "." of a
path, so the tool-config directory prefixes never fired at repository root.
Row 34 stands in for the corrected predicate by subtracting the reclassified
filenames from the stored column (``linker.net_sub_excluding``) and
asserting the three candidate frames do not move.

Those real counts (49 filenames / 167 file-rows; 4,505/1,031, 2,012/421,
4,063/596) need the gitignored AIDev lake and stay pinned in the gate. What is
checkable here is the mechanism they rest on: these tests build a tiny lake
whose stored column carries the OLD classification and drive the same
substitution through the real ``rb.compute``, both when the reclassified files
are never a pair's only link (frame unchanged, as on the real data) and when one
is (frame moves, so the gate's assertion has teeth rather than passing
vacuously).
"""

from __future__ import annotations

import duckdb
import pytest

from prfix.rq1_fix_rates import linker as rb
from prfix.rq1_fix_rates.boilerplate import is_boilerplate

# Stored as EVIDENTIAL (the old rule reduced them to `vscode/...`), boilerplate
# under the corrected one. The quoted name also drives the SQL escaping.
RECLASSIFIED = [".vscode/settings.json", ".vscode/a'b.json"]
CO_EDIT = "src/a.py"


def _lake(tmp_path, monkeypatch, sole_link: bool):
    """One-repo lake: agent P 100 co-edits `CO_EDIT` with the fix Q 900, and
    (when `sole_link`) P 101 shares only a reclassified config file with it."""
    path = tmp_path / "lake.duckdb"
    con = duckdb.connect(str(path))
    con.execute("CREATE TABLE repository(id INT, url VARCHAR, stars INT)")
    con.execute("INSERT INTO repository VALUES (1, 'https://github.com/o/r', 1000)")
    con.execute("CREATE TABLE pull_request"
                "(id INT, repo_id INT, agent VARCHAR, merged_at VARCHAR, title VARCHAR)")
    con.executemany("INSERT INTO pull_request VALUES (?, 1, 'Devin', ?, ?)",
                    [(100, "2025-01-05", "feat: a"),
                     (101, "2025-01-05", "feat: b"),
                     (199, "2025-06-01", "feat: z"),      # pushes the own cut-off out
                     (900, "2025-01-08", "fix: y")])
    con.execute("CREATE TABLE pr_task_type(id INT, type VARCHAR)")
    con.execute("INSERT INTO pr_task_type VALUES (900, 'fix')")
    con.execute("CREATE TABLE human_pull_request"
                "(id INT, repo_url VARCHAR, merged_at VARCHAR, title VARCHAR)")
    con.execute("INSERT INTO human_pull_request "
                "VALUES (200, 'https://github.com/o/r', '2025-06-02', 'feat: h')")
    con.execute("CREATE TABLE human_pr_task_type(id INT, type VARCHAR)")
    con.execute("CREATE TABLE pr_net_files"
                "(pr_id INT, filename VARCHAR, is_boilerplate BOOL)")
    rows = [(900, CO_EDIT), (100, CO_EDIT)]
    rows += [(900, f) for f in RECLASSIFIED] + [(100, f) for f in RECLASSIFIED]
    if sole_link:
        rows.append((101, RECLASSIFIED[0]))
    # Every filename is stored EVIDENTIAL, exactly as the old rule left it.
    con.executemany("INSERT INTO pr_net_files VALUES (?, ?, FALSE)", rows)
    con.close()
    monkeypatch.setattr(rb, "connect", lambda read_only=False:
                        duckdb.connect(str(path), read_only=read_only))
    return path


def _reclassified_from(path) -> list[str]:
    """The gate's own derivation: stored column + the CURRENT rule."""
    con = duckdb.connect(str(path), read_only=True)
    stored = dict(con.execute("SELECT filename, any_value(is_boilerplate) "
                              "FROM pr_net_files GROUP BY 1").fetchall())
    con.close()
    assert not [f for f, b in stored.items() if b and not is_boilerplate(f)], \
        "subtracting filenames only stands in for the corrected rule while it " \
        "is a superset of the stored one"
    return [f for f, b in stored.items() if is_boilerplate(f) and not b]


def _frame(linker: str):
    f = rb.compute(linker=linker)
    f = f[f.censored_30d == 0]
    per_pr = dict(zip(f.pr_id, f.has_followup_30d, strict=True))
    return len(f), int(f.has_followup_30d.sum()), per_pr


def test_frame_is_unchanged_when_reclassified_files_are_never_the_only_link(
        tmp_path, monkeypatch):
    """Row 34's claim: the reclassified files ride along with a substantive
    co-edit, so dropping them leaves the candidate frame identical."""
    path = _lake(tmp_path, monkeypatch, sole_link=False)
    assert sorted(_reclassified_from(path)) == sorted(RECLASSIFIED)
    rb.FILES_SOURCE["net_fixed"] = rb.net_sub_excluding(_reclassified_from(path))

    n, followups, per_pr = _frame("net_sub")
    assert (n, followups) == _frame("net_fixed")[:2]
    assert per_pr == _frame("net_fixed")[2]
    assert per_pr[100] == 1, "the co-edit link must survive, or this is vacuous"


def test_frame_moves_when_a_reclassified_file_is_a_pairs_only_link(
        tmp_path, monkeypatch):
    """The same substitution on a lake where one pair is linked ONLY by a
    reclassified config file: that pair loses its follow-up, which is the
    failure row 34 would report."""
    path = _lake(tmp_path, monkeypatch, sole_link=True)
    rb.FILES_SOURCE["net_fixed"] = rb.net_sub_excluding(_reclassified_from(path))

    before = _frame("net_sub")[2]
    after = _frame("net_fixed")[2]
    assert before[101] == 1 and after[101] == 0
    assert before[100] == 1 and after[100] == 1


def test_empty_reclassified_list_falls_back_to_the_stored_column(
        tmp_path, monkeypatch):
    """Once the column is rebuilt under the corrected rule nothing reclassifies;
    the substitution must then be the identity, not invalid SQL."""
    _lake(tmp_path, monkeypatch, sole_link=True)
    rb.FILES_SOURCE["net_fixed"] = rb.net_sub_excluding([])
    assert _frame("net_fixed") == _frame("net_sub")


@pytest.fixture(autouse=True)
def _drop_net_fixed():
    yield
    rb.FILES_SOURCE.pop("net_fixed", None)
