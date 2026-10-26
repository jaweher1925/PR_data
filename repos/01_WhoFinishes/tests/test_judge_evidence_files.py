"""The ``evidence_files`` column of the judge verdict CSV.

The verdict CSV (``data/judge/rq1_llm_judge_*.csv``) is a shipped artifact:
``evidence_files`` is a ``;``-joined list of repository paths, and
``gen_labeler.py`` splits it back on ``;`` to render the evidence line. The
judge's forced tool call declares the field as an array but the model
sometimes answers with a bare string, so consolidation must coerce before it
joins -- otherwise a single path is scattered one character per list element.
"""

from __future__ import annotations

import pandas as pd
import pytest

from prfix.rq1_fix_rates.judge import run_judge

CASES = [{"pair_id": 1, "agent": "Devin", "q_is_human": False, "n_shared": 1,
          "context_complete": True, "p_url": "https://x/p", "q_url": "https://x/q"}]


def _consolidate(tmp_path, monkeypatch, evidence):
    monkeypatch.setattr(run_judge, "_csv_path",
                        lambda *a, **k: tmp_path / "verdicts.csv")
    preds = {1: {"label": "Direct fix", "confidence": 0.9, "rationale": "",
                 "evidence_files": evidence}}
    df = run_judge._consolidate(CASES, preds, "anthropic", "m", "pop")
    written = pd.read_csv(tmp_path / "verdicts.csv", dtype=str, keep_default_na=False)
    assert written["evidence_files"].tolist() == df["evidence_files"].tolist()
    return written["evidence_files"].iloc[0]


def test_bare_string_stays_one_path(tmp_path, monkeypatch):
    """A bare-string answer is one path, not one element per character."""
    assert _consolidate(tmp_path, monkeypatch, "a/b.py") == "a/b.py"


def test_list_is_joined_on_semicolons(tmp_path, monkeypatch):
    assert _consolidate(tmp_path, monkeypatch, ["a/b.py", "c/d.py"]) == "a/b.py;c/d.py"


@pytest.mark.parametrize("empty", [None, [], ""])
def test_empty_answers_give_an_empty_cell(tmp_path, monkeypatch, empty):
    assert _consolidate(tmp_path, monkeypatch, empty) == ""


@pytest.mark.parametrize("odd", [{"files": ["a/b.py"]}, 7])
def test_unexpected_shapes_do_not_leak_keys_or_raise(tmp_path, monkeypatch, odd):
    """A dict must not become its keys, and a number must not abort the run."""
    cell = _consolidate(tmp_path, monkeypatch, odd)
    assert cell != "files"
    assert ";" not in cell


def test_round_trips_through_the_labeler_split(tmp_path, monkeypatch):
    """gen_labeler.py rebuilds the evidence list by splitting the cell on ';'."""
    paths = ["apps/web/preload/dist/index.js", "pyproject.toml"]
    cell = _consolidate(tmp_path, monkeypatch, paths)
    assert [e for e in cell.split(";") if e] == paths
