"""Pure-logic tests for blame-based attribution (no network)."""
from types import SimpleNamespace

import prfix.rq1_fix_rates.judge.attribution as ba
from prfix.rq1_fix_rates.judge.attribution import (
    _is_rate_limited,
    attribute_removed,
    gh_json,
    parse_pr_url,
    removed_oldlines,
)

# A unified diff: removes old lines 10 and 11, keeps 12 as context, adds two lines.
_PATCH = (
    "@@ -10,3 +10,4 @@ def f():\n"
    "-    old_a = 1\n"
    "-    old_b = 2\n"
    "     keep = 3\n"
    "+    new_a = 1\n"
    "+    new_b = 2\n"
)


def test_removed_oldlines_basic():
    assert removed_oldlines(_PATCH) == [10, 11]


def test_removed_oldlines_multi_hunk():
    patch = (
        "@@ -1,2 +1,1 @@\n"
        "-a\n"
        " b\n"
        "@@ -50,3 +49,2 @@\n"
        " c\n"
        "-d\n"
        " e\n"
    )
    # first hunk removes line 1; second hunk: context 50, remove 51, context 52
    assert removed_oldlines(patch) == [1, 51]


def test_removed_oldlines_empty():
    assert removed_oldlines("") == []
    assert removed_oldlines(None) == []
    assert removed_oldlines("+only additions\n@@ -1,0 +1,1 @@\n+x\n") == []


def test_attribute_no_removed():
    r = attribute_removed([], [[1, 5, "sha"]], {"sha"})
    assert r["status"] == "no_removed"
    assert r["n_blame_to_p"] == 0


def test_attribute_unavailable():
    r = attribute_removed([10, 11], None, {"p"})
    assert r["status"] == "unavailable"
    assert r["n_removed"] == 2 and r["n_resolved"] == 0


def test_attribute_all_to_p():
    # lines 10,11 both fall in a range owned by one of P's shas
    ranges = [[1, 9, "other"], [10, 12, "psha"]]
    r = attribute_removed([10, 11], ranges, {"psha", "psha2"})
    assert r == {"status": "ok", "n_removed": 2, "n_resolved": 2, "n_blame_to_p": 2}


def test_attribute_split_authors():
    # line 10 -> P, line 11 -> an intermediate PR (not P)
    ranges = [[10, 10, "psha"], [11, 11, "intermediate"]]
    r = attribute_removed([10, 11], ranges, {"psha"})
    assert r["status"] == "ok" and r["n_blame_to_p"] == 1 and r["n_resolved"] == 2


def test_attribute_unresolved_line():
    # removed line 99 falls in no blame range (file shorter at merge-base)
    ranges = [[1, 12, "psha"]]
    r = attribute_removed([99], ranges, {"psha"})
    assert r["status"] == "ok" and r["n_resolved"] == 0 and r["n_blame_to_p"] == 0


def test_parse_pr_url():
    assert parse_pr_url("https://github.com/o/r/pull/42") == ("o", "r", "42")
    assert parse_pr_url("not a url") is None


def test_is_rate_limited():
    assert _is_rate_limited("gh: API rate limit exceeded for user")
    assert _is_rate_limited("You have exceeded a secondary rate limit")
    assert _is_rate_limited("HTTP 429: Too Many Requests")
    assert not _is_rate_limited("HTTP 404: Not Found")
    assert not _is_rate_limited("")


def test_gh_json_waits_then_retries_on_rate_limit(monkeypatch):
    """A rate-limited call should pause (not crash) and then succeed on retry."""
    calls = {"n": 0}
    slept = {"n": 0}

    def fake_run(args, **kw):
        # the rate_limit probe inside _begin_pause -> report exhausted, reset soon
        if args[:3] == ["gh", "api", "rate_limit"]:
            body = '{"resources":{"core":{"remaining":0,"reset":0}}}'
            return SimpleNamespace(returncode=0, stdout=body, stderr="")
        calls["n"] += 1
        if calls["n"] == 1:  # first real call: rate limited
            return SimpleNamespace(returncode=1, stdout="", stderr="API rate limit exceeded")
        return SimpleNamespace(returncode=0, stdout='{"ok":true}', stderr="")

    monkeypatch.setattr(ba.subprocess, "run", fake_run)
    monkeypatch.setattr(ba.time, "sleep", lambda s: slept.__setitem__("n", slept["n"] + 1))
    monkeypatch.setattr(ba, "_rl_state", {"resume_at": 0.0})  # isolate shared state

    out = gh_json(["api", "repos/o/r/pulls/1"])
    assert out == {"ok": True}
    assert calls["n"] == 2          # retried after the limit cleared
    assert slept["n"] >= 1          # actually waited rather than crashing


def test_gh_json_returns_none_on_real_error(monkeypatch):
    """A non-rate-limit failure (e.g. 404) returns None without waiting."""
    def fake_run(args, **kw):
        return SimpleNamespace(returncode=1, stdout="", stderr="HTTP 404: Not Found")

    monkeypatch.setattr(ba.subprocess, "run", fake_run)
    monkeypatch.setattr(ba, "_rl_state", {"resume_at": 0.0})
    assert gh_json(["api", "repos/o/r/pulls/1"]) is None
