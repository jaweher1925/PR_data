"""Smoke tests for the author-class classifier."""

from __future__ import annotations

import pathlib
import re

from prfix.rq3_commit_authors import classify_comment, classify_commit, classify_login
from tests._texguard import read_tex


def test_classify_login_agent_accounts():
    assert classify_login("Copilot") == "agent"
    assert classify_login("devin-ai-integration[bot]") == "agent"
    assert classify_login("cursoragent") == "agent"


def test_classify_login_known_bots():
    assert classify_login("dependabot[bot]") == "automation"
    assert classify_login("renovate-bot") == "automation"
    assert classify_login("github-actions") == "automation"
    assert classify_login("autofix-ci") == "automation"


def test_classify_login_humans():
    assert classify_login("alice") == "human"
    assert classify_login("MontrealAI") == "human"
    assert classify_login("") == "human"
    assert classify_login(None) == "human"


def test_classify_commit_agent_marker_in_message():
    # Login is a human but message has agent marker -> agent.
    msg = "Add caching\n\nCo-Authored-By: Claude <noreply@anthropic.com>"
    assert classify_commit("alice", "alice", msg) == "agent"

    msg2 = "fix(api): retry on 429\n\n🤖 Generated with [Claude Code]"
    assert classify_commit("alice", "alice", msg2) == "agent"


def test_classify_commit_agent_login_wins():
    assert classify_commit("Copilot", "Copilot", "ordinary message") == "agent"
    assert classify_commit("devin-ai-integration[bot]", None, None) == "agent"


def test_classify_commit_human_passes_through():
    assert classify_commit("alice", "alice", "Refactor user model") == "human"


def test_classify_commit_bot_committer():
    assert classify_commit("dependabot[bot]", "dependabot[bot]", "bump dep") == "automation"


def test_classify_comment_human_review():
    assert classify_comment("LGTM, ship it.", "bob", "User") == "human_review"


def test_classify_comment_agent_steering():
    assert classify_comment("@claude please rebase", "alice", "User") == "agent_steering"
    assert classify_comment("/copilot fix", "alice", "User") == "agent_steering"


def test_classify_comment_bot_user_type():
    assert classify_comment("build passed", "ci-bot", "Bot") == "automation"


def test_classify_comment_agent_account_is_automation_in_comments():
    # A comment posted by the agent account itself is bucketed with automation
    # (P2's convention: comments are about who is *directing* whom).
    assert classify_comment("here is the patch", "Copilot", "User") == "automation"


# --- Table 2 (tab:authorclass-evidence) -------------------------------------
# The RQ3 approach typesets the classifier's evidence as a table, so the paper
# is a text contract on that side: parse its rows into (class, logins,
# markers) and put each cell back through the classifier. A cell the
# classifier no longer honours means the table documents criteria the code
# does not apply.

RESULTS_TEX = (pathlib.Path(__file__).resolve().parents[2]
               / "paper-emse" / "sections" / "results.tex")

# Cells that name a pattern rather than spell one out: swap in a probe string
# that the pattern is meant to match.
_PROBES = [
    (r"\texttt{Generated with} $+$ robot emoji",
     r"\texttt{🤖 Generated with the agent}"),
    (r"\texttt{Generated with \ldots{} Codex}",
     r"\texttt{Generated with the OpenAI Codex CLI}"),
    (r"any other \texttt{*[bot]}", r"\texttt{some-other-tool[bot]}"),
]


def _table_rows():
    """tab:authorclass-evidence -> [(class, [logins], [markers])]."""
    body = re.search(r"\\label\{tab:authorclass-evidence\}.*?\\midrule(.*?)\\bottomrule",
                     read_tex(RESULTS_TEX), re.S)
    assert body, "tab:authorclass-evidence is no longer typeset in results.tex"
    rows = []
    for raw_row in re.split(r"\\\\", body.group(1)):
        raw = raw_row.replace("\\addlinespace", "").replace("\\midrule", "").strip()
        if not raw:
            continue
        cells = [re.sub(r"\s+", " ", c).strip() for c in raw.split("&")]
        assert len(cells) == 3, raw
        cols = []
        for raw_cell in cells[1:]:
            cell = raw_cell
            for pattern, probe in _PROBES:
                cell = cell.replace(pattern, probe)
            cols.append(re.findall(r"\\texttt\{(.+?)\}(?!\w)", cell))
        rows.append((cells[0], *cols))
    return rows


AGENT_ROWS = ["GitHub Copilot", "Devin", "Cursor", "Claude Code", "OpenAI Codex"]


def test_table2_agent_rows_classify_as_agent():
    rows = {r[0]: r[1:] for r in _table_rows()}
    assert list(rows) == [*AGENT_ROWS, "automation", "human"]
    for name in AGENT_ROWS:
        logins, markers = rows[name]
        assert logins and markers, name
        for login in logins:
            assert classify_commit(login, login, "ordinary message") == "agent", login
        for marker in markers:
            # Pooled set: a human login carrying any agent's marker is agent.
            assert classify_commit("alice", "alice", f"work\n\n{marker}") == "agent", marker


def test_table2_automation_and_human_rows():
    rows = {r[0]: r[1:] for r in _table_rows()}
    for login in rows["automation"][0]:
        assert classify_commit(login, login, "bump dependency") == "automation", login
    assert rows["human"] == ([], [])  # the default: no evidence needed
    assert classify_commit("alice", "alice", "Refactor user model") == "human"


def test_table2_caption_decision_order():
    """Caption's order: agent login, else marker, else bot login, else human."""
    # claude[bot] matches the bot denylist too; the agent login wins.
    assert classify_commit("claude[bot]", "claude[bot]", "chore") == "agent"
    # A marker beats a bot login.
    assert classify_commit("dependabot[bot]", "dependabot[bot]",
                           "bump\n\nCo-Authored-By: Devin") == "agent"
    # Bot login beats the human default.
    assert classify_commit("renovate-bot", None, "bump") == "automation"


def test_table2_codex_prior_disabled_is_what_the_shares_rest_on():
    """The caption says the shares use evidence alone, prior disabled."""
    unsigned = ("alice", "alice", "chore: tidy imports")
    assert classify_commit(*unsigned, pr_agent=None) == "human"
    assert classify_commit(*unsigned, pr_agent="OpenAI_Codex", pr_user="alice") == "agent"
