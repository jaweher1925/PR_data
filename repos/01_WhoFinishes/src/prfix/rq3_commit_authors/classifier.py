"""Rule-based author-class classifier.

Three classes:

* ``agent``     -- authored by one of the five AI coding agents
                   (login match OR commit-message marker).
* ``automation`` -- authored by a non-agent bot (CI, dependency updater,
                   semantic-release, etc.).
* ``human``     -- everything else.

This extends P2 (Duma et al., EASE 2026), which used a regex classifier for
PR review comments only, to commits as well.
"""

from __future__ import annotations

from prfix.rq3_commit_authors.rules import (
    AGENT_GENERATED_MARKERS,
    AGENT_LOGIN_PATTERNS,
    AGENT_STEERING_PATTERNS,
    BOT_LOGIN_PATTERNS,
)


def _as_str(value) -> str:
    """Coerce DuckDB/pandas string-ish values to ``str`` (NULL/NaN -> '')."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    # pandas converts SQL NULL strings into float('nan'); also handle bytes.
    if isinstance(value, float):  # NaN
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def classify_login(login) -> str:
    """Return ``'agent'``, ``'automation'``, or ``'human'`` for a GitHub login."""
    s = _as_str(login)
    if not s:
        return "human"
    for p in AGENT_LOGIN_PATTERNS:
        if p.match(s):
            return "agent"
    for p in BOT_LOGIN_PATTERNS:
        if p.match(s):
            return "automation"
    return "human"


def _message_has_agent_marker(message) -> bool:
    s = _as_str(message)
    if not s:
        return False
    return any(p.search(s) for p in AGENT_GENERATED_MARKERS)


def classify_commit(
    author,
    committer,
    message,
    pr_agent: str | None = None,
    pr_user: str | None = None,
) -> str:
    """Classify a commit row.

    Decision order:
        1. login is a known agent account                 -> ``agent``
        2. commit message has an agent-generated marker   -> ``agent``
        3. *Codex prior*: PR labelled ``OpenAI_Codex`` and
           commit author == PR opener                     -> ``agent``
        4. login is a known non-agent bot                 -> ``automation``
        5. otherwise                                      -> ``human``

    The Codex prior addresses a classifier blind-spot: Codex commits are
    authored under the human developer's GitHub login and carry no
    standardised marker (<2% of merged Codex commits contain any of the
    obvious strings we probed). Using the PR-level AIDev label as a prior
    is the simplest principled way to recover the agent attribution we'd
    otherwise lose for ~99% of Codex commits.

    Pass ``pr_agent=None`` (the default) to disable the prior -- the
    classifier then runs in pure content-based mode, matching its
    behaviour on synthetic test inputs.
    """
    a_login = _as_str(author)
    if classify_login(a_login) == "agent":
        return "agent"
    if classify_login(committer) == "agent":
        return "agent"
    if _message_has_agent_marker(message):
        return "agent"
    if (pr_agent == "OpenAI_Codex"
            and pr_user is not None
            and a_login != ""
            and a_login == _as_str(pr_user)):
        return "agent"
    if classify_login(a_login) == "automation":
        return "automation"
    if classify_login(committer) == "automation":
        return "automation"
    return "human"


def classify_comment(body, login, user_type) -> str:
    """Classify a PR comment / review / inline-review-comment row.

    Three classes:

    * ``agent_steering`` -- a HUMAN comment that directs the agent (@claude,
      /copilot, etc.). This is P2's ``agentic`` category.
    * ``automation``     -- posted by an agent account or any other bot.
    * ``human_review``   -- substantive human comments (everything else).
    """
    if _as_str(user_type).lower() == "bot":
        return "automation"
    login_class = classify_login(login)
    if login_class in ("agent", "automation"):
        return "automation"
    body_s = _as_str(body)
    if body_s and any(p.search(body_s) for p in AGENT_STEERING_PATTERNS):
        return "agent_steering"
    return "human_review"
