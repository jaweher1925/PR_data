"""Regex patterns for the author-class classifier.

Logins observed empirically in AIDev (verified 2026-06-03):

    Copilot                       (9k commits, the Copilot account itself)
    devin-ai-integration[bot]     (8k commits, Devin's bot account)
    cursoragent                   (2k commits, Cursor's agent account)
    autofix-ci[bot]               (CI bot, non-agent)

OpenAI Codex and Claude Code post under the human developer's login, so
they must be detected via *commit-message markers* rather than login.
"""

from __future__ import annotations

import re

# --- Login patterns ---------------------------------------------------------

# Logins that ARE an AI coding agent (the 5 AIDev agents).
AGENT_LOGIN_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"^Copilot$"),
    re.compile(r"^devin-ai(?:-integration)?(?:\[bot\])?$"),
    re.compile(r"^cursor[a-z\-]*agent(?:\[bot\])?$"),
    re.compile(r"^claude(?:-code)?(?:\[bot\])?$"),
    re.compile(r"^codex(?:-cli)?(?:\[bot\])?$"),
]

# Logins that are bots but NOT one of the five AI coding agents.
BOT_LOGIN_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r".+\[bot\]$"),
    re.compile(r"^dependabot(?:-preview)?$"),
    re.compile(r"^renovate(?:-bot)?$"),
    re.compile(r"^codecov(?:-bot)?$"),
    re.compile(r"^github-actions$"),
    re.compile(r"^autofix-ci$"),
    re.compile(r"^vercel$"),
    re.compile(r"^netlify$"),
    re.compile(r"^semantic-release-bot$"),
    re.compile(r"^pre-commit-ci$"),
    re.compile(r"^allcontributors$"),
]

# --- Commit-message markers (agent-generated) -------------------------------

AGENT_GENERATED_MARKERS: list[re.Pattern[str]] = [
    re.compile(r"🤖\s*Generated with", re.IGNORECASE),
    re.compile(r"Generated with \[?Claude Code", re.IGNORECASE),
    re.compile(r"Co-Authored-By:\s*Claude", re.IGNORECASE),
    re.compile(r"Generated with.*Codex", re.IGNORECASE),
    re.compile(r"Co-Authored-By:\s*openai-codex", re.IGNORECASE),
    re.compile(r"Devin Run Link", re.IGNORECASE),
    re.compile(r"Co-Authored-By:\s*Devin", re.IGNORECASE),
    re.compile(r"Co-Authored-By:\s*Cursor", re.IGNORECASE),
    re.compile(r"Co-Authored-By:\s*Copilot", re.IGNORECASE),
]

# --- Comment-body patterns (P2's "agent-steering" category) -----------------
# Human comments that DIRECT an agent (slash commands, @-mentions).

AGENT_STEERING_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"^\s*@(claude|copilot|cursor|devin|codex|"
               r"copilot-pull-request-reviewer)\b",                re.IGNORECASE | re.MULTILINE),
    re.compile(r"^\s*/(claude|copilot|cursor|devin|codex)\b",      re.IGNORECASE | re.MULTILINE),
    re.compile(r"\bcopilot[ :]?(rebase|fix|review|merge|update)\b", re.IGNORECASE),
    re.compile(r"\b@?(claude|devin)[, ]+(please|can you)\b",       re.IGNORECASE),
]
