"""Author-class classifier extending P2 (Duma et al. EASE 2026) to commits."""

from prfix.rq3_commit_authors.classifier import (
    classify_comment,
    classify_commit,
    classify_login,
)

__all__ = ["classify_comment", "classify_commit", "classify_login"]
