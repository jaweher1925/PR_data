"""Shared boilerplate / non-evidential file denylist for the RQ1 fix-linker.

A shared filename between an originating PR (P) and a candidate follow-up fix (Q)
is only *evidence of a fix relationship* if touching it reflects the substantive
change. These patterns match files that churn in nearly every PR regardless of
intent (lockfiles, changelogs, version bumps, repo/tool config, generated
artifacts), so their overlap is coincidental rather than evidential.

Used by both the validation-sheet enrichment (to flag/collapse such files) and
the tightened-linker preview (to drop boilerplate-only links). Keeping one
definition guarantees the two analyses agree.

This module is also the source of the rule table typeset in the paper as
appendix ``app:boilerplate`` (``paper-emse/sections/supplementary.tex``): the
table is a hand copy of the patterns below, so any rule added or removed here
has to be carried into it. ``tests/test_boilerplate.py`` pins the matching
behaviour the appendix describes.
"""
from __future__ import annotations

import fnmatch

# Exact basenames (compared case-insensitively).
_BASENAMES = {
    # --- lockfiles (auto-generated on any dependency change) ---
    "package-lock.json", "npm-shrinkwrap.json", "yarn.lock", "pnpm-lock.yaml",
    "bun.lock", "bun.lockb", "gemfile.lock", "poetry.lock", "pipfile.lock",
    "uv.lock", "cargo.lock", "composer.lock", "go.sum", "packages.lock.json",
    "pubspec.lock",
    # --- changelog / version (bumped on nearly every merge) ---
    "history.md", "releases.md", "version", "version.rb", "__version__.py",
    ".release-please-manifest.json",
    # --- repo meta (broad, intent-free churn) ---
    ".gitignore", ".gitattributes", ".ds_store",
}

# Glob patterns on the basename.
_BASENAME_GLOBS = ["*.snap", "*.generated.*", "*_pb2.py"]

# Path-prefix patterns (tooling / agent config directories).
_DIR_PREFIXES = (".cursor/", ".vscode/", ".idea/", ".devcontainer/")


def is_boilerplate(path: str) -> bool:
    """True if *path* is a non-evidential (boilerplate) file."""
    p = path.lower()
    if p.startswith("./"):
        p = p[2:]
    base = p.rsplit("/", 1)[-1]
    if base in _BASENAMES:
        return True
    if base.startswith("changelog"):           # CHANGELOG, CHANGELOG.md, .rst, ...
        return True
    if any(p.startswith(d) for d in _DIR_PREFIXES):
        return True
    if "__snapshots__/" in p:
        return True
    if any(fnmatch.fnmatch(base, g) for g in _BASENAME_GLOBS):
        return True
    return False
