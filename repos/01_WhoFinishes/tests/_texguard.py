"""Paper-LaTeX guard for the replication package.

A handful of tests treat the typeset paper as the contract: they parse the
LaTeX tables and prose and assert the cells against the pinned digests. The
replication package ships without the paper sources, so those tests skip
(rather than error) when the tex tree is absent. In the authors' working tree,
where paper-emse/ exists next to the code, they run in full.
"""
from __future__ import annotations

from pathlib import Path

import pytest


def read_tex(path: Path) -> str:
    if not path.exists():
        pytest.skip(f"paper LaTeX not shipped in the replication package ({path.name})")
    return path.read_text()
