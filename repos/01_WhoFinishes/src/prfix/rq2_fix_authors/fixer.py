"""Fixer identity of judged direct-fix pairs (RQ2, paper table `tab:rq3-fixer`).

Split out of human-annotation/consolidation_gate.py: the gate script only runs
with the full DuckDB lake, so the classification and the per-agent aggregation
live here where they can be tested on a synthetic frame.
"""
from __future__ import annotations

import pandas as pd

from prfix.config import DIGESTS_DIR

FIXERS = ("self", "human", "other-agent")


def add_fixer(da: pd.DataFrame, agent_of: pd.Series) -> pd.DataFrame:
    """Label each judged pair by who merged the fixing PR Q: a human, the same
    agent that merged P (self), or a different agent. `agent_of` maps PR id ->
    agent name."""
    out = da.copy()
    out["q_agent"] = out.q_id.map(agent_of)
    out["p_agent"] = out.p_id.map(agent_of)
    out["fixer"] = [
        "human" if h else ("self" if q == p else "other-agent")
        for h, q, p in zip(out.q_is_human, out.q_agent, out.p_agent, strict=True)
    ]
    return out


def per_agent_split(da: pd.DataFrame) -> pd.DataFrame:
    """Fixer counts per merging agent: one row per p_agent, one column per
    FIXERS category (absent categories are 0, so callers can index them
    directly) plus `n`. Agents with no judged pair get no row -- callers gate
    on that."""
    t = (da.groupby("p_agent").fixer.value_counts().unstack(fill_value=0)
         .reindex(columns=list(FIXERS), fill_value=0))
    t["n"] = t[list(FIXERS)].sum(axis=1)
    return t


# ------------------------------------------------- lake-free digest (row 28)
# Table `tab:rq3-fixer` is typeset on the RESTRICTED frame: the verified
# direct-fix pairs whose merged PR P sits in the cohort's fully observed frame.
# Everything about a pair except that membership is recoverable from the
# committed judge/gold artifacts, so the digest carries one row per
# unrestricted pair plus an `in_frame` flag -- the only lake-bound part. A lake
# run of consolidation_gate.py writes it and re-checks the round-trip;
# committed, and re-committed with any change to the fix-PR author rows.
DIGEST_FIXER = DIGESTS_DIR / "rq2_fixer_frame.csv"
_FIXER_COLS = ["p_id", "q_id", "p_agent", "fixer", "in_frame"]


def fixer_digest(da_all: pd.DataFrame, frame_ids: set[int]) -> pd.DataFrame:
    """One row per verified direct-fix pair (the unrestricted frame): merging
    agent, fixer class, and whether P is in the fully observed frame."""
    out = da_all.copy()
    out["in_frame"] = out.p_id.isin(frame_ids).astype(int)
    return out[_FIXER_COLS].sort_values(["p_id", "q_id"], ignore_index=True)


def write_fixer_digest(da_all: pd.DataFrame, frame_ids: set[int]) -> None:
    """Write the fixer-frame digest (both frames, one flag apart)."""
    DIGESTS_DIR.mkdir(parents=True, exist_ok=True)
    fixer_digest(da_all, frame_ids).to_csv(DIGEST_FIXER, index=False)
