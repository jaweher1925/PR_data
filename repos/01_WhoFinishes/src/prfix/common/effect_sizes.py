"""Pairwise Mann-Whitney U + Cliff's delta, with Benjamini-Hochberg correction.

For a per-PR feature column and an agent label, we compute, for each
ordered pair (a, b) of distinct agents:

* ``U_a``               -- Mann-Whitney U statistic for group ``a``.
* ``cliffs_delta``       -- (2 * U_a) / (n_a * n_b) - 1   (range [-1, +1]).
* ``p_value``            -- two-sided MWU p-value (asymptotic).
* ``q_value``            -- Benjamini-Hochberg-corrected p-value across all
                            (signal, pair) tests for this signal.

Cliff's delta magnitude interpretation (Romano et al. 2006):

    |delta| < 0.147   negligible
    0.147..0.330      small
    0.330..0.474      medium
    >= 0.474          large
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def cliffs_delta_from_u(u_a: float, n_a: int, n_b: int) -> float:
    """Cliff's delta derived from the Mann-Whitney U statistic for group a."""
    return (2.0 * u_a) / (n_a * n_b) - 1.0


# Cliff's delta magnitude cut-offs (Romano et al. 2006).
DELTA_SMALL = 0.147
DELTA_MEDIUM = 0.330
DELTA_LARGE = 0.474


def magnitude(delta: float) -> str:
    """Map ``|delta|`` to a qualitative magnitude label."""
    a = abs(delta)
    if a < DELTA_SMALL:
        return "negligible"
    if a < DELTA_MEDIUM:
        return "small"
    if a < DELTA_LARGE:
        return "medium"
    return "large"


def pairwise(df: pd.DataFrame, signal_col: str, group_col: str = "agent") -> pd.DataFrame:
    """Return a DataFrame of pairwise MWU + Cliff's delta for ``signal_col``.

    Output columns: agent_a, agent_b, n_a, n_b, U_a, cliffs_delta, magnitude,
                    p_value.
    """
    groups = sorted(df[group_col].dropna().unique().tolist())
    rows = []
    for i, a in enumerate(groups):
        xa = df.loc[df[group_col] == a, signal_col].dropna().to_numpy()
        for b in groups[i + 1:]:
            xb = df.loc[df[group_col] == b, signal_col].dropna().to_numpy()
            if len(xa) == 0 or len(xb) == 0:
                continue
            try:
                u_a, p = stats.mannwhitneyu(xa, xb, alternative="two-sided")
            except ValueError:
                continue
            d = cliffs_delta_from_u(u_a, len(xa), len(xb))
            rows.append({
                "agent_a":      a,
                "agent_b":      b,
                "n_a":          len(xa),
                "n_b":          len(xb),
                "U_a":          float(u_a),
                "cliffs_delta": float(d),
                "magnitude":    magnitude(d),
                "p_value":      float(p),
            })
    return pd.DataFrame(rows)


def benjamini_hochberg(p_values: np.ndarray, fdr: float = 0.05) -> np.ndarray:
    """Compute BH-corrected q-values for an array of p-values."""
    p = np.asarray(p_values, dtype=float)
    n = len(p)
    if n == 0:
        return p
    order = np.argsort(p)
    ranked = p[order]
    q = ranked * n / (np.arange(n) + 1)
    # Enforce monotonicity: q_(i) = min(q_(i), q_(i+1), ...)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.clip(q, 0.0, 1.0)
    return out
