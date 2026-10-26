"""RQ4 figure: split violins of the four merge-time signals (Fig. 4).

The frame comes from prfix.common.metrics.signal_frame -- the same population
the RQ4 table tests -- so the figure and the gate always agree. This script
only plots.

Writes into figures/:

  fig_violin_candidate  -- split violins (candidate fix vs none) for the
                           four merge-time signals, own-cut-off agent frame.
  fig_violin_verified   -- same, verified-fix outcome.
  fig_violin_combined   -- both transparent; verified split saturated in
                           front, candidate split pale behind (the version
                           typeset in the paper).

Run:  uv run python -m prfix.rq4_signals.fig_violins
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Patch

from prfix.common import metrics
from prfix.config import FIGURES_DIR

COL_NONE, COL_FIX = "#4c72b8", "#c44e52"

sns.set_theme(style="whitegrid", font_scale=0.9)

LABELS = {
    "hours_under_review": "Time under review (h)",
    "n_commits": "Commits per PR",
    "lines_post_first": "Post-first-push churn (lines)",
    "n_review_items": "Review items",
}
# Signals and their order come from prfix.common.metrics, so the figure
# always draws the population the RQ4 table tests.
SIGNALS = [(c, LABELS[c]) for c in metrics.RQ4_SIGNALS]
TICKS = [0, 1, 10, 100, 1000, 10000]


def _half_violins(ax, df: pd.DataFrame, col: str, split_col: str,
                  labels: tuple[str, str], alpha: float = 1.0) -> None:
    d = df[[col, split_col]].dropna().copy()
    d["v"] = np.log1p(d[col].clip(lower=0))
    d["grp"] = np.where(d[split_col] == 1, labels[1], labels[0])
    sns.violinplot(data=d, y="v", hue="grp", hue_order=list(labels),
                   split=True, inner="quart", ax=ax, cut=0, linewidth=0.8,
                   palette={labels[0]: COL_NONE, labels[1]: COL_FIX},
                   alpha=alpha, legend=False)


def fig_violin(df: pd.DataFrame, outcome: str, name: str, title: str) -> None:
    labels = ("No follow-up fix", "Follow-up fix")
    fig, axes = plt.subplots(1, len(SIGNALS), figsize=(9.6, 3.1))
    for ax, (col, lab) in zip(axes, SIGNALS, strict=True):
        _half_violins(ax, df, col, outcome, labels)
        style_axis(ax, lab)
    n1, n0 = int(df[outcome].sum()), int((1 - df[outcome]).sum())
    add_legend(fig, labels, f"{title} — {n1:,} vs {n0:,} merges")
    save(fig, name)


def fig_violin_combined(df: pd.DataFrame) -> None:
    # Both distributions transparent so they read through each other; the
    # VERIFIED split carries the emphasis, candidate sits behind as pale
    # context (the version typeset in the paper).
    labels = ("No follow-up fix", "Follow-up fix")
    fig, axes = plt.subplots(1, len(SIGNALS), figsize=(9.6, 3.1))
    for ax, (col, lab) in zip(axes, SIGNALS, strict=True):
        _half_violins(ax, df, col, "has_followup_30d", labels, alpha=0.22)
        _half_violins(ax, df, col, "verified", labels, alpha=0.60)
        style_axis(ax, lab)
    add_legend(fig, labels,
               "Verified split (saturated) over candidate split (pale)")
    save(fig, "fig_violin_combined")


def style_axis(ax, label: str) -> None:
    ax.set_ylabel("")
    ax.set_xlabel("")
    ax.set_title(label, fontsize=9)
    ax.set_yticks([np.log1p(t) for t in TICKS])
    ax.set_yticklabels([f"{t:,}" for t in TICKS], fontsize=8)
    ax.set_ylim(bottom=0)


def add_legend(fig, labels: tuple[str, str], note: str) -> None:
    handles = [Patch(facecolor=COL_NONE, label=labels[0]),
               Patch(facecolor=COL_FIX, label=labels[1])]
    fig.legend(handles=handles, loc="upper right", ncol=2, fontsize=8.5,
               frameon=False)
    fig.suptitle(note, x=0.01, ha="left", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.92))


def save(fig, name: str) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(FIGURES_DIR / f"{name}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {name}.pdf/.png")


def main() -> None:
    va, _ = metrics.verified_pair_sets()
    print("== violins (own cut-off, uncensored 30d) ==")
    df = metrics.signal_frame(va)
    print(f"  frame n={len(df)}  candidate={int(df.has_followup_30d.sum())} "
          f"verified={int(df.verified.sum())}")
    fig_violin(df, "has_followup_30d", "fig_violin_candidate",
               "Candidate fix vs none")
    fig_violin(df, "verified", "fig_violin_verified", "Verified fix vs none")
    fig_violin_combined(df)


if __name__ == "__main__":
    main()
