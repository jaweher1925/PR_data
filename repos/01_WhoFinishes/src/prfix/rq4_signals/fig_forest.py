"""RQ4 conditional-logit figure candidates.

Paper-quality renderings of the within-repository layer: four options, two of
them also in a verified-only variant. Option 1's verified-only variant is the
adopted one (fig:rq4-forest); the rest are kept as alternatives. PDFs and PNGs
land in figures/,
which is where the paper copy of the adopted figure comes from. Style matches
fig_survival.py (whitegrid, COL_NONE/COL_FIX).

The frame and the conditional logit come from prfix.common.metrics --
the same functions behind the gate-asserted rq4_signal_stats.csv -- so the
figure's odds ratios and the table's cannot drift apart. This
script only plots.

Run:  uv run python -m prfix.rq4_signals.fig_forest
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from prfix.common import metrics

sns.set_theme(style="whitegrid", font_scale=0.9)
COL_NONE, COL_FIX = "#4c72b8", "#c44e52"
from prfix.config import FIGURES_DIR as OUT_FIG  # noqa: E402

OUT_FIG.mkdir(parents=True, exist_ok=True)

LABELS = {"hours_under_review": "Time under review",
          "n_commits": "Commits per PR",
          "lines_post_first": "Post-first-push churn",
          "n_review_items": "Review items"}


def clogit_ci(df: pd.DataFrame) -> pd.DataFrame:
    """OR per tenfold of 1+signal with 95% CI, both outcomes. The point
    estimates come from prfix.common.metrics, so they are the ORs the gate pins; only the whiskers
    (from the same fits' standard errors) are new here."""
    rows = []
    for outcome in ("has_followup_30d", "verified"):
        for col in metrics.RQ4_SIGNALS:
            _, fit = metrics.signal_clogit(df, col, outcome)
            b, se = float(np.asarray(fit.params)[0]), float(np.asarray(fit.bse)[0])
            rows.append({"outcome": outcome, "signal": col,
                         "or10": metrics.signal_or_10x(b),
                         "lo": metrics.signal_or_10x(b - 1.96 * se),
                         "hi": metrics.signal_or_10x(b + 1.96 * se),
                         "p": float(np.asarray(fit.pvalues)[0])})
    return pd.DataFrame(rows)


def save(fig, name: str) -> None:
    fig.savefig(OUT_FIG / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(OUT_FIG / f"{name}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)


def opt1_forest(ci: pd.DataFrame) -> None:
    order = ["n_commits", "n_review_items", "lines_post_first", "hours_under_review"]
    fig, ax = plt.subplots(figsize=(4.4, 2.4))
    for i, sig in enumerate(order):
        for outcome, color, dy, lab in (
                ("verified", COL_FIX, 0.12, "verified fix"),
                ("has_followup_30d", COL_NONE, -0.12, "candidate fix")):
            r = ci[(ci.outcome == outcome) & (ci.signal == sig)].iloc[0]
            ax.errorbar(r.or10, i + dy, xerr=[[r.or10 - r.lo], [r.hi - r.or10]],
                        fmt="o", color=color, ms=4.5, capsize=2.5, lw=1.2,
                        label=lab if i == 0 else None)
    ax.axvline(1, ls="--", c="0.4", lw=0.8)
    ax.set_xscale("log")
    ax.set_xticks([0.5, 1, 2, 4, 8])
    ax.set_xticklabels(["0.5", "1", "2", "4", "8"])
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([LABELS[s] for s in order])
    ax.invert_yaxis()
    ax.set_xlabel("Odds ratio per tenfold increase in 1+signal (log scale)")
    ax.legend(frameon=False, loc="lower right", fontsize=8)
    save(fig, "fig_rq4_forest")


def opt2_reversal(ci: pd.DataFrame) -> None:
    stats = pd.read_csv(metrics.DIGESTS_DIR / "rq4_signal_stats.csv")
    order = ["n_commits", "n_review_items", "lines_post_first", "hours_under_review"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.4), sharey=True,
                                   gridspec_kw={"width_ratios": [1, 1.3]})
    ypos = np.arange(len(order))
    for outcome, color, dy in (("verified", COL_FIX, 0.18),
                               ("has_followup_30d", COL_NONE, -0.18)):
        d = (stats[stats.outcome == outcome].set_index("signal")
             .reindex(order))
        ax1.barh(ypos + dy, d.cliffs_delta, height=0.32, color=color,
                 label="verified fix" if outcome == "verified" else "candidate fix")
    ax1.axvline(0, c="0.4", lw=0.8)
    ax1.set_yticks(ypos)
    ax1.set_yticklabels([LABELS[s] for s in order])
    ax1.invert_yaxis()
    ax1.set_xlabel("Pooled Cliff's $\\delta$ (fixed vs. not)")
    ax1.set_title("Pooled: nothing, or reversed", fontsize=9)
    ax1.legend(frameon=False, fontsize=7, loc="lower left")
    for i, sig in enumerate(order):
        for outcome, color, dy in (("verified", COL_FIX, 0.12),
                                   ("has_followup_30d", COL_NONE, -0.12)):
            r = ci[(ci.outcome == outcome) & (ci.signal == sig)].iloc[0]
            ax2.errorbar(r.or10, i + dy, xerr=[[r.or10 - r.lo], [r.hi - r.or10]],
                         fmt="o", color=color, ms=4.5, capsize=2.5, lw=1.2)
    ax2.axvline(1, ls="--", c="0.4", lw=0.8)
    ax2.set_xscale("log")
    ax2.set_xticks([0.5, 1, 2, 4, 8])
    ax2.set_xticklabels(["0.5", "1", "2", "4", "8"])
    ax2.set_xlabel("Within-repository OR per tenfold (log scale)")
    ax2.set_title("Within repositories: effort marks fixes", fontsize=9)
    save(fig, "fig_rq4_reversal")


def opt3_binned(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.4), sharey=True)
    for ax, col in zip(axes, ["n_commits", "n_review_items"], strict=True):
        d = df[[col, "verified", "repo_id"]].dropna().copy()
        d["x"] = np.log1p(d[col])
        # remove each repository's baseline fix rate
        d["dev"] = d.verified - d.groupby("repo_id").verified.transform("mean")
        d["bin"] = pd.qcut(d.x, 6, duplicates="drop")
        g = d.groupby("bin", observed=True).agg(x=("x", "median"), y=("dev", "mean"),
                                                se=("dev", "sem"), n=("dev", "size"))
        ax.errorbar(g.x, 100 * g.y, yerr=100 * 1.96 * g.se, fmt="o-",
                    color=COL_FIX, ms=4, capsize=2.5, lw=1.2)
        ax.axhline(0, ls="--", c="0.4", lw=0.8)
        ax.set_xlabel(f"log(1 + {LABELS[col].lower()})")
        ax.set_title(LABELS[col], fontsize=9)
    axes[0].set_ylabel("Verified-fix rate,\npp vs. repository baseline")
    save(fig, "fig_rq4_binned")


def opt4_slopes(df: pd.DataFrame) -> None:
    d = df[["n_commits", "verified", "repo_id"]].dropna()
    per = (d.groupby(["repo_id", "verified"]).n_commits.median().unstack())
    per = per.dropna()  # repos with both outcomes
    fig, ax = plt.subplots(figsize=(3.4, 2.8))
    for _, r in per.iterrows():
        up = r[1] > r[0]
        ax.plot([0, 1], [np.log1p(r[0]), np.log1p(r[1])],
                color=COL_FIX if up else COL_NONE, alpha=0.35, lw=1)
    med = per.median()
    ax.plot([0, 1], np.log1p(med.values), color="black", lw=2.5, marker="o")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["without\nfollow-up fix", "with\nfollow-up fix"])
    ax.set_ylabel("log(1 + median commits per PR)")
    n_up = int((per[1] > per[0]).sum())
    ax.set_title(f"Per-repository medians ({n_up}/{len(per)} rise)", fontsize=9)
    save(fig, "fig_rq4_slopes")


def opt1_forest_verified(ci: pd.DataFrame) -> None:
    """Verified-only forest (Wannita 2026-08-21: candidate series dropped)."""
    order = ["n_commits", "n_review_items", "lines_post_first", "hours_under_review"]
    fig, ax = plt.subplots(figsize=(4.2, 2.0))
    for i, sig in enumerate(order):
        r = ci[(ci.outcome == "verified") & (ci.signal == sig)].iloc[0]
        ax.errorbar(r.or10, i, xerr=[[r.or10 - r.lo], [r.hi - r.or10]],
                    fmt="o", color=COL_FIX, ms=5, capsize=3, lw=1.4)
    ax.axvline(1, ls="--", c="0.4", lw=0.8)
    ax.set_xscale("log")
    ax.set_xticks([0.5, 1, 2, 4, 8])
    ax.set_xticklabels(["0.5", "1", "2", "4", "8"])
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([LABELS[s] for s in order])
    ax.invert_yaxis()
    ax.set_xlabel("Odds ratio (log scale)")  # unit (per tenfold of 1+signal) lives in the caption
    save(fig, "fig_rq4_forest_verified")


def opt2_reversal_verified(ci: pd.DataFrame) -> None:
    """Verified-only two-panel reversal."""
    stats = pd.read_csv(metrics.DIGESTS_DIR / "rq4_signal_stats.csv")
    order = ["n_commits", "n_review_items", "lines_post_first", "hours_under_review"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.8, 2.0), sharey=True,
                                   gridspec_kw={"width_ratios": [1, 1.3]})
    ypos = np.arange(len(order))
    d = (stats[stats.outcome == "verified"].set_index("signal").reindex(order))
    ax1.barh(ypos, d.cliffs_delta, height=0.5, color=COL_FIX)
    ax1.axvline(0, c="0.4", lw=0.8)
    ax1.set_yticks(ypos)
    ax1.set_yticklabels([LABELS[s] for s in order])
    ax1.invert_yaxis()
    ax1.set_xlabel("Pooled Cliff's $\\delta$")
    ax1.set_title("Pooled: nothing, or reversed", fontsize=9)
    for i, sig in enumerate(order):
        r = ci[(ci.outcome == "verified") & (ci.signal == sig)].iloc[0]
        ax2.errorbar(r.or10, i, xerr=[[r.or10 - r.lo], [r.hi - r.or10]],
                     fmt="o", color=COL_FIX, ms=5, capsize=3, lw=1.4)
    ax2.axvline(1, ls="--", c="0.4", lw=0.8)
    ax2.set_xscale("log")
    ax2.set_xticks([0.5, 1, 2, 4, 8])
    ax2.set_xticklabels(["0.5", "1", "2", "4", "8"])
    ax2.set_xlabel("Within-repository OR per tenfold (log scale)")
    ax2.set_title("Within repositories: effort marks fixes", fontsize=9)
    save(fig, "fig_rq4_reversal_verified")


if __name__ == "__main__":
    df = metrics.signal_clogit_frame()
    ci = clogit_ci(df)
    print(ci.round(3).to_string(index=False))
    opt1_forest(ci)
    opt2_reversal(ci)
    opt3_binned(df)
    opt4_slopes(df)
    opt1_forest_verified(ci)
    opt2_reversal_verified(ci)
