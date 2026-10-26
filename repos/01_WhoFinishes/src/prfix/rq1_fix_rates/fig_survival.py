"""RQ1 figure: cumulative incidence of the first follow-up fix (Fig. 3).

Statistics (verified pairs, survival frames, KM) come from
prfix.common.metrics -- the same functions the consolidation gate asserts --
so the figure and the gate always agree. This script only plots.

Writes fig_repair_survival.{pdf,png} into figures/: candidate + verified
panels, agent vs human cohorts on the SHARED cut-off frame.

Run:  uv run python -m prfix.rq1_fix_rates.fig_survival
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from prfix.common import metrics
from prfix.config import FIGURES_DIR

WINDOW = metrics.WINDOW
COL_AGENT, COL_HUMAN = "#c44e52", "#4c72b8"

sns.set_theme(style="whitegrid", font_scale=0.9)


def fig_survival(sa: pd.DataFrame, sh: pd.DataFrame) -> dict:
    both = pd.concat([sa, sh], ignore_index=True)
    stats = {}
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.4), sharex=True)
    for ax, kind, title in zip(
            axes, ("cand", "ver"),
            ("Candidate fix (linker)", "Verified fix (judge + human gold)"),
            strict=True):
        for sub, color in ((sa, COL_AGENT), (sh, COL_HUMAN)):
            xs, ys = metrics.km_cuminc(sub[f"{kind}_time"].to_numpy(),
                                       sub[f"{kind}_event"].to_numpy())
            ax.step(xs, 100 * ys, where="post", color=color,
                    label=f"{sub.cohort.iloc[0]} (n={len(sub):,})")
        hr, lo, hi, p = metrics.cox_hr(both, kind)
        stats[kind] = {"hr": hr, "lo": lo, "hi": hi, "p": p}
        # No HR annotation on the figure: the paper reports MH estimates only;
        # the Cox values stay gate-asserted.
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Days since merge")
        ax.set_xlim(0, WINDOW)
        ax.set_ylim(bottom=0)
    axes[0].set_ylabel("Merges followed by a fix (%)")
    axes[1].legend(loc="lower right", fontsize=8.5, frameon=False)
    fig.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / "fig_repair_survival.pdf", bbox_inches="tight")
    fig.savefig(FIGURES_DIR / "fig_repair_survival.png", dpi=200,
                bbox_inches="tight")
    plt.close(fig)
    print("  wrote fig_repair_survival.pdf/.png")
    return stats


def main() -> None:
    va, vh = metrics.verified_pair_sets()
    print(f"verified pairs: agent {len(va)}, human {len(vh)}")

    print("== survival (shared cut-off) ==")
    sa = metrics.survival_frame("agent", va)
    sh = metrics.survival_frame("human", vh)
    stats = fig_survival(sa, sh)
    for kind, s in stats.items():
        print(f"  {kind}: HR {s['hr']:.3f} [{s['lo']:.3f},{s['hi']:.3f}] "
              f"p={s['p']:.2e}")


if __name__ == "__main__":
    main()
