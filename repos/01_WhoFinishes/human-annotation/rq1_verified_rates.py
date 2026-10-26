"""RQ1.2 paper numbers: verified downstream-fix rates + kappa table.

Joins the LLM-judge population run (2,510 pairs) back onto the RQ1
uncensored frame (4,505 merged agent PRs), then emits every number the
RQ1.2 section cites:

  1. reconciliation of the judge frame vs the RQ1 frame,
  2. per-agent / overall verified Direct-fix rates (raw + precision-adjusted),
  3. the kappa table (both validation rounds, 3-way + binary, reweighted),
  4. the human-touch test re-run on verified labels.

Run:  uv run python human-annotation/rq1_verified_rates.py
Outputs: data/digests/rq1_verified_rates.csv, rq1_kappa_table.csv (+ stdout report).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

HERE = Path(__file__).resolve().parent
CODE = HERE.parent
OUT = CODE / "data" / "digests"
JUDGE = CODE / "data" / "judge"

JUDGE_POP = JUDGE / "rq1_llm_judge_pop_anthropic_claude-opus-4-8_low.csv"
CASES = CODE / "data" / "judge" / "rq1_judge_cases_population_12k.json"
PER_PR = OUT / "rq1_linker_per_pr.csv"
R1 = HERE / "labels" / "humans" / "rq1_followup_validation_agreed.csv"
R2_KEY = HERE / "labels" / "judge" / "rq1_judge_sample50_enriched_model_key.csv"
R2_HUM = HERE / "labels" / "judge" / "rq1_judge_sample50_enriched_annotator_anon_NHATDUONG.csv"
R2_META = HERE / "labels" / "judge" / "rq1_judge_sample50_enriched_sample_meta.json"
HP_KEY = HERE / "labels" / "judge" / "rq1_judge_sample50_enriched_humanP_model_key.csv"
HP_HUM = HERE / "labels" / "judge" / "rq1_judge_sample50_enriched_humanP_annotator_nhat3.csv"


# ---------------------------------------------------------------- helpers
def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def kappa(a, b, w=None) -> tuple[float, float]:
    """(observed agreement, Cohen's kappa), optionally per-item weighted."""
    a = pd.Series(a).astype(str).values
    b = pd.Series(b).astype(str).values
    wv = np.ones(len(a)) if w is None else np.asarray(w, float)
    W = wv.sum()
    po = float((wv * (a == b)).sum() / W)
    cats = sorted(set(a) | set(b))
    pe = sum(((wv * (a == c)).sum() / W) * ((wv * (b == c)).sum() / W) for c in cats)
    return po, (po - pe) / (1 - pe)


def as_binary(s) -> np.ndarray:
    return np.where(pd.Series(s).astype(str).str.startswith("Direct"), "fix", "nonfix")


# ---------------------------------------------------------------- 1. reconcile
judge = pd.read_csv(JUDGE_POP)
cases = pd.DataFrame(json.loads(CASES.read_text()))[["pair_id", "p_id", "n_substantive"]]
cases["pair_id"] = cases.pair_id.astype(int)
cases["p_id"] = cases.p_id.astype(int)
cases["n_substantive"] = cases.n_substantive.astype(int)
judge = judge.merge(cases, on="pair_id", validate="1:1")

per_pr = pd.read_csv(PER_PR)
unc = per_pr[per_pr.censored_30d == 0].copy()

# The judge population was gated on ANY shared file (n_shared>0); the paper's
# RQ1.1 linker requires a NON-BOILERPLATE shared file. Restricting to
# substantive pairs makes the two constructs coincide, so verified PRs are a
# strict subset of the linker's followed PRs.
sub = judge[judge.n_substantive > 0]
judge_prs = set(sub.p_id)
frame_prs = set(unc.pr_id)
followed_prs = set(unc[unc.has_followup_30d == 1].pr_id)

print("=== 1. Reconciliation: judge frame vs RQ1.1 uncensored frame ===")
print(f"judge population pairs: {len(judge)} over {judge.p_id.nunique()} PRs; "
      f"substantive (non-boilerplate) pairs: {len(sub)} over {len(judge_prs)} PRs")
_boiler_direct = int((judge.label.str.startswith("Direct") & (judge.n_substantive == 0)).sum())
print(f"boilerplate-only pairs dropped: {len(judge) - len(sub)} "
      f"(judge called Direct on {_boiler_direct})")
print(f"RQ1.1 uncensored frame: {len(unc)} PRs, followed: {len(followed_prs)}")
print(f"judge PRs in frame & followed:      {len(judge_prs & followed_prs)}")
print(f"judge PRs censored (out of frame):  {len(judge_prs - frame_prs)}")
bad = judge_prs & frame_prs - followed_prs
print(f"judge PRs in frame NOT followed (must be 0): {len(bad)}")
assert not bad, "substantive judge pairs must be a subset of linker-followed PRs"
# The 50 human-labelled (gold) pairs are deliberately excluded from the judge
# population (they informed the judge's prompt). 27 followed PRs have a gold
# pair as their ONLY candidate; without backfill they would be zeroed despite
# carrying human verdicts. Rule: verified = judge-Direct on any substantive
# pair OR human-consensus-Direct on any gold pair (humans are the gold
# standard, so their verdicts take no precision haircut).
missing = followed_prs - judge_prs
print(f"followed PRs with only gold (human-labelled, judge-excluded) pairs: "
      f"{len(missing)} ({len(missing) / len(followed_prs):.1%})")
att = unc[unc.pr_id.isin(missing)].groupby("agent").size()
print(f"gold-only PRs by agent: {att.to_dict()}")

from prfix.common.loaders.base import connect  # noqa: E402

_norm = lambda u: str(u).strip().rstrip("/").lower()  # noqa: E731
agreed = pd.read_csv(HERE / "labels" / "humans" / "rq1_followup_validation_agreed.csv")
_pr = connect(read_only=True).execute("SELECT id, html_url FROM pull_request").df()
_url2id = {_norm(u): int(i) for i, u in zip(_pr.id, _pr.html_url, strict=True)}
agreed["p_id"] = agreed.p_url.map(lambda u: _url2id[_norm(u)])
gold_direct = set(agreed[(agreed.label_1 == agreed.label_2)
                         & (agreed.label_1 == "Direct fix")].p_id)
gold_backfilled = missing & set(agreed.p_id)
assert gold_backfilled == missing, "every excluded followed PR must be a gold PR"
print(f"human-consensus Direct fixes among gold PRs: {len(gold_direct)} "
      f"(on gold-only PRs: {len(gold_direct & missing)})")

# paper frame: substantive judge pairs on uncensored PRs
jf = sub[sub.p_id.isin(frame_prs)].copy()
jf["direct"] = jf.label.str.startswith("Direct")
judge_verified = set(jf[jf.direct].p_id)

# ---------------------------------------------------------------- 2. verified rates
print("\n=== 2. Verified Direct-fix rates (uncensored frame, gold-backfilled) ===")
unc["hum_ver"] = unc.pr_id.isin(gold_direct & frame_prs).astype(int)
unc["judge_ver"] = (unc.pr_id.isin(judge_verified) & ~unc.hum_ver.astype(bool)).astype(int)
unc["verified"] = ((unc.hum_ver + unc.judge_ver) > 0).astype(int)

# Precision is POOLED over both blind spot-checks (agent-merge and human-merge
# pairs, 27/30 each -- user-locked 2026-08-17): both 30-call sets are uniform
# samples of the judge's Direct calls in their populations, and the scaling is
# applied to both cohorts' verified rates, so the pooled estimate matches its
# use and tightens the Wilson interval (74-97% -> 80-95%). The per-side split
# stays typeset in threats as the transfer check.
def _direct_ok(hum_path: Path, key_path: Path) -> tuple[int, int]:
    m = pd.read_csv(hum_path)[["pair_id", "label"]].merge(
        pd.read_csv(key_path)[["pair_id", "label"]].rename(columns={"label": "llm"}),
        on="pair_id", validate="1:1")
    d = m[m.llm == "Direct fix"]
    return int((d.label == "Direct fix").sum()), len(d)


_ok_a, _nd_a = _direct_ok(R2_HUM, R2_KEY)
_ok_h, _nd_h = _direct_ok(HP_HUM, HP_KEY)
assert (_ok_a, _nd_a) == (27, 30) and (_ok_h, _nd_h) == (27, 30), \
    f"spot-check counts drifted: agent {_ok_a}/{_nd_a}, human {_ok_h}/{_nd_h}"
PRECISION, P_LO, P_HI = ((_ok_a + _ok_h) / (_nd_a + _nd_h),
                         *wilson(_ok_a + _ok_h, _nd_a + _nd_h))
rows = []
for agent, sub in [*unc.groupby("agent"), ("Overall", unc)]:
    n, cand = len(sub), int(sub.has_followup_30d.sum())
    jv, hv = int(sub.judge_ver.sum()), int(sub.hum_ver.sum())
    ver = jv + hv
    rows.append({
        "agent": agent, "n_uncensored": n,
        "candidate_pct": 100 * cand / n,
        "judge_verified": jv, "human_verified": hv,
        "verified_pct": 100 * ver / n,
        "verified_adj_pct": 100 * (jv * PRECISION + hv) / n,
        "verified_adj_lo": 100 * (jv * P_LO + hv) / n,
        "verified_adj_hi": 100 * (jv * P_HI + hv) / n,
        "share_of_candidates_pct": 100 * ver / cand if cand else np.nan,
    })
rates = pd.DataFrame(rows)
print(rates.round(1).to_string(index=False))

# tab:rq3-verified as typeset: (uncensored PRs, candidate %, verified %,
# adjusted %, precision band). None = no band (human-consensus verdicts only,
# which take no haircut, so the band collapses onto the point estimate).
TYPESET = {
    "OpenAI_Codex": (2049, 24.8, 5.5, 4.9, (4.4, 5.2)),
    "Cursor":       (153, 15.7, 5.2, 5.0, (4.8, 5.1)),
    "Devin":        (1519, 21.5, 3.7, 3.4, (3.0, 3.5)),
    "Claude_Code":  (63, 4.8, 3.2, 3.2, None),
    "Copilot":      (721, 23.4, 3.5, 3.1, (2.8, 3.3)),
    "Overall":      (4505, 22.9, 4.5, 4.1, (3.7, 4.3)),
}
assert set(rates.agent) == set(TYPESET), \
    f"table rows drifted: {sorted(set(rates.agent) ^ set(TYPESET))}"
for _, r in rates.iterrows():
    n, cand, ver, adj, band = TYPESET[r.agent]
    got = (int(r.n_uncensored), round(r.candidate_pct, 1),
           round(r.verified_pct, 1), round(r.verified_adj_pct, 1))
    assert got == (n, cand, ver, adj), \
        f"{r.agent}: {got} != typeset {(n, cand, ver, adj)}"
    got_band = (round(r.verified_adj_lo, 1), round(r.verified_adj_hi, 1))
    assert got_band == (band or (adj, adj)), \
        f"{r.agent} adjusted band {got_band} != typeset {band or (adj, adj)}"
print(f"(judge-verified portion scaled x{PRECISION:.2f} [{P_LO:.2f}, {P_HI:.2f}]; "
      f"human-verified portion unscaled)")

# population label mix on the paper frame
mix = jf.label.value_counts()
print(f"\npair-level label mix on frame (n={len(jf)}): {mix.to_dict()}")

# ---------------------------------------------------------------- 3. kappa table
print("\n=== 3. Kappa table ===")
r1 = pd.read_csv(R1)
key = pd.read_csv(R2_KEY)
hum = pd.read_csv(R2_HUM)
m = key.merge(hum[["pair_id", "label"]], on="pair_id", suffixes=("_j", "_h"))
meta = json.loads(R2_META.read_text())
strata = meta.get("strata", meta)
wmap = {s: d["population"] / d["sampled"] for s, d in strata.items()
        if isinstance(d, dict) and "population" in d}
if not wmap:  # ponytail: fixed fallback matches the recorded meta, recompute if strata change
    wmap = {"positive": 334 / 30, "decoy": 2176 / 20}
w = m.stratum.map(wmap).values

krows = []
for name, a, b, wv in [
    ("round1 human-human 3way", r1.label_1, r1.label_2, None),
    ("round1 human-human binary", as_binary(r1.label_1), as_binary(r1.label_2), None),
    ("round2 human-judge 3way raw", m.label_j, m.label_h, None),
    ("round2 human-judge binary raw", as_binary(m.label_j), as_binary(m.label_h), None),
    ("round2 human-judge 3way reweighted", m.label_j, m.label_h, w),
    ("round2 human-judge binary reweighted", as_binary(m.label_j), as_binary(m.label_h), w),
]:
    po, k = kappa(a, b, wv)
    krows.append({"comparison": name, "n": len(m) if "round2" in name else len(r1),
                  "agreement_pct": 100 * po, "kappa": k})
kt = pd.DataFrame(krows)
print(kt.round(3).to_string(index=False))
jd, hd = m.label_j.str.startswith("Direct"), m.label_h.str.startswith("Direct")
_lo_a, _hi_a = wilson(int((jd & hd).sum()), int(jd.sum()))
print(f"judge Direct-fix precision (agent side): {(jd & hd).sum()}/{jd.sum()} = "
      f"{(jd & hd).mean() / jd.mean():.1%} [Wilson 95% CI {_lo_a:.0%}-{_hi_a:.0%}]")
print(f"judge Direct-fix precision (pooled, used for scaling): "
      f"{_ok_a + _ok_h}/{_nd_a + _nd_h} = {PRECISION:.1%} "
      f"[Wilson 95% CI {P_LO:.0%}-{P_HI:.0%}]")

# ---------------------------------------------------------------- 4. human-touch on verified labels
print("\n=== 4. Human-touch test, outcome = verified Direct fix (non-Codex) ===")
from prfix.rq4_signals.per_pr_signals import compute as per_pr_compute  # noqa: E402

signals = per_pr_compute()[["pr_id", "agent", "n_human_commits"]].dropna()
ht = unc.merge(signals[["pr_id", "n_human_commits"]], on="pr_id", how="inner")
ht = ht[ht.agent != "OpenAI_Codex"].copy()
med = ht.groupby("agent")["n_human_commits"].transform("median")
ht["high_touch"] = (ht.n_human_commits > med).astype(int)

for label, outcome in [("candidate (replicates RQ1.1)", "has_followup_30d"),
                       ("verified Direct fix", "verified")]:
    hi = ht[ht.high_touch == 1][outcome]
    lo = ht[ht.high_touch == 0][outcome]
    tab = [[hi.sum(), len(hi) - hi.sum()], [lo.sum(), len(lo) - lo.sum()]]
    chi2, p = chi2_contingency(tab)[:2]
    print(f"{label}: high {hi.mean():.1%} vs low {lo.mean():.1%} "
          f"({100 * (hi.mean() - lo.mean()):+.1f}pp, chi2={chi2:.2f}, p={p:.2f}, N={len(ht)})")

# ---------------------------------------------------------------- write outputs
OUT.mkdir(parents=True, exist_ok=True)
rates.to_csv(OUT / "rq1_verified_rates.csv", index=False)
kt.to_csv(OUT / "rq1_kappa_table.csv", index=False)
print(f"\nwrote {OUT / 'rq1_verified_rates.csv'}")
print(f"wrote {OUT / 'rq1_kappa_table.csv'}")
