from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
HUMANS = HERE / "labels" / "humans"   # exported human annotator sheets
JUDGE = HERE / "labels" / "judge"     # judge model keys + blind/annotator sheets

LABELS = ["Direct fix", "Related touch", "Unrelated/mislabeled", "Indeterminate"]


def cohens_kappa(a: pd.Series, b: pd.Series, w: pd.Series | None = None) -> float:
    """Cohen's kappa. With per-item weights ``w`` it is the weighted (population-
    reweighted) kappa — pass the sampling-stratum weights to undo an enriched draw.
    ``w=None`` reproduces the plain unweighted kappa."""
    a = a.reset_index(drop=True)
    b = b.reset_index(drop=True)
    wv = np.ones(len(a)) if w is None else np.asarray(w, float)
    W = wv.sum()
    if W == 0:
        return float("nan")
    cats = sorted(set(a) | set(b))
    po = (wv * (a.values == b.values)).sum() / W
    pe = sum(((wv * (a.values == c)).sum() / W) * ((wv * (b.values == c)).sum() / W) for c in cats)
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    m = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - m) / d, (c + m) / d)


def fleiss_kappa(counts: np.ndarray) -> float:
    """Fleiss' kappa for N raters. counts: items x categories (raters per
    cat); each row sums to m."""
    counts = np.asarray(counts, float)
    n, _ = counts.shape
    m = counts.sum(axis=1)[0]
    if m < 2:
        return float("nan")
    p_j = counts.sum(axis=0) / (n * m)
    P_i = (np.square(counts).sum(axis=1) - m) / (m * (m - 1))
    P_bar, P_e = P_i.mean(), np.square(p_j).sum()
    return (P_bar - P_e) / (1 - P_e) if P_e < 1 else 1.0


def discover_sheets() -> dict[str, pd.DataFrame]:
    """Any number of labelled annotator CSVs from labels/humans/ (each has pair_id, label)."""
    sheets = {}
    for f in sorted(HUMANS.glob("rq1_followup_validation_annotator*.csv")):
        d = pd.read_csv(f).dropna(subset=["label"])
        d = d[d["label"].astype(str).str.strip() != ""]
        if len(d):
            who = f.stem.replace("rq1_followup_validation_annotator", "").lstrip("_")
            sheets[who or f.stem] = d
    return sheets


def _pick_master() -> tuple[Path, str]:
    """Sanity-check that the annotator sheets match the net-diff master (the only
    labelled set; the superseded commit-union draw was archived 2026-07)."""
    cand = sorted(HUMANS.glob("rq1_followup_validation_annotator*.csv"))
    a = pd.read_csv(cand[0])[["pair_id", "q_url"]]
    p = DATA / "net_diff" / "master.csv"
    j = a.merge(pd.read_csv(p)[["pair_id", "q_url"]].rename(columns={"q_url": "q_m"}), on="pair_id")
    if len(j) and (j["q_url"] == j["q_m"]).mean() > 0.8:
        return p, "net_diff"
    raise SystemExit("annotator sheets do not match data/net_diff/master.csv -- "
                     "wrong sample?")


def cmd_humans(_args: argparse.Namespace) -> None:
    sheets = discover_sheets()
    if len(sheets) < 2:
        print(f"Need >=2 labelled annotator sheets (found {len(sheets)}). "
              "Fill `label` and export one CSV per rater.")
        return
    mpath, mname = _pick_master()
    print(f"matched validation set: {mname}")
    master = pd.read_csv(mpath)[["pair_id", "touch_group"]]
    raters = list(sheets)
    print(f"raters ({len(raters)}): {', '.join(raters)}")

    # Keep only pairs labelled by ALL raters (Fleiss needs equal rater count).
    wide = master.set_index("pair_id")
    for r, d in sheets.items():
        wide = wide.join(d.set_index("pair_id")["label"].rename(r))
    full = wide.dropna(subset=raters)
    dropped = len(wide) - len(full)
    note = f"  ({dropped} dropped — not all raters)" if dropped else ""
    print(f"pairs labelled by all raters: {len(full)}{note}")

    mat = np.array([[(full.loc[i, raters] == c).sum() for c in LABELS] for i in full.index])
    print(f"Fleiss' kappa: {fleiss_kappa(mat):.3f}")
    if len(raters) == 2:
        print(f"Cohen's kappa ({raters[0]} vs {raters[1]}): "
              f"{cohens_kappa(full[raters[0]], full[raters[1]]):.3f}")

    # Plurality vote; ties flagged for manual resolution.
    def majority(row):
        c = Counter(row[raters])
        top, k = c.most_common(1)[0]
        return top if list(c.values()).count(k) == 1 else "TIE"
    full = full.copy()
    full["adj"] = full.apply(majority, axis=1)
    ties = full[full.adj == "TIE"]
    if len(ties):
        print(f"  {len(ties)} pairs with no majority — adjudicate manually:")
        print(ties[raters].to_string())

    def prec(sub: pd.DataFrame):
        n = len(sub)
        if not n:
            print("    (no pairs)")
            return
        strict = int((sub.adj == "Direct fix").sum())
        lenient = int(sub.adj.isin(["Direct fix", "Related touch"]).sum())
        ls, hs = wilson(strict, n)
        ll, hl = wilson(lenient, n)
        print(f"    n={n:2d}  strict={strict/n:.2f} [{ls:.2f},{hs:.2f}]  "
              f"lenient={lenient/n:.2f} [{ll:.2f},{hl:.2f}]")

    scored = full[full.adj != "TIE"]
    print("precision on majority-vote label (overall):")
    prec(scored)
    print("precision per touch-group:")
    for g in ("high", "low"):
        print(f"  {g}:", end=" ")
        prec(scored[scored.touch_group == g])


def discover_sets() -> list[tuple[str, Path]]:
    """Every sample set with a model key: (stem, model_key_path)."""
    return [(key.name[: -len("_model_key.csv")], key)
            for key in sorted(JUDGE.glob("rq1_judge_sample*_model_key.csv"))]


def load_annotator_sheet(path: Path) -> tuple[str, pd.DataFrame] | None:
    """(who, [pair_id, label]) for a filled human sheet, or None if it has no
    labels. `who` prefers the annotator column, else the filename's trailing
    '..._annotator_<who>' token (the labeler often exports the column blank)."""
    d = pd.read_csv(path)
    if "label" not in d.columns:
        return None
    d = d[d["label"].astype(str).str.strip().ne("") & d["label"].notna()]
    if not len(d):
        return None
    names = ([str(v).strip() for v in d["annotator"].dropna().tolist()]
             if "annotator" in d.columns else [])
    names = [v for v in names if v and v.lower() != "nan"]
    who = names[0] if names else path.stem.split("annotator_", 1)[-1]
    return who, d[["pair_id", "label"]].copy()


def discover_annotators(stem: str) -> dict[str, pd.DataFrame]:
    """Filled human sheets for a stem, keyed by annotator; dedup keeps the
    fullest sheet per person (a blind sheet and its filled export can coexist).

    Stems are prefixes of one another (``..._enriched`` vs ``..._enriched_humanP``),
    so a sheet is skipped when it belongs to a longer stem that has its own model
    key -- otherwise one set's labels get scored against the other set's key."""
    longer = [s for s, _ in discover_sets() if s != stem and s.startswith(f"{stem}_")]
    best: dict[str, pd.DataFrame] = {}
    for f in sorted(JUDGE.glob(f"{stem}_*annotator*.csv")):
        if f.name.endswith("_model_key.csv"):
            continue
        if any(f.name.startswith(f"{s}_") for s in longer):
            continue
        got = load_annotator_sheet(f)
        if not got:
            continue
        who, d = got
        if who not in best or len(d) > len(best[who]):
            best[who] = d
    return best


def report(who: str, human: pd.DataFrame, key: pd.DataFrame,
           wmap: dict | None = None) -> None:
    """Human-vs-judge agreement, Cohen's kappa, and the judge's Direct-fix
    precision.

    Enriched samples oversample the judge's Direct-fix calls, so the RAW
    agreement/kappa over the drawn pairs are NOT population-representative. When
    the sample's stratum weights are available (``sample_meta.json``) we also
    report the population-reweighted agreement and kappa. The Direct-fix
    precision is read straight off the judge's Direct-fix calls (the enriched
    ``positive`` stratum is a uniform sample of them), so it already estimates
    the population precision; we add a Wilson 95% CI."""
    df = human.rename(columns={"label": "you"}).merge(
        key.rename(columns={"label": "llm"}), on="pair_id", how="inner")
    df["you"] = df["you"].astype(str).str.strip()
    df["llm"] = df["llm"].astype(str).str.strip()
    df["agree"] = df["you"] == df["llm"]

    n, agreed = len(df), int(df["agree"].sum())
    print(f"\n=== {who} ===")
    missing = len(human) - n
    if missing:
        print(f"  ! {missing} labelled pair(s) not found in the model key (skipped)")
    if not n:
        print("  no overlapping pairs")
        return
    print(f"  pairs labelled: {n}  (AGREED {agreed} / DISAGREED {n - agreed})")

    # per-item population weight from the sampling stratum (undoes the enriched draw)
    w = None
    if wmap and "stratum" in df.columns:
        df["w"] = df["stratum"].map(wmap)
        if df["w"].notna().all():
            w = df["w"]
        else:
            miss = sorted(set(df.loc[df["w"].isna(), "stratum"].astype(str)))
            print(f"  ! no weight for stratum(s) {miss}; raw sample rates only")

    k_raw = cohens_kappa(df["you"], df["llm"])
    print(f"  raw sample  : agreement {100 * agreed / n:.1f}%   "
          f"Cohen's kappa {k_raw:.3f}   (enriched draw, not population rate)")
    if w is not None:
        W = w.sum()
        agr_w = float((w.values * df["agree"].values).sum() / W)
        k_w = cohens_kappa(df["you"], df["llm"], w)
        print(f"  population   : agreement {100 * agr_w:.1f}%   "
              f"Cohen's kappa {k_w:.3f}   (reweighted via sample_meta strata)")
    else:
        print("  population   : n/a (no stratum weights; use a random/stratified "
              "sample or add sample_meta.json)")

    k_bin = cohens_kappa(df["you"] == "Direct fix", df["llm"] == "Direct fix")
    print(f"  raw binary  : Cohen's kappa {k_bin:.3f}   "
          f"(Direct fix vs not, enriched draw)")

    # Precision of the judge's Direct-fix calls (+ Wilson 95% CI).
    llm_direct = df[df["llm"] == "Direct fix"]
    nd = len(llm_direct)
    if nd:
        ok = int((llm_direct["you"] == "Direct fix").sum())
        lo, hi = wilson(ok, nd)
        print(f"  judge Direct-fix precision: {ok}/{nd} = {100 * ok / nd:.1f}%  "
              f"[95% CI {100 * lo:.0f}-{100 * hi:.0f}%]")
    else:
        print("  judge Direct-fix precision: n/a (judge labelled no pair 'Direct fix')")

    dis = df[~df["agree"]].sort_values("pair_id")
    if not len(dis):
        print("  (no disagreements)")
        return
    cw = max(len("YOU said"), *(len(x) for x in dis["you"]))
    print("  disagreeing pairs:")
    print(f"    {'pair':>7}  {'YOU said':<{cw}}  LLM said")
    for _, r in dis.iterrows():
        print(f"    #{r['pair_id']:<6} {r['you']:<{cw}}  {r['llm']}")


def _load_weights(key_path: Path) -> dict | None:
    """stratum -> population/sample weight from the sample's ``sample_meta.json``,
    used to reweight an enriched draw back to population rates. None if absent."""
    suffix = "_model_key.csv"
    stem = key_path.name[: -len(suffix)] if key_path.name.endswith(suffix) else key_path.stem
    mp = key_path.with_name(f"{stem}_sample_meta.json")
    if not mp.exists():
        return None
    strata = json.loads(mp.read_text(encoding="utf-8")).get("strata", {})
    return {name: s["weight"] for name, s in strata.items()
            if isinstance(s, dict) and "weight" in s}


def score_set(title: str, key_path: Path, anns: dict[str, pd.DataFrame]) -> None:
    if not anns:
        print(f"\n########## {title} ##########\n  (no filled annotator sheets)")
        return
    print(f"\n########## {title} ##########")
    print(f"model key: {key_path.name}   annotators: {', '.join(anns)}")
    kdf = pd.read_csv(key_path)
    cols = ["pair_id", "label"] + (["stratum"] if "stratum" in kdf.columns else [])
    key = kdf[cols]
    wmap = _load_weights(key_path)
    for who, d in anns.items():
        report(who, d, key, wmap)


def cmd_judge(args: argparse.Namespace) -> None:
    if args.annotator:
        if not args.model_key:
            raise SystemExit("--model-key is required when --annotator is given")
        anns: dict[str, pd.DataFrame] = {}
        for p in args.annotator:
            got = load_annotator_sheet(p)
            if not got:
                print(f"  (skipped {p.name}: no filled labels)")
                continue
            who, d = got
            anns[who if who not in anns else f"{who} ({p.stem})"] = d
        score_set("custom", args.model_key, anns)
        return

    sets = discover_sets()
    if not sets:
        print("No rq1_judge_sample*_model_key.csv found in labels/judge/.")
        return
    for stem, key_path in sets:
        score_set(stem, key_path, discover_annotators(stem))


def main() -> None:
    ap = argparse.ArgumentParser(description="Score the RQ1 validation labels")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("humans", help="inter-rater agreement + detector precision "
                                  "(auto-discovers annotator sheets in labels/humans/)")

    jp = sub.add_parser("judge", help="human-vs-LLM-judge agreement on the "
                                      "rq1_judge_sample sheets in labels/judge/")
    jp.add_argument("-a", "--annotator", type=Path, nargs="+", metavar="CSV",
                    help="one or more filled human sheets (blind sheet with `label` filled in)")
    jp.add_argument("-k", "--model-key", type=Path, metavar="CSV",
                    help="judge model-key CSV (pair_id, label); required with --annotator")

    args = ap.parse_args()
    {"humans": cmd_humans, "judge": cmd_judge}[args.cmd](args)


if __name__ == "__main__":
    main()
