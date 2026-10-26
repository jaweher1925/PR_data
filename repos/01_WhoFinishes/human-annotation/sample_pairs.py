"""Sample RQ1 human-annotation pairs. Two subcommands:

  * ``gold``  -- sample + enrich candidate (P, Q) pairs from the DuckDB database
                 into the gold *detector*-validation set (per ``--linker`` / ``--stage``).
  * ``judge`` -- draw a blind human-vs-*judge* validation sample from a population
                 LLM-judge CSV (``rq1_llm_judge_pop_*.csv``): random / stratified /
                 enriched, writing the blind sheet + held-back model key + reweighting
                 meta into ``labels/judge/``. (Moved here from ``llm_judge/sampling.py``
                 so all sampling lives in this folder; the judged population already
                 excludes the annotator-agreed pairs, so no exclusion is needed here.)

    uv run python human-annotation/sample_pairs.py gold  --linker net-diff --stage sample
    uv run python human-annotation/sample_pairs.py judge --mode enriched --n 50 --pop-csv <pop.csv>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
SEED = 42
AGENTS_NONCODEX = ["Devin", "Copilot", "Cursor", "Claude_Code"]

PAIR_SQL = """
WITH p_pop AS (
  SELECT p.id p_id, p.repo_id, p.agent, TRY_CAST(p.merged_at AS TIMESTAMP) p_merged
  FROM pull_request p JOIN repository r ON r.id=p.repo_id
  WHERE p.merged_at IS NOT NULL AND r.stars>=500),
q_agent AS (
  SELECT q.id q_id, q.repo_id, TRY_CAST(q.merged_at AS TIMESTAMP) q_merged
  FROM pull_request q JOIN repository r ON r.id=q.repo_id LEFT JOIN pr_task_type t ON t.id=q.id
  WHERE q.merged_at IS NOT NULL AND r.stars>=500
    AND (t.type='fix' OR q.title LIKE 'Revert %' OR q.title LIKE 'revert %')),
q_human AS (
  SELECT h.id q_id, rh.id repo_id, TRY_CAST(h.merged_at AS TIMESTAMP) q_merged
  FROM human_pull_request h JOIN repository rh ON rh.url=h.repo_url
       LEFT JOIN human_pr_task_type t ON t.id=h.id
  WHERE h.merged_at IS NOT NULL AND rh.stars>=500
    AND (t.type='fix' OR h.title LIKE 'Revert %' OR h.title LIKE 'revert %')),
q_all AS (SELECT * FROM q_agent UNION ALL SELECT * FROM q_human),
files AS ({files_cte}),
pair AS (
  SELECT p.p_id, p.agent, q.q_id,
         DATE_DIFF('day', p.p_merged, q.q_merged) AS day_gap
  FROM p_pop p JOIN q_all q
    ON q.repo_id=p.repo_id AND q.q_merged>p.p_merged
   AND q.q_merged<=p.p_merged+INTERVAL 30 DAY AND q.q_id<>p.p_id)
SELECT pair.p_id, pair.agent, pair.q_id, pair.day_gap,
       STRING_AGG(DISTINCT pf.filename, ' | ') AS shared_files,
       COUNT(DISTINCT pf.filename) AS n_shared
FROM pair
JOIN files pf ON pf.pr_id=pair.p_id
JOIN files qf ON qf.pr_id=pair.q_id AND qf.filename=pf.filename
GROUP BY pair.p_id, pair.agent, pair.q_id, pair.day_gap
"""


def _censored_net_sub() -> pd.DataFrame:
    from prfix.rq1_fix_rates import linker as rb
    return rb.compute(linker="net_sub")[["pr_id", "censored_30d"]]


# `alloc` is the agent-stratified pick per touch group (25 high / 25 low), capped
# at availability and topped up from the largest agents. `censored` yields the
# per-PR censored_30d flags matching that linker's follow-up definition.
# (The superseded "commit-union" mode -- the first, looser draw that was never
# labelled -- was removed 2026-07 and is not part of this package.)
MODES = {
    "net-diff": {
        "dir": DATA / "net_diff",
        "files_cte": "SELECT pr_id, filename FROM pr_net_files WHERE NOT is_boilerplate",
        "alloc": {
            ("Claude_Code", "high"): 2, ("Cursor", "high"): 6,
            ("Copilot", "high"):     8, ("Devin", "high"):  9,
            ("Claude_Code", "low"):  1, ("Cursor", "low"):  6,
            ("Copilot", "low"):      8, ("Devin", "low"):   10,
        },
        "censored": _censored_net_sub,
        "blank_templates": False,
    },
}


def stage_sample(mode: dict) -> None:
    from prfix.common.loaders.base import connect
    from prfix.rq4_signals import per_pr_signals as pps

    con = connect(read_only=True)
    pairs = con.execute(PAIR_SQL.format(files_cte=mode["files_cte"])).df()

    # One representative Q per P: earliest follow-up, then most shared files.
    pairs = (pairs.sort_values(["p_id", "day_gap", "n_shared"], ascending=[True, True, False])
                  .groupby("p_id", as_index=False).first())

    # Touch group: per-agent median n_human over uncensored non-Codex PRs.
    sig = pps.compute()[["pr_id", "agent", "n_human_commits"]]
    sig = sig.merge(mode["censored"](), on="pr_id", how="left")
    unc = sig[(sig.agent.isin(AGENTS_NONCODEX)) & (sig.censored_30d == 0)].copy()
    med = unc.groupby("agent")["n_human_commits"].median()
    unc["touch_group"] = unc.apply(
        lambda r: "high" if r.n_human_commits > med[r.agent] else "low", axis=1)

    cand = unc.merge(pairs.drop(columns=["agent"]), left_on="pr_id", right_on="p_id", how="inner")

    rng = np.random.RandomState(SEED)
    picks = []
    for (ag, grp), k in mode["alloc"].items():
        sub = cand[(cand.agent == ag) & (cand.touch_group == grp)]
        picks.append(sub.sample(n=min(k, len(sub)), random_state=rng))
    sample = pd.concat(picks, ignore_index=True)
    for grp in ("high", "low"):
        have = (sample.touch_group == grp).sum()
        if have < 25:
            pool = cand[(cand.touch_group == grp) & (~cand.pr_id.isin(sample.pr_id))]
            picks.append(pool.sample(n=min(25 - have, len(pool)), random_state=rng))
    sample = pd.concat(picks, ignore_index=True).drop_duplicates("pr_id")

    prmeta = con.execute(
        "SELECT id, number, title, body, html_url FROM pull_request "
        "UNION ALL SELECT id, number, title, body, html_url FROM human_pull_request").df()
    ttype = con.execute(
        "SELECT id, type FROM pr_task_type UNION ALL SELECT id, type FROM human_pr_task_type").df()
    con.close()
    meta = prmeta.set_index("id")
    typ = ttype.set_index("id")["type"]

    def s(x) -> str:
        return x if isinstance(x, str) else ""

    rows = []
    for i, r in enumerate(sample.itertuples(), 1):
        p, q = meta.loc[r.p_id], meta.loc[r.q_id]
        pnum = str(int(p.number)) if pd.notna(p.number) else ""
        qbody = s(q.body)
        refs = bool(pnum) and ((f"#{pnum}" in qbody) or (s(p.html_url) in qbody))
        rows.append({
            "pair_id": i, "agent": r.agent, "touch_group": r.touch_group,
            "day_gap": int(r.day_gap), "n_shared_files": int(r.n_shared),
            "p_id": r.p_id, "p_url": s(p.html_url), "p_title": s(p.title)[:140],
            "q_id": r.q_id, "q_url": s(q.html_url), "q_title": s(q.title)[:140],
            "q_type": typ.get(r.q_id, "(none)"), "q_references_p": refs,
            "shared_files": s(r.shared_files)[:300],
        })
    master = pd.DataFrame(rows)
    mode["dir"].mkdir(parents=True, exist_ok=True)
    master.to_csv(mode["dir"] / "master.csv", index=False)

    if mode["blank_templates"]:
        tmpl = HERE / "labels" / "humans" / "template"
        tmpl.mkdir(parents=True, exist_ok=True)
        cols = ["pair_id", "agent", "p_url", "p_title", "q_url", "q_title", "q_type",
                "shared_files", "day_gap", "q_references_p"]
        for who in ("A", "B"):
            sheet = master[cols].copy()
            sheet["label"] = ""
            sheet["reason"] = ""
            sheet.to_csv(tmpl / f"rq1_followup_validation_annotator{who}.csv", index=False)

    print(f"sampled {len(master)} pairs | agents: {master.agent.value_counts().to_dict()}")
    print(f"touch: {master.touch_group.value_counts().to_dict()} | "
          f"q_references_p: {int(master.q_references_p.sum())}/{len(master)}")
    print(f"wrote {(mode['dir'] / 'master.csv').relative_to(HERE)}")


_URL = re.compile(r"github\.com/([^/]+)/([^/]+)/pull/(\d+)")
_cache: dict[str, list | None] = {}


def fetch_files(url: str) -> list | None:
    """Net-diff files (with patches) for a PR URL; None if unavailable."""
    if url in _cache:
        return _cache[url]
    m = _URL.search(url or "")
    if not m:
        _cache[url] = None
        return None
    owner, repo, num = m.groups()
    try:
        out = subprocess.run(
            ["gh", "api", "--paginate", "--slurp",
             f"repos/{owner}/{repo}/pulls/{num}/files"],
            capture_output=True, text=True, timeout=60, check=True).stdout
        pages = json.loads(out)
        files = [f for pg in pages for f in pg] if pages and isinstance(pages[0], list) else pages
    except (subprocess.CalledProcessError, json.JSONDecodeError, subprocess.TimeoutExpired):
        files = None
    _cache[url] = files
    return files


def anchor(url: str, filename: str) -> str:
    """GitHub Files-changed deep-link, anchored to a file (sha256 of path)."""
    h = hashlib.sha256(filename.encode()).hexdigest()
    return f"{url}/files#diff-{h}"


def stage_enrich(mode: dict) -> None:
    from prfix.rq1_fix_rates.boilerplate import is_boilerplate

    out_path = mode["dir"] / "pairs_enriched.json"
    m = pd.read_csv(mode["dir"] / "master.csv")
    out_rows, surv = [], {"net_any": 0, "net_substantive": 0, "fetch_ok": 0}

    for r in m.itertuples():
        pf, qf = fetch_files(r.p_url), fetch_files(r.q_url)
        rec = {
            "pair_id": int(r.pair_id), "agent": r.agent, "day_gap": int(r.day_gap),
            "q_type": r.q_type, "q_references_p": bool(getattr(r, "q_references_p", False)),
            "p_url": r.p_url, "p_title": r.p_title, "q_url": r.q_url, "q_title": r.q_title,
            "n_shared_commitunion": int(r.n_shared_files),
            "fetch_ok": pf is not None and qf is not None,
        }
        if pf is not None and qf is not None:
            surv["fetch_ok"] += 1
            p_names = {f["filename"] for f in pf}
            q_patch = {f["filename"]: (f.get("patch") or "") for f in qf}
            net_shared = sorted(p_names & set(q_patch))
            files = [{
                "name": fn,
                "boilerplate": is_boilerplate(fn),
                "q_patch": q_patch[fn][:6000],
                "q_link": anchor(r.q_url, fn),
                "p_link": anchor(r.p_url, fn),
            } for fn in net_shared]
            rec["shared"] = files
            rec["n_net_shared"] = len(files)
            rec["n_net_substantive"] = sum(1 for f in files if not f["boilerplate"])
            surv["net_any"] += rec["n_net_shared"] > 0
            surv["net_substantive"] += rec["n_net_substantive"] > 0
        else:
            rec["shared"], rec["n_net_shared"], rec["n_net_substantive"] = [], None, None
        out_rows.append(rec)

    out_path.write_text(json.dumps(out_rows, indent=1), encoding="utf-8")

    n = len(out_rows)
    print(f"enriched {n} pairs -> {out_path.relative_to(HERE)}")
    print(f"  fetch ok:                 {surv['fetch_ok']}/{n}")
    print("  --- NET-DIFF tightening preview (of fetched pairs) ---")
    ok = surv["fetch_ok"] or 1
    print(f"  >=1 net-diff shared file:        {surv['net_any']}/{surv['fetch_ok']} "
          f"({100*surv['net_any']/ok:.0f}%)  <- survive net-diff linker")
    print(f"  >=1 SUBSTANTIVE net-diff shared: {surv['net_substantive']}/{surv['fetch_ok']} "
          f"({100*surv['net_substantive']/ok:.0f}%)  <- survive net-diff + denylist")
    print(f"  pruned by net-diff:              {surv['fetch_ok']-surv['net_any']} pairs "
          "(flagged only via per-commit/merge files)")


# ============================================================================
# JUDGE-CSV SAMPLING -- blind human-vs-judge validation draw over a population
# judge CSV. Moved verbatim from llm_judge/sampling.py (flag-suspects dropped).
# ============================================================================
JUDGE_OUTPUT_DIR = HERE / "labels" / "judge"
BLIND_COLS = ["pair_id", "agent", "p_url", "q_url", "label", "reason", "annotator"]
KEY_COLS = ["pair_id", "agent", "p_url", "q_url", "label", "confidence",
            "rationale", "evidence_files", "stratum"]
POSITIVE_LABEL = "Direct fix"
JUDGE_MODES = ("random", "stratified", "enriched")


def _largest_remainder(frac: pd.Series, n: int) -> pd.Series:
    """Allocate n seats across categories in proportion to ``frac`` (sums to n)."""
    raw = frac * n
    take = raw.astype(int)
    for lab in raw.sort_values(key=lambda s: s - s.astype(int), ascending=False).index:
        if take.sum() >= n:
            break
        take[lab] += 1
    return take


def _proportional(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """Sample n rows allocated proportionally across df['label']."""
    if not n:
        return df.iloc[0:0]
    take = _largest_remainder(df["label"].value_counts(normalize=True), n)
    parts = [g.sample(n=int(take[lab]), random_state=seed)
             for lab, g in df.groupby("label") if take[lab]]
    return pd.concat(parts) if parts else df.iloc[0:0]


def _sample_enriched(df: pd.DataFrame, n: int, seed: int, pos_frac: float) -> pd.DataFrame:
    """Oversample Direct-fix positives (pos_frac of n), camouflaged with
    proportionally-drawn non-Direct decoys. Backfills if either pool is short."""
    pos = df[df["label"] == POSITIVE_LABEL]
    neg = df[df["label"] != POSITIVE_LABEL]
    n_pos = min(int(round(n * pos_frac)), len(pos))
    n_neg = min(n - n_pos, len(neg))
    short = n - (n_pos + n_neg)                       # backfill from whichever pool has room
    if short > 0:
        add = min(short, len(pos) - n_pos)
        n_pos += add
        short -= add
    if short > 0:
        n_neg += min(short, len(neg) - n_neg)
    s_pos = pos.sample(n=n_pos, random_state=seed).assign(stratum="positive")
    s_neg = _proportional(neg, n_neg, seed).assign(stratum="decoy")
    return pd.concat([s_pos, s_neg])


def _draw(df: pd.DataFrame, mode: str, n: int, seed: int, pos_frac: float) -> pd.DataFrame:
    if mode == "random":
        out = df.sample(n=n, random_state=seed)
    elif mode == "stratified":
        out = _proportional(df, n, seed)
    elif mode == "enriched":
        out = _sample_enriched(df, n, seed, pos_frac)
    else:
        raise SystemExit(f"unknown --mode {mode!r}; choose from {JUDGE_MODES}")
    if "stratum" not in out.columns:
        out = out.assign(stratum="all")
    # sort by pair_id so neither the judge call nor the stratum leaks via row order
    return out.sort_values("pair_id").reset_index(drop=True)


def _excluded_ids(paths: list[Path]) -> set[int]:
    ids: set[int] = set()
    for p in paths:
        if not p.exists():
            raise SystemExit(f"--exclude: file not found: {p}")
        ids |= {int(x) for x in pd.read_csv(p)["pair_id"]}
    return ids


def _strata_meta(pool: pd.DataFrame, s: pd.DataFrame, mode: str) -> dict:
    """Per-stratum population vs sampled counts + reweighting factor.

    Strata are positive/decoy for enriched (the Direct-fix split), else the
    judge's per-label classes. weight = population / sampled lets a scorer
    reweight an oversampled stratum back to its population share.
    """
    if mode == "enriched":
        pop = {"positive": int((pool["label"] == POSITIVE_LABEL).sum()),
               "decoy": int((pool["label"] != POSITIVE_LABEL).sum())}
        samp = s["stratum"].value_counts().to_dict()
    else:
        pop = pool["label"].value_counts().to_dict()
        samp = s["label"].value_counts().to_dict()
    out = {}
    for k, npop in pop.items():
        ns = int(samp.get(k, 0))
        out[str(k)] = {"population": int(npop), "sampled": ns,
                       "weight": (npop / ns) if ns else None}
    return out


def _provenance_path(p: Path) -> str:
    """``p`` as a package-relative string for provenance metadata, falling
    back to the bare filename when it lives outside the package. Never
    absolute -- these values ship in the public labels/ artifacts."""
    from prfix.config import PROJECT_ROOT
    resolved = Path(p).resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return resolved.name


def stage_judge_sample(args: argparse.Namespace) -> None:
    from prfix.config import JUDGE_DIR

    pop_csv = args.pop_csv or (
        JUDGE_DIR / "rq1_llm_judge_pop_anthropic_claude-opus-4-8_low.csv")
    JUDGE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pool = pd.read_csv(pop_csv)
    pool = pool[pool["label"] != "ERROR"]            # never ask a human to validate a failed call
    avail = pool
    if args.exclude:
        drop = _excluded_ids(list(args.exclude))
        avail = pool[~pool["pair_id"].isin(drop)]
        print(f"excluded {len(drop)} previously-sampled pairs "
              f"({len(pool)} -> {len(avail)} available)")
    if args.n > len(avail):
        raise SystemExit(f"--n {args.n} > {len(avail)} judged pairs available")

    s = _draw(avail, args.mode, args.n, args.seed, args.pos_frac)
    tag = f"_{args.tag}" if getattr(args, "tag", None) else ""
    stem = f"rq1_judge_sample{args.n}_{args.mode}{tag}"

    blind = s.copy()
    blind["label"] = ""          # human fills this, blind to the judge
    blind["reason"] = ""
    blind["annotator"] = args.annotator
    who = (args.annotator or "TODO").replace(" ", "_")
    blind_path = JUDGE_OUTPUT_DIR / f"{stem}_blind_annotator_{who}.csv"
    blind[BLIND_COLS].to_csv(blind_path, index=False)

    key_path = JUDGE_OUTPUT_DIR / f"{stem}_model_key.csv"
    s[KEY_COLS].to_csv(key_path, index=False)

    # population strata computed over the FULL judged pool (not the disjoint remainder)
    meta = {
        "mode": args.mode, "n": int(len(s)), "seed": args.seed,
        "pos_frac": args.pos_frac if args.mode == "enriched" else None,
        "pop_csv": _provenance_path(pop_csv),
        "population_total_nonerror": int(len(pool)),
        "population_label_counts": {str(k): int(v)
                                    for k, v in pool["label"].value_counts().items()},
        "strata": _strata_meta(pool, s, args.mode),
    }
    meta_path = JUDGE_OUTPUT_DIR / f"{stem}_sample_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"mode={args.mode}  sampled {len(s)} of {len(avail)} available "
          f"(seed={args.seed})")
    if args.mode == "enriched":
        print(f"  pos-frac={args.pos_frac}  (blind sheet is sorted by pair_id; "
              f"stratum is held in the model key + meta, not the blind sheet)")
    print("  judge label distribution in sample:")
    for lab, cnt in s["label"].value_counts().items():
        print(f"    {lab:<24} {cnt}")
    print("  strata (population -> sampled, weight):")
    for k, v in meta["strata"].items():
        w = f"{v['weight']:.1f}" if v["weight"] is not None else "-"
        print(f"    {k:<24} {v['population']:>5} -> {v['sampled']:<3} (w={w})")
    print(f"  blind human sheet -> {blind_path}")
    print(f"  model key (compare) -> {key_path}")
    print(f"  sample meta (reweight) -> {meta_path}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Sample RQ1 validation pairs (gold detector task or judge validation)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("gold",
                       help="sample + enrich the gold detector-validation pairs from the DB")
    g.add_argument("--linker", choices=sorted(MODES), default="net-diff",
                   help="how 'sharing a file' is defined (default: net-diff, the paper's)")
    g.add_argument("--stage", choices=["sample", "enrich", "all"], default="all",
                   help="sample needs the project DB; enrich needs `gh` (default: all)")

    j = sub.add_parser("judge",
                       help="draw a blind human-vs-judge sample from a population judge CSV")
    j.add_argument("--mode", choices=JUDGE_MODES, default="random",
                   help="random (uniform) | stratified (proportional 3-way) | "
                        "enriched (positive-enriched, camouflaged blind)")
    j.add_argument("--pop-csv", type=Path, default=None,
                   help="population LLM-judge CSV to sample from "
                        "(default: data/judge/"
                        "rq1_llm_judge_pop_anthropic_claude-opus-4-8_low.csv)")
    j.add_argument("--n", type=int, default=50, help="number of pairs (default 50)")
    j.add_argument("--seed", type=int, default=42, help="RNG seed (reproducible sample)")
    j.add_argument("--pos-frac", type=float, default=0.5,
                   help="enriched mode: fraction of the sample drawn from the judge's "
                        "Direct-fix calls (rest = decoys). Default 0.5")
    j.add_argument("--exclude", type=Path, nargs="*", default=[],
                   help="annotator/blind sheets whose pair_ids to exclude (disjoint draw)")
    j.add_argument("--annotator", default="", help="prefill the annotator column / filename")
    j.add_argument("--tag", default="",
                   help="suffix for the output stem (e.g. humanP) so a new sample "
                        "never collides with earlier ones")

    args = ap.parse_args()
    if args.cmd == "gold":
        mode = MODES[args.linker]
        if args.stage in ("sample", "all"):
            stage_sample(mode)
        if args.stage in ("enrich", "all"):
            stage_enrich(mode)
    else:
        stage_judge_sample(args)


if __name__ == "__main__":
    main()
