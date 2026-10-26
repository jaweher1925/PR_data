"""Multi-rater LLM-judge over the RQ1 net-diff population (resumable).

Runs the configured raters (``provider:model[:effort]`` specs, from ``judge.yaml``
or ``--raters``) over the population built by ``build_cases.py`` with prompt =
RUBRIC + LEARNED INSIGHTS + worked-example anchors. ``judge.yaml`` defines both
the net-diff set and how to judge it; the cases
file is derived from ``patch_cap``/``judgeable_only`` so builder and runner agree.
Each rater streams to an append-only checkpoint and skips pair_ids already there on
restart, so a killed run resumes with no duplicate calls.

Usage:
    uv run python -m prfix.rq1_fix_rates.judge.run_judge \
        --config src/prfix/rq1_fix_rates/judge/judge.yaml
    # replay the pinned checkpoints (no API key); one run per cohort:
    uv run python -m prfix.rq1_fix_rates.judge.run_judge \
        --config src/prfix/rq1_fix_rates/judge/judge.yaml --consolidate-only
    uv run python -m prfix.rq1_fix_rates.judge.run_judge \
        --config src/prfix/rq1_fix_rates/judge/judge.yaml --consolidate-only \
        --p-side human
"""
from __future__ import annotations

import argparse
import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from rich.console import Console
from rich.table import Table

from prfix.config import JUDGE_DIR, ensure_dirs
from prfix.rq1_fix_rates.judge import build_cases as build_mod
from prfix.rq1_fix_rates.judge import engine as engine_mod
from prfix.rq1_fix_rates.judge.prompt import (
    build_system_prompt,
    coerce_label,
    load_insights,
    pick_exemplars,
)

console = Console()

_PREVIEW_IDS = 10  # how many ERROR pair ids to name before "..."
_MIN_RATERS_FOR_KAPPA = 2


def _ident(provider: str, model: str, effort: str | None) -> str:
    s = f"{provider}_{engine_mod.slug(model)}"
    return f"{s}_{effort}" if effort else s


def _ckpt_path(provider: str, model: str, mode: str, effort: str | None = None) -> Path:
    return JUDGE_DIR / f"judge_{mode}_{_ident(provider, model, effort)}.jsonl"


def _csv_path(provider: str, model: str, mode: str, effort: str | None = None) -> Path:
    return JUDGE_DIR / f"rq1_llm_judge_{mode}_{_ident(provider, model, effort)}.csv"


def _load_ckpt(path: Path) -> dict[int, dict]:
    done: dict[int, dict] = {}
    if not path.exists():
        return done
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
            done[int(rec["pair_id"])] = rec
        except (json.JSONDecodeError, KeyError, ValueError):
            continue  # torn last line from a hard kill -> re-judge it
    return done


def _rewrite_ckpt(path: Path, done: dict[int, dict]) -> None:
    """Atomically rewrite the checkpoint to one line per pair in ``done`` (sorted,
    deduped). --retry-errors uses this to physically drop old ERROR rows before
    re-judging, so a successful retry leaves no stale record behind."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as f:
        for pid in sorted(done):
            f.write(json.dumps(done[pid]) + "\n")
    tmp.replace(path)  # atomic on POSIX


def _judge_one(rater, case: dict, system_prompt: str) -> dict:
    try:
        j = rater.judge_case(case, system_prompt)
    except Exception as exc:  # noqa: BLE001 -- record and continue
        j = {"label": "ERROR", "confidence": 0.0,
             "rationale": f"{type(exc).__name__}: {exc}", "evidence_files": []}
    return {"pair_id": int(case["pair_id"]), "label": j.get("label", "ERROR"),
            "confidence": j.get("confidence"), "rationale": j.get("rationale", ""),
            "evidence_files": _as_list(j.get("evidence_files"))}


def _as_list(v) -> list:
    """The tool schema declares evidence_files as an array, but the model
    sometimes returns a bare string; joining that would scatter it
    character-by-character into the CSV. Any other shape (dict, number) is
    stringified rather than iterated, so a surprising payload cannot silently
    become a list of dict keys or abort a live run."""
    if not v:
        return []
    if isinstance(v, str):
        return [v]
    if isinstance(v, (list, tuple)):
        return [str(x) for x in v]
    return [str(v)]


def _run_rater(rater, cases: list[dict], system_prompt: str, ckpt: Path,
               concurrency: int, retry_errors: bool = False) -> dict[int, dict]:
    done = _load_ckpt(ckpt)
    # An ERROR row is "in the checkpoint" so a normal resume skips it. --retry-errors
    # drops those pairs from `done` and rewrites the file to delete the old ERROR rows
    # so they re-judge, leaving only the good record on success.
    if retry_errors:
        err_ids = [pid for pid, rec in done.items() if rec.get("label") == "ERROR"]
        for pid in err_ids:
            del done[pid]
        if err_ids:
            _rewrite_ckpt(ckpt, done)
            preview = (", ".join(map(str, err_ids[:_PREVIEW_IDS]))
                       + ("..." if len(err_ids) > _PREVIEW_IDS else ""))
            console.print(f"[yellow]retry-errors: re-judging {len(err_ids)} ERROR "
                          f"pair(s): {preview}[/yellow]")
    todo = [c for c in cases if int(c["pair_id"]) not in done]
    tag = f"{rater.provider}:{rater.model}"
    console.print(f"[bold]{tag}[/bold]: {len(done)} done, {len(todo)} remaining "
                  f"(of {len(cases)})")
    if not todo:
        return done

    lock = threading.Lock()
    fh = ckpt.open("a")

    def record(rec: dict) -> None:
        with lock:
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            done[rec["pair_id"]] = rec

    # Warm up one case synchronously so the cached system prompt is written before
    # the pool fans out (avoids a cache stampede).
    record(_judge_one(rater, todo[0], system_prompt))
    rest = todo[1:]
    n = len(rest)
    workers = max(1, min(concurrency, n)) if n else 1
    with console.status(f"[bold]{tag}[/bold] judging (1/{len(todo)})...") as status:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futs = [pool.submit(_judge_one, rater, c, system_prompt) for c in rest]
            for i, fut in enumerate(as_completed(futs), 2):
                record(fut.result())
                status.update(f"[bold]{tag}[/bold] judging ({i}/{len(todo)})...")
    fh.close()
    return done


def _consolidate(cases: list[dict], preds: dict[int, dict], provider: str,
                 model: str, mode: str,
                 effort: str | None = None) -> pd.DataFrame:
    by_id = {int(c["pair_id"]): c for c in cases}
    rows = []
    for pid in sorted(preds):
        c = by_id.get(pid, {})
        p = preds[pid]
        # Normalise to the enum here too, so --consolidate-only retro-fixes an
        # off-enum value on disk without new API calls. ERROR stays ERROR.
        raw = p["label"]
        label = raw if raw == "ERROR" else (coerce_label(raw) or "ERROR")
        row = {
            "pair_id": pid, "provider": provider, "model": model,
            "agent": c.get("agent", ""),
            "q_is_human": bool(c.get("q_is_human", False)),
            "label": label, "confidence": p.get("confidence"),
            "n_shared": c.get("n_shared"), "context_complete": c.get("context_complete"),
            "rationale": p.get("rationale", ""),
            "evidence_files": ";".join(_as_list(p.get("evidence_files"))),
            "p_url": c.get("p_url", ""), "q_url": c.get("q_url", ""),
        }
        rows.append(row)
    df = pd.DataFrame(rows)
    out = _csv_path(provider, model, mode, effort)
    df.to_csv(out, index=False)
    console.log(f"[green]wrote[/green] {out}  ({len(df)} rows)")
    return df


def _label_dist(df: pd.DataFrame, tag: str) -> None:
    vc = df["label"].value_counts()
    t = Table(title=f"label distribution  ({tag}, N={len(df)})")
    t.add_column("label")
    t.add_column("count", justify="right")
    t.add_column("%", justify="right")
    for lab, cnt in vc.items():
        t.add_row(str(lab), str(cnt), f"{cnt/len(df)*100:.1f}%")
    console.print(t)


def _agreement(dfs: dict[str, pd.DataFrame]) -> None:
    """Model-model kappa between the first two raters, on the pairs both judged."""
    if len(dfs) < _MIN_RATERS_FOR_KAPPA:
        return
    (ta, da), (tb, db) = list(dfs.items())[:2]
    a = da.set_index("pair_id")["label"]
    b = db.set_index("pair_id")["label"]
    common = sorted(set(a.index) & set(b.index))
    common = [i for i in common if a[i] != "ERROR" and b[i] != "ERROR"]
    la = [a[i] for i in common]
    lb = [b[i] for i in common]
    k3 = engine_mod._kappa(la, lb)
    ks = engine_mod._kappa([engine_mod._binary(x, "strict") for x in la],
                          [engine_mod._binary(x, "strict") for x in lb])
    kl = engine_mod._kappa([engine_mod._binary(x, "lenient") for x in la],
                          [engine_mod._binary(x, "lenient") for x in lb])
    agree = sum(x == y for x, y in zip(la, lb, strict=False))
    console.rule(f"model-model agreement  ({ta} vs {tb}, N={len(common)})")
    console.print(f"  raw agreement = {agree}/{len(common)} ({agree/len(common)*100:.1f}%)")
    console.print(f"  Cohen kappa: 3-way=[bold]{k3:.3f}[/bold]  "
                  f"strict(is-fix)={ks:.3f}  lenient={kl:.3f}")
    # Disagreements -> seeds for the human audit.
    dis = [(i, a[i], b[i]) for i in common if a[i] != b[i]]
    console.print(f"  disagreements: {len(dis)}")
    for i, x, y in dis[:12]:
        console.print(f"    pair {i}: {ta}=[cyan]{x}[/cyan]  {tb}=[magenta]{y}[/magenta]")


def _population_cases(patch_cap: int, judgeable_only: bool, auto_build: bool,
                      cases_override: str | None, rebuild: bool = False,
                      p_side: str = "agent") -> tuple[list[dict], Path]:
    """Resolve the net-diff cases file (derived from build params, or ``cases_override``).
    Builds it when missing (and ``auto_build``), or unconditionally when ``rebuild`` --
    so a new build-time signal is picked up without hand-running the builder.
    Returns (cases, resolved path); the caller derives the run identity from the path."""
    path = (Path(cases_override) if cases_override
            else build_mod.population_cases_path(patch_cap, judgeable_only, p_side))
    if rebuild and cases_override:
        console.print("[yellow]--rebuild ignored: an explicit --cases path was given; "
                      "build that file yourself or drop --cases.[/yellow]")
    force = rebuild and not cases_override
    if force or not path.exists():
        if not force and not auto_build:
            raise SystemExit(
                f"cases file not found: {path}\n"
                f"  build it: uv run python -m prfix.rq1_fix_rates.judge.build_cases "
                f"--mode population --patch-cap {patch_cap}"
                f"{' --judgeable-only' if judgeable_only else ''}"
                f"{' --p-side human' if p_side == 'human' else ''}\n"
                f"  or set auto_build: true in the config.")
        why = "rebuild requested" if force else "cases file missing"
        console.print(f"[dim]{why} -> building {path.name} "
                      f"(patch_cap={patch_cap:,}, judgeable_only={judgeable_only})...[/dim]")
        build_mod.build_population(patch_cap=patch_cap, judgeable_only=judgeable_only,
                                   p_side=p_side)
    return json.loads(path.read_text()), path


def _cases_are_human_p(cases: list[dict], cases_path: Path) -> bool:
    """Which P cohort these cases belong to -- the checkpoint/CSV identity.

    Taken from the cases themselves (the ``p_is_human`` marker build_cases
    writes), falling back to the ``_humanP`` file-name convention for case files
    built before the marker existed. Never from ``p_side``: pair_ids overlap
    between the two cohorts, so a run must only ever touch the checkpoint
    belonging to the cases it is actually judging."""
    marked = {bool(c["p_is_human"]) for c in cases if "p_is_human" in c}
    if len(marked) > 1:
        raise SystemExit(f"cases file {cases_path.name} mixes agent-P and human-P "
                         f"cases; rebuild it for a single P side.")
    return marked.pop() if marked else "_humanP" in cases_path.stem


def run(rater_specs: list[str], concurrency: int,
        limit: int | None, restart: bool, consolidate_only: bool, dry_run: bool,
        patch_cap: int, judgeable_only: bool, auto_build: bool,
        cases_override: str | None = None, rebuild: bool = False,
        retry_errors: bool = False, p_side: str = "agent") -> None:
    if not rater_specs:
        raise SystemExit(
            "no raters configured -- set `raters:` in the config (e.g. "
            "src/prfix/rq1_fix_rates/judge/judge.yaml) or pass "
            "--raters provider:model[:effort].")
    ensure_dirs()

    # System prompt = RUBRIC + LEARNED INSIGHTS (frozen in insights.md) + worked-example
    # anchors. Anchors + exemplar cases come from the gold case file; insights are
    # loaded by build_system_prompt from the frozen artifact.
    validated = engine_mod.load_cases()
    exemplar_ids, _ = pick_exemplars(validated)
    system_prompt = build_system_prompt(validated, exemplar_ids)
    if not load_insights():
        console.print("[yellow]no insights.md yet -- the prompt is RUBRIC + anchors "
                      "only. Distil it first: uv run python -m "
                      "prfix.rq1_fix_rates.judge.prompt insights build[/yellow]")

    cases, cases_path = _population_cases(patch_cap, judgeable_only, auto_build,
                                          cases_override, rebuild, p_side)
    if limit:
        cases = cases[:limit]
    is_human_cases = _cases_are_human_p(cases, cases_path)
    mode = "pop_humanP" if is_human_cases else "pop"
    if is_human_cases != (p_side == "human"):
        raise SystemExit(
            f"cases file {cases_path.name} holds "
            f"{'human' if is_human_cases else 'agent'}-P cases but p_side is "
            f"{p_side!r}. Run identity comes from the cases, so this run would write "
            f"the {mode} checkpoint/CSV from a cohort you did not ask for. Pass the "
            f"matching --p-side, or the cases file for {p_side}-P.")
    console.print(f"  cases: {len(cases)} | system prompt ~{len(system_prompt)//4:,} tokens "
                  f"| population")

    if dry_run:
        console.print(f"[bold]raters:[/bold] {rater_specs}")
        from prfix.rq1_fix_rates.judge.prompt import render_case  # noqa: PLC0415
        console.rule("sample case (first)")
        console.print(render_case(cases[0])[:1200])
        return

    dfs: dict[str, pd.DataFrame] = {}
    for spec in rater_specs:
        rater = None
        try:
            if consolidate_only:
                provider, model, effort = engine_mod.parse_spec(spec)
            else:
                rater = engine_mod.make_rater(spec)
                provider, model = rater.provider, rater.model
                effort = getattr(rater, "effort", None)
        except SystemExit as e:
            console.print(f"[yellow]skip {spec}: {e}[/yellow]")
            continue
        ckpt = _ckpt_path(provider, model, mode, effort)
        if restart and ckpt.exists():
            ckpt.unlink()
        if not consolidate_only:
            preds = _run_rater(rater, cases, system_prompt, ckpt, concurrency, retry_errors)
        else:
            preds = _load_ckpt(ckpt)
        if not preds:
            console.print(f"[yellow]{spec}: no predictions on disk[/yellow]")
            continue
        tag = f"{provider}:{model}" + (f":{effort}" if effort else "")
        df = _consolidate(cases, preds, provider, model, mode, effort)
        n_err = int((df["label"] == "ERROR").sum())
        if n_err:
            console.print(f"[yellow]{tag}: {n_err} ERROR rows (excluded from stats)[/yellow]")
        _label_dist(df[df["label"] != "ERROR"], tag)
        dfs[tag] = df

    _agreement(dfs)


# Canonical settings + defaults. A config may set any of these (dashes or
# underscores); CLI flags win; unlisted keys are rejected as typos.
DEFAULTS = {
    "raters": [],             # provider:model[:effort] specs; REQUIRED (config or --raters)
    # net-diff build params (the cases file is derived from these)
    "patch_cap": build_mod.POP_PATCH_CAP,
    "judgeable_only": True,
    "auto_build": True,
    "cases": None,            # explicit cases path override; None => derive from patch_cap
    "p_side": "agent",        # P cohort: agent (published run) | human (incidence baseline;
                              # own cases file + checkpoint/CSV suffix pop_humanP)
    "rebuild": False,         # force-rebuild the population cases file before judging
    # judge run params
    "concurrency": 8,
    "limit": None,
    "restart": False,
    "retry_errors": False,    # re-judge only the pairs whose checkpoint row is ERROR
    "consolidate_only": False,
    "dry_run": False,
}


def load_config(path: Path) -> dict:
    """Load a YAML / JSON / TOML config into a dict of known keys."""
    if not path.exists():
        raise SystemExit(f"--config: file not found: {path}")
    text = path.read_text()
    suf = path.suffix.lower()
    if suf in (".yaml", ".yml"):
        import yaml  # noqa: PLC0415
        data = yaml.safe_load(text) or {}
    elif suf == ".toml":
        import tomllib  # noqa: PLC0415
        data = tomllib.loads(text)
    elif suf == ".json":
        data = json.loads(text)
    else:
        raise SystemExit(f"--config: unsupported extension {suf!r} (use .yaml/.json/.toml)")
    if not isinstance(data, dict):
        raise SystemExit(f"--config: {path} must be a mapping at the top level")
    out = {}
    for k, v in data.items():
        kk = str(k).replace("-", "_")
        if kk not in DEFAULTS:
            raise SystemExit(f"--config: unknown key {k!r}; valid keys: {sorted(DEFAULTS)}")
        out[kk] = v
    return out


def _as_rater_list(val) -> list[str]:
    """Accept a YAML/JSON list or a comma-separated string of rater specs."""
    if isinstance(val, str):
        return [s.strip() for s in val.split(",") if s.strip()]
    return [str(s).strip() for s in val if str(s).strip()]


if __name__ == "__main__":
    import argparse as _ap  # noqa: PLC0415 -- local alias for SUPPRESS
    ap = argparse.ArgumentParser(description="Two-rater LLM-judge over the population")
    ap.add_argument("--config", default=None,
                    help="YAML/JSON/TOML file with any of the settings below. Values "
                         "override the built-in defaults; explicit CLI flags still "
                         "override the config.")
    # All other args default to SUPPRESS so only explicitly-passed flags land in
    # the namespace, keeping config-vs-CLI precedence unambiguous.
    ap.add_argument("--raters", default=_ap.SUPPRESS,
                    help="comma list of provider:model[:effort] specs (required "
                         "here or in the config; the model id is passed through "
                         "to your deployment as-is). effort = openai "
                         "reasoning_effort (none|minimal|low|medium|high|xhigh) "
                         "or anthropic effort; e.g. anthropic:claude-opus-4-8:low. "
                         "Each effort writes its own checkpoint/CSV so you can "
                         "A/B them.")
    ap.add_argument("--patch-cap", type=int, default=_ap.SUPPRESS,
                    help="chars per file patch; also selects the derived cases file")
    ap.add_argument("--judgeable-only", action="store_true", default=_ap.SUPPRESS,
                    help="net-diff set = only pairs with a real overlap (n_shared>0)")
    ap.add_argument("--no-auto-build", dest="auto_build", action="store_false",
                    default=_ap.SUPPRESS,
                    help="fail instead of building the cases file if it is missing")
    ap.add_argument("--cases", default=_ap.SUPPRESS,
                    help="explicit cases JSON path (overrides the derived path)")
    ap.add_argument("--rebuild", action="store_true", default=_ap.SUPPRESS,
                    help="force-rebuild the population cases file before judging "
                         "(picks up new build-time signals, e.g. intervening-PR "
                         "counts); population mode only")
    ap.add_argument("--concurrency", type=int, default=_ap.SUPPRESS)
    ap.add_argument("--p-side", choices=["agent", "human"], default=_ap.SUPPRESS,
                    help="P cohort: agent (published run, default) or human "
                         "(incidence baseline; own cases file + pop_humanP "
                         "checkpoint/CSV, never touches the agent run)")
    ap.add_argument("--limit", type=int, default=_ap.SUPPRESS,
                    help="first N cases (smoke test)")
    ap.add_argument("--restart", action="store_true", default=_ap.SUPPRESS,
                    help="ignore + delete checkpoints")
    ap.add_argument("--retry-errors", action="store_true", default=_ap.SUPPRESS,
                    help="re-judge only the pairs whose checkpoint row is ERROR "
                         "(e.g. earlier timeouts); keeps all other results")
    ap.add_argument("--consolidate-only", action="store_true", default=_ap.SUPPRESS,
                    help="rebuild CSVs + report from checkpoints; no API calls")
    ap.add_argument("--dry-run", action="store_true", default=_ap.SUPPRESS,
                    help="no API; print sizes + sample")
    ns = vars(ap.parse_args())

    cfg = dict(DEFAULTS)                                   # 1. defaults
    config_path = ns.pop("config", None)
    if config_path:
        cfg.update(load_config(Path(config_path)))        # 2. config file
    cfg.update(ns)                                         # 3. explicit CLI flags win

    run(_as_rater_list(cfg["raters"]), cfg["concurrency"], cfg["limit"],
        cfg["restart"], cfg["consolidate_only"], cfg["dry_run"],
        cfg["patch_cap"], cfg["judgeable_only"], cfg["auto_build"], cfg["cases"],
        cfg["rebuild"], cfg["retry_errors"], cfg["p_side"])
