"""Build RQ1 LLM-judge case files -- the exact context the judge may see.

One builder, two modes: ``gold`` (the fixed 50-pair validation set) and
``population`` (the config-sized net-diff pool; filename derived from ``patch_cap``
so build and ``run_judge`` never drift). A case carries the P/Q descriptions,
commit messages, net-diff shared files, and (gold ``--full-diff``) non-shared files.
"""
from __future__ import annotations

import argparse
import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import duckdb
import pandas as pd
from rich.console import Console

from prfix.config import DUCKDB_PATH, JUDGE_DIR, PROJECT_ROOT, ensure_dirs
from prfix.rq1_fix_rates.boilerplate import is_boilerplate
from prfix.rq1_fix_rates.judge import attribution as ba
from prfix.rq1_fix_rates.judge.attribution import is_code
from prfix.rq1_fix_rates.judge.prompt import truncate

console = Console()

HV = PROJECT_ROOT / "human-annotation"          # human-labeling inputs (gold mode)
DEFAULT_CASES = JUDGE_DIR / "rq1_judge_cases.json"
AGREED_CSV = HV / "labels" / "humans" / "rq1_followup_validation_agreed.csv"
# Frozen diff snapshot: reproducible rebuilds; the same diff the annotators saw.
SNAPSHOT_PATH = HV / "data" / "rq1_pairs_diff_snapshot.json"

PATCH_CAP = 6000        # chars per file patch (gold default)
POP_PATCH_CAP = 12000   # chars per file patch (population default = production run)
BODY_CAP = 4000         # chars per PR body
N_COMMITS = 40          # commit messages kept per PR (covers every pair in the set)
COMMIT_CAP = 300        # chars per commit message
MAX_EXTRA_FILES = 12    # cap on non-shared files kept per PR in full-diff mode

_URL = re.compile(r"github\.com/([^/]+)/([^/]+)/pull/(\d+)")
_fetch_cache: dict[str, list | None] = {}
_snap_state: dict = {"data": None, "dirty": False}


def _load_snapshot() -> dict[str, list]:
    """Lazy-load the on-disk diff snapshot (url -> [{filename, patch}, ...])."""
    if _snap_state["data"] is None:
        try:
            _snap_state["data"] = json.loads(SNAPSHOT_PATH.read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            _snap_state["data"] = {}
    return _snap_state["data"]


def _save_snapshot() -> None:
    """Persist the snapshot if any new diffs were fetched this run."""
    if _snap_state["dirty"] and _snap_state["data"] is not None:
        SNAPSHOT_PATH.write_text(
            json.dumps(_snap_state["data"], sort_keys=True), encoding="utf-8")


def _norm(url: str) -> str:
    return (url or "").rstrip("/")


def _agreed_url_pairs() -> set[tuple[str, str]]:
    """``(p_url, q_url)`` of every annotator-agreed gold pair (``label_1`` ==
    ``label_2``). ``build_population`` drops these (they train the prompt), matched
    by URL since gold/population number pairs separately. Empty set if CSV absent."""
    try:
        df = pd.read_csv(AGREED_CSV)
    except FileNotFoundError:
        return set()
    con = df[df["label_1"] == df["label_2"]]
    return {(_norm(r.p_url), _norm(r.q_url)) for r in con.itertuples()}


def fetch_files(url: str) -> list | None:
    """Changed files (filename + patch) for a PR URL, read from the frozen snapshot
    first (so rebuilds are reproducible), else fetched once via ``gh`` and recorded.
    Returns None if the URL is unparseable or the fetch fails; transient failures
    are not recorded, so a later rebuild retries."""
    key = _norm(url)
    if key in _fetch_cache:
        return _fetch_cache[key]
    snap = _load_snapshot()
    if key in snap:
        _fetch_cache[key] = snap[key]
        return snap[key]
    m = _URL.search(url or "")
    if not m:
        _fetch_cache[key] = None
        return None
    owner, repo, num = m.groups()
    # ba.gh_json pauses + retries on a GitHub rate limit rather than crashing.
    pages = ba.gh_json(["api", "--paginate", "--slurp",
                        f"repos/{owner}/{repo}/pulls/{num}/files"], timeout=60)
    if pages is None:
        _fetch_cache[key] = None   # genuine failure / not rate-limited: retried next run
        return None
    try:
        raw = [f for pg in pages for f in pg] if pages and isinstance(pages[0], list) else pages
        files = [{"filename": f["filename"], "patch": f.get("patch") or ""} for f in raw]
    except (KeyError, TypeError):
        _fetch_cache[key] = None
        return None
    snap[key] = files
    _snap_state["dirty"] = True
    _fetch_cache[key] = files
    return files


def _fetch_all_diffs(urls: list[str], concurrency: int) -> None:
    """Fetch (and snapshot) every distinct PR url. Resumable + checkpointed."""
    ba.set_console(console)  # surface rate-limit pauses during the diff fetch
    snap = _load_snapshot()
    todo = [u for u in urls if _norm(u) not in snap]
    done_cached = len(urls) - len(todo)
    console.print(f"  diffs: {len(urls):,} distinct PRs | {done_cached:,} cached | "
                  f"{len(todo):,} to fetch (concurrency={concurrency})")
    if not todo:
        return
    CHUNK = 250  # checkpoint the snapshot every CHUNK fetches
    for start in range(0, len(todo), CHUNK):
        chunk = todo[start:start + CHUNK]
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            futs = {pool.submit(fetch_files, u): u for u in chunk}
            for fut in as_completed(futs):
                try:
                    fut.result()
                except Exception:  # noqa: BLE001 -- transient; not snapshotted, retried later
                    pass
        _save_snapshot()  # checkpoint between chunks (no threads running)
        console.print(f"    fetched {min(start + CHUNK, len(todo)):,}/{len(todo):,} "
                      f"(snapshot checkpointed)")
    console.print("  [green]done fetching[/green]")


def _commit_msgs(con, pr_id) -> list[str]:
    """Up to N_COMMITS commit messages for a PR. ORDER BY sha (pr_commits has no
    timestamp) keeps the subset deterministic across rebuilds."""
    if pr_id is None:
        return []
    df = con.execute(
        "SELECT COALESCE(message,'') m FROM pr_commits WHERE pr_id=? ORDER BY sha LIMIT ?",
        [int(pr_id), N_COMMITS]).df()
    return [truncate(m, COMMIT_CAP) for m in df["m"] if m.strip()]


def _make_case(*, pair_id: int, agent, p_url: str, q_url: str,
               p_title: str, q_title: str, p_body: str, q_body: str,
               p_commits: list[str], q_commits: list[str], day_gap,
               shared: list[dict], q_only: list[dict], p_only: list[dict],
               full_diff: bool, has_p_patch: bool,
               p_id: int | None = None, q_id: int | None = None,
               consensus_label=None, label_1=None, label_2=None,
               reason_note: str = "", q_type: str = "",
               q_references_p: bool = False, n_intervening: int = 0,
               q_is_human: bool = False, p_is_human: bool = False) -> dict:
    """Assemble one case dict in the shape the judge consumes -- the single source
    of the case schema for both modes. ``p_id``/``q_id`` are emitted only when
    provided (population carries them; gold does not)."""
    case = {
        "pair_id": int(pair_id),
        "agent": agent,
        "consensus_label": consensus_label,
        "label_1": label_1,
        "label_2": label_2,
        "reason_note": reason_note,
        "p_url": p_url,
        "q_url": q_url,
        "p_title": p_title,
        "q_title": q_title,
        "p_body": (p_body or "")[:BODY_CAP],
        "q_body": (q_body or "")[:BODY_CAP],
        "p_commits": p_commits,
        "q_commits": q_commits,
        "day_gap": day_gap,
        "n_intervening": n_intervening,
        "q_is_human": bool(q_is_human),
        "p_is_human": bool(p_is_human),
        "q_type": q_type,
        "q_references_p": q_references_p,
        "shared": shared,
        "n_shared": len(shared),
        "n_substantive": sum(1 for s in shared if not s["boilerplate"]),
        "full_diff": full_diff,
        "q_only": q_only,
        "p_only": p_only,
        "n_q_only": len(q_only),
        "n_p_only": len(p_only),
        "context_complete": bool(shared) and bool(has_p_patch),
    }
    if p_id is not None:
        case["p_id"] = int(p_id)
    if q_id is not None:
        case["q_id"] = int(q_id)
    return case


def _pick_extra(patch_map: dict[str, str], exclude: set[str], key: str,
                patch_cap: int) -> list[dict]:
    """Non-shared changed files (substantive first), capped at MAX_EXTRA_FILES."""
    items = [(n, p) for n, p in patch_map.items() if n not in exclude and p]
    items.sort(key=lambda np: (is_boilerplate(np[0]), -len(np[1])))
    return [{"name": n, "boilerplate": is_boilerplate(n), key: truncate(p, patch_cap)}
            for n, p in items[:MAX_EXTRA_FILES]]


def build_gold(full_diff: bool = False, out_path: str | Path | None = None,
               patch_cap: int = PATCH_CAP) -> Path:
    """Build the fixed 50-pair validation set with consensus labels attached, from
    pinned sources (the agreed labels, master metadata, and the cached net-diff the
    annotators labeled against). Returns the output path."""
    ensure_dirs()
    ba.set_console(console)  # surface rate-limit pauses if gold fetches any diffs
    agreed = pd.read_csv(HV / "labels" / "humans" / "rq1_followup_validation_agreed.csv")
    # net_diff/master.csv shares the agreed set's pair_id numbering (verified
    # 50/50 by URL); commit_union/master.csv is the SUPERSEDED first draw with
    # its own numbering, so joining it by pair_id attaches the wrong metadata.
    master = pd.read_csv(HV / "data" / "net_diff" / "master.csv")
    cache = {int(r["pair_id"]): r
             for r in json.loads((HV / "data" / "net_diff" / "pairs_enriched.json").read_text())}

    con = duckdb.connect(str(DUCKDB_PATH), read_only=True)
    con.execute("""CREATE OR REPLACE TEMP VIEW ALLPR AS
        SELECT id, number, repo_id, html_url, COALESCE(body,'') body FROM pull_request
        UNION ALL
        SELECT h.id, h.number, rh.id, h.html_url, COALESCE(h.body,'') body
        FROM human_pull_request h JOIN repository rh ON rh.url=h.repo_url""")
    pr_rows = con.execute("SELECT id, number, repo_id, html_url, body FROM ALLPR").df()
    pr_idx = {_norm(r.html_url): r for r in pr_rows.itertuples()}
    meta = {int(r.pair_id): r for r in master.itertuples()}

    def resolve(url: str):
        return pr_idx.get(_norm(url))

    cases = []
    for r in agreed.itertuples():
        pid = int(r.pair_id)
        P, Q = resolve(r.p_url), resolve(r.q_url)
        c = cache.get(pid, {})

        shared_cached = c.get("shared", []) or []
        # Shared files come from the cached net-diff (never a live fetch), so the
        # judge sees the exact case the humans labeled.
        p_files = fetch_files(r.p_url) if (shared_cached or full_diff) else None
        p_patch = ({f["filename"]: (f.get("patch") or "") for f in p_files}
                   if p_files else {})

        shared = [{
            "name": f["name"],
            "boilerplate": bool(f.get("boilerplate", is_boilerplate(f["name"]))),
            "p_patch": truncate(p_patch.get(f["name"], ""), patch_cap),
            "q_patch": truncate(f.get("q_patch") or "", patch_cap),
        } for f in shared_cached]

        # full-diff: add each PR's non-shared files (a relocated fix may live there).
        q_only, p_only = [], []
        if full_diff:
            q_files = fetch_files(r.q_url)
            q_patch = ({f["filename"]: (f.get("patch") or "") for f in q_files}
                       if q_files else {})
            excl = {f["name"] for f in shared_cached}
            q_only = _pick_extra(q_patch, excl, "q_patch", patch_cap)
            p_only = _pick_extra(p_patch, excl, "p_patch", patch_cap)

        m = meta.get(pid)
        cases.append(_make_case(
            pair_id=pid, agent=r.agent,
            consensus_label=(r.label_1 if r.label_1 == r.label_2 else None),
            label_1=r.label_1, label_2=r.label_2,
            reason_note=(r.reason if isinstance(r.reason, str) else ""),
            p_url=r.p_url, q_url=r.q_url,
            p_title=c.get("p_title", ""), q_title=c.get("q_title", ""),
            p_body=(P.body if P is not None else ""),
            q_body=(Q.body if Q is not None else ""),
            p_commits=_commit_msgs(con, P.id if P is not None else None),
            q_commits=_commit_msgs(con, Q.id if Q is not None else None),
            day_gap=(int(getattr(m, "day_gap", 0) or 0) if m is not None else None),
            q_type=(getattr(m, "q_type", "") if m is not None else ""),
            q_references_p=(bool(getattr(m, "q_references_p", False)) if m is not None else False),
            shared=shared, q_only=q_only, p_only=p_only,
            full_diff=full_diff, has_p_patch=bool(p_patch),
        ))

    _save_snapshot()  # freeze any newly fetched P/Q diffs for reproducible rebuilds
    out = Path(out_path) if out_path else DEFAULT_CASES
    out.write_text(json.dumps(cases, indent=1), encoding="utf-8")
    n_cons = sum(1 for c in cases if c["consensus_label"])
    n_diff = sum(1 for c in cases if c["context_complete"])
    console.print(f"wrote {len(cases)} gold cases -> {out}  (full_diff={full_diff})")
    console.print(f"  consensus-labeled (both raters agree): {n_cons}")
    console.print(f"  with both-side net-diff context:       {n_diff}")
    console.print(f"  diff snapshot:                         {SNAPSHOT_PATH.name} "
                  f"({len(_load_snapshot())} PRs frozen)")
    if full_diff:
        tot_q = sum(len(c["q_only"]) for c in cases)
        tot_p = sum(len(c["p_only"]) for c in cases)
        console.print(f"  extra non-shared files surfaced:       Q={tot_q}, P={tot_p} "
                      f"(capped at {MAX_EXTRA_FILES}/PR)")
    return out


# P = a merged agent PR (repo >=500*); Q = a fix-typed/revert PR in the same repo
# merged 0-30 days later, drawn from BOTH the agent (`pull_request`) and human
# (`human_pull_request`) tables -- agent regressions are often fixed by humans
# (mirrors the RQ1 candidate linker). The Q sources differ only in the pre-gate: agent Q is
# pre-gated on `pr_commit_details` overlap (keeps agent pair_ids stable); human Q has
# no such table, so it is gated only by repo+30d here. The binding shared-file test
# for both is the live gh net-diff (n_shared>0) in build_population.
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
  LEFT JOIN human_pr_task_type th ON th.id=h.id
  WHERE h.merged_at IS NOT NULL AND rh.stars>=500
    AND (th.type='fix' OR h.title LIKE 'Revert %' OR h.title LIKE 'revert %')),
files AS (SELECT DISTINCT pr_id, filename FROM pr_commit_details),
pair_agent AS (
  SELECT p.p_id, p.agent, p.repo_id, q.q_id,
         p.p_merged, q.q_merged,
         DATE_DIFF('day', p.p_merged, q.q_merged) AS day_gap
  FROM p_pop p JOIN q_agent q
    ON q.repo_id=p.repo_id AND q.q_merged>p.p_merged
   AND q.q_merged<=p.p_merged+INTERVAL 30 DAY AND q.q_id<>p.p_id),
pair_human AS (
  SELECT p.p_id, p.agent, p.repo_id, q.q_id,
         p.p_merged, q.q_merged,
         DATE_DIFF('day', p.p_merged, q.q_merged) AS day_gap
  FROM p_pop p JOIN q_human q
    ON q.repo_id=p.repo_id AND q.q_merged>p.p_merged
   AND q.q_merged<=p.p_merged+INTERVAL 30 DAY AND q.q_id<>p.p_id)
-- agent-Q pairs: gated on pr_commit_details overlap (unchanged -> stable pair_ids)
SELECT pa.p_id, pa.agent, pa.q_id, pa.repo_id,
       pa.p_merged, pa.q_merged, pa.day_gap,
       FALSE AS q_is_human,
       COUNT(DISTINCT pf.filename) AS n_shared_commitdetails
FROM pair_agent pa
JOIN files pf ON pf.pr_id=pa.p_id
JOIN files qf ON qf.pr_id=pa.q_id AND qf.filename=pf.filename
GROUP BY pa.p_id, pa.agent, pa.q_id, pa.repo_id,
         pa.p_merged, pa.q_merged, pa.day_gap
UNION ALL
-- human-Q pairs: no human file table, so no pre-gate here; the gh net-diff in
-- build_population (judgeable_only) is the binding shared-file test.
SELECT ph.p_id, ph.agent, ph.q_id, ph.repo_id,
       ph.p_merged, ph.q_merged, ph.day_gap,
       TRUE AS q_is_human,
       CAST(NULL AS BIGINT) AS n_shared_commitdetails
FROM pair_human ph
"""

# Human-P variant for the incidence baseline: P = merged contemporaneous
# human-cohort PRs (same >=500-star repos), Q = fix-typed/revert PRs from both
# tables. No commit-details pre-gate on either Q branch (human P's have no
# pr_commit_details rows to pre-gate against); the binding shared-file test is
# the live gh net-diff in build_population (judgeable_only), same as always.
PAIR_SQL_HUMAN_P = """
WITH p_pop AS (
  SELECT h.id p_id, rh.id repo_id, 'Human' AS agent,
         TRY_CAST(h.merged_at AS TIMESTAMP) p_merged
  FROM human_pull_request h JOIN repository rh ON rh.url=h.repo_url
  WHERE h.merged_at IS NOT NULL AND rh.stars>=500),
q_agent AS (
  SELECT q.id q_id, q.repo_id, TRY_CAST(q.merged_at AS TIMESTAMP) q_merged,
         FALSE AS q_is_human
  FROM pull_request q JOIN repository r ON r.id=q.repo_id LEFT JOIN pr_task_type t ON t.id=q.id
  WHERE q.merged_at IS NOT NULL AND r.stars>=500
    AND (t.type='fix' OR q.title LIKE 'Revert %' OR q.title LIKE 'revert %')),
q_human AS (
  SELECT h.id q_id, rh.id repo_id, TRY_CAST(h.merged_at AS TIMESTAMP) q_merged,
         TRUE AS q_is_human
  FROM human_pull_request h JOIN repository rh ON rh.url=h.repo_url
  LEFT JOIN human_pr_task_type th ON th.id=h.id
  WHERE h.merged_at IS NOT NULL AND rh.stars>=500
    AND (th.type='fix' OR h.title LIKE 'Revert %' OR h.title LIKE 'revert %')),
q_all AS (SELECT * FROM q_agent UNION ALL SELECT * FROM q_human)
SELECT p.p_id, p.agent, q.q_id, p.repo_id,
       p.p_merged, q.q_merged,
       DATE_DIFF('day', p.p_merged, q.q_merged) AS day_gap,
       q.q_is_human,
       CAST(NULL AS BIGINT) AS n_shared_commitdetails
FROM p_pop p JOIN q_all q
  ON q.repo_id=p.repo_id AND q.q_merged>p.p_merged
 AND q.q_merged<=p.p_merged+INTERVAL 30 DAY AND q.q_id<>p.p_id
"""

PAIR_SQLS = {"agent": None, "human": PAIR_SQL_HUMAN_P}  # None -> module PAIR_SQL


def population_cases_path(patch_cap: int = POP_PATCH_CAP,
                          judgeable_only: bool = True,
                          p_side: str = "agent") -> Path:
    """Population cases path, derived from the build params: the patch cap names the
    file (12000 -> ``..._12k``) so builder and ``run_judge`` resolve the same one.
    Non-judgeable builds get an ``_all`` suffix; human-P builds get ``_humanP``."""
    k = f"{patch_cap // 1000}k" if patch_cap % 1000 == 0 else str(patch_cap)
    suffix = "" if judgeable_only else "_all"
    side = "" if p_side == "agent" else "_humanP"
    return JUDGE_DIR / f"rq1_judge_cases_population_{k}{suffix}{side}.json"


def _load_pop_pairs(con, p_side: str = "agent") -> pd.DataFrame:
    pairs = con.execute(PAIR_SQLS[p_side] or PAIR_SQL).df()
    # Metadata index over BOTH PR tables: P (and agent Q) live in `pull_request`,
    # human Q lives in `human_pull_request`. GitHub PR ids are globally unique, so
    # the two id spaces never collide.
    pr = con.execute(
        "SELECT id, html_url, COALESCE(title,'') title, COALESCE(body,'') body "
        "FROM pull_request "
        "UNION ALL "
        "SELECT id, html_url, COALESCE(title,'') title, COALESCE(body,'') body "
        "FROM human_pull_request").df()
    pr_by_id = {int(r.id): r for r in pr.itertuples()}

    def col(ids, attr):
        return [getattr(pr_by_id.get(int(i)), attr, "") if int(i) in pr_by_id else ""
                for i in ids]

    pairs["p_url"] = col(pairs["p_id"], "html_url")
    pairs["q_url"] = col(pairs["q_id"], "html_url")
    pairs["p_title"] = col(pairs["p_id"], "title")
    pairs["q_title"] = col(pairs["q_id"], "title")
    pairs["p_body"] = col(pairs["p_id"], "body")
    pairs["q_body"] = col(pairs["q_id"], "body")
    return pairs


# Every (PR, file) touch with merge time, scoped to the repos in play. Used to count
# OTHER PRs that touched a shared file between P and Q (the intervening-PR signal
# behind prompt RUBRIC #6).
INTERVENING_SQL = """
SELECT pcd.filename, p.repo_id, p.id AS pr_id,
       TRY_CAST(p.merged_at AS TIMESTAMP) AS merged_ts
FROM (SELECT DISTINCT pr_id, filename FROM pr_commit_details) pcd
JOIN pull_request p ON p.id = pcd.pr_id
WHERE p.merged_at IS NOT NULL AND p.repo_id IN ({repos})
"""


def _intervening_index(con, repo_ids: list[int]) -> dict[tuple[int, str], list[tuple]]:
    """``(repo_id, filename) -> [(merged_ts, pr_id), ...]`` for the given repos.
    Built once so per-pair counting is in-memory (no SQL round-trip per pair)."""
    if not repo_ids:
        return {}
    inlist = ",".join(str(int(r)) for r in repo_ids)
    df = con.execute(INTERVENING_SQL.format(repos=inlist)).df()
    idx: dict[tuple[int, str], list[tuple]] = {}
    for r in df.itertuples(index=False):
        if pd.isna(r.merged_ts) or pd.isna(r.repo_id):
            continue
        idx.setdefault((int(r.repo_id), r.filename), []).append((r.merged_ts, int(r.pr_id)))
    return idx


def _count_intervening(idx, repo_id, filename, p_ts, q_ts, exclude: set[int]) -> set[int]:
    """Distinct OTHER PR ids that touched ``filename`` in this repo and merged
    strictly between P and Q. Empty set when the window/timestamps are unusable."""
    entries = idx.get((repo_id, filename))
    if not entries or pd.isna(p_ts) or pd.isna(q_ts):
        return set()
    return {pid for ts, pid in entries if p_ts < ts < q_ts and pid not in exclude}


def _p_sha_index(con, p_ids: list[int]) -> dict[int, set[str]]:
    """``p_id -> set(commit shas)`` from ``pr_commits``, for blame attribution.
    Built once so per-pair lookup is in-memory (P's merge_commit_sha is added
    per-pair from the PR-meta snapshot, to cover squash-merges)."""
    if not p_ids:
        return {}
    inlist = ",".join(str(int(i)) for i in p_ids)
    df = con.execute(f"SELECT pr_id, sha FROM pr_commits WHERE pr_id IN ({inlist})").df()
    idx: dict[int, set[str]] = {}
    for r in df.itertuples(index=False):
        idx.setdefault(int(r.pr_id), set()).add(r.sha)
    return idx


def _blame_targets(pairs) -> list[tuple]:
    """Distinct ``(owner, repo, q_merge_base_oid, filename)`` to blame: every shared
    CODE file from which Q removes >=1 line (blame only speaks where Q deletes).
    Uses the already-cached diffs + PR-meta/merge-base, so it makes no new calls."""
    targets: list[tuple] = []
    for row in pairs.itertuples():
        qmb = ba.q_merge_base(row.q_url)
        if not qmb:
            continue
        p_files, q_files = fetch_files(row.p_url), fetch_files(row.q_url)
        if not p_files or not q_files:
            continue
        p_names = {f["filename"] for f in p_files}
        for f in q_files:
            n = f["filename"]
            if n in p_names and is_code(n) and ba.removed_oldlines(f.get("patch")):
                targets.append((qmb[0], qmb[1], qmb[2], n))
    return targets


def _prefetch_blame(con, pairs, concurrency: int) -> dict[int, set[str]]:
    """Prefetch all Method-B inputs (PR metadata, Q merge-bases, the blame of every
    shared CODE file Q deletes from) into the caches, and return P's sha index.
    Every step is cached + resumable, so a re-run only fetches what is missing."""
    ba.prefetch_meta(list(pairs["p_url"]) + list(pairs["q_url"]), concurrency, console)
    ba.prefetch_merge_bases(list(pairs["q_url"]), concurrency, console)
    ba.prefetch_blame(_blame_targets(pairs), concurrency, console)
    ba.save_all()
    return _p_sha_index(con, sorted({int(r) for r in pairs["p_id"] if pd.notna(r)}))


def _blame_for_file(qmb, pset, name: str, q_patch: str) -> dict:
    """Per-file blame attribution fields for one shared file (Method B).
    ``qmb`` = (owner, repo, merge_base_oid) for Q, ``pset`` = P's sha set."""
    rem = ba.removed_oldlines(q_patch)
    ranges = ba.blame(qmb[0], qmb[1], qmb[2], name) if (qmb and rem) else None
    res = ba.attribute_removed(rem, ranges, pset)
    return {"blame_status": res["status"], "blame_n_removed": res["n_removed"],
            "blame_n_resolved": res["n_resolved"], "blame_to_p": res["n_blame_to_p"]}


def build_population(patch_cap: int = POP_PATCH_CAP, judgeable_only: bool = True,
                     limit: int | None = None, concurrency: int = 8,
                     fetch_only: bool = False, blame: bool = True,
                     exclude_agreed: bool = True,
                     out_path: str | Path | None = None,
                     p_side: str = "agent") -> Path | None:
    """Build the net-diff population cases, sized by ``patch_cap``.

    Enumerates (P, Q) pairs from DuckDB, fetches each diff (snapshot-cached), and
    assembles net-diff cases (shared = P's ∩ Q's files). With ``blame`` (default)
    each shared CODE file carries the Method-B attribution (``blame_to_p``: lines Q
    removes that ``git blame`` at Q's merge-base credits to P). With ``exclude_agreed``
    (default) the annotator-agreed pairs are dropped so the judged set stays disjoint
    from the prompt's training pairs. Returns the output path (None for ``fetch_only``).
    """
    ensure_dirs()
    con = duckdb.connect(str(DUCKDB_PATH), read_only=True)
    console.log(f"[dim]enumerating population pairs (p_side={p_side})...[/dim]")
    pairs = _load_pop_pairs(con, p_side)
    # q_is_human sorts first (False<True) so agent-Q pairs keep their original
    # (p_id, q_id) order + pair_ids; human-Q pairs append after as new ids.
    pairs = pairs.sort_values(["q_is_human", "p_id", "q_id"]).reset_index(drop=True)
    if limit:
        pairs = pairs.head(limit)
    console.print(f"  population: {len(pairs):,} pairs  (patch_cap={patch_cap:,})")

    distinct_urls = sorted({u for u in set(pairs["p_url"]) | set(pairs["q_url"]) if u})
    _fetch_all_diffs(distinct_urls, concurrency)
    if fetch_only:
        console.print("[green]fetch-only: snapshot populated, exiting.[/green]")
        return None

    repo_ids = sorted({int(r) for r in pairs["repo_id"] if pd.notna(r)})
    inter_idx = _intervening_index(con, repo_ids)
    console.print(f"  intervening-PR index: {len(inter_idx):,} (repo, file) keys "
                  f"over {len(repo_ids):,} repos")

    p_sha_idx = _prefetch_blame(con, pairs, concurrency) if blame else {}

    console.log("[dim]assembling cases from snapshot...[/dim]")
    cases = []
    for i, row in enumerate(pairs.itertuples(), start=1):
        p_files = fetch_files(row.p_url)
        q_files = fetch_files(row.q_url)
        p_patch = {f["filename"]: f.get("patch") or "" for f in p_files} if p_files else {}
        q_patch = {f["filename"]: f.get("patch") or "" for f in q_files} if q_files else {}
        shared_names = set(p_patch) & set(q_patch)
        repo_id = int(row.repo_id) if pd.notna(row.repo_id) else None
        exclude = {int(row.p_id), int(row.q_id)}
        inter_union: set[int] = set()
        qmb = ba.q_merge_base(row.q_url) if blame else None
        pset = (ba.p_sha_set(p_sha_idx.get(int(row.p_id), set()), row.p_url)
                if blame else set())
        shared = []
        for n in sorted(shared_names):
            hits = (_count_intervening(inter_idx, repo_id, n, row.p_merged, row.q_merged,
                                       exclude) if repo_id is not None else set())
            inter_union |= hits
            entry = {"name": n, "boilerplate": is_boilerplate(n),
                     "p_patch": truncate(p_patch[n], patch_cap),
                     "q_patch": truncate(q_patch[n], patch_cap),
                     "intervening": len(hits)}
            if blame and is_code(n):
                entry.update(_blame_for_file(qmb, pset, n, q_patch[n]))
            shared.append(entry)
        cases.append(_make_case(
            pair_id=i, agent=row.agent,
            p_id=row.p_id, q_id=row.q_id,
            p_url=row.p_url, q_url=row.q_url,
            p_title=row.p_title, q_title=row.q_title,
            p_body=row.p_body, q_body=row.q_body,
            p_commits=_commit_msgs(con, row.p_id),
            q_commits=_commit_msgs(con, row.q_id),
            day_gap=(int(row.day_gap) if row.day_gap == row.day_gap else None),
            shared=shared, q_only=[], p_only=[],
            full_diff=False, has_p_patch=bool(p_patch),
            n_intervening=len(inter_union),
            q_is_human=bool(getattr(row, "q_is_human", False)),
            p_is_human=(p_side == "human"),
        ))
        if i % 500 == 0:
            console.print(f"    assembled {i:,}/{len(pairs):,} cases")

    # pair_id assigned before filtering, so ids are stable across cap settings and
    # refer to the same (P, Q) pair everywhere.
    n_all = len(cases)
    n_complete = sum(1 for c in cases if c["context_complete"])
    n_shared = sum(1 for c in cases if c["n_shared"] > 0)
    if judgeable_only:
        cases = [c for c in cases if c["n_shared"] > 0]

    # Drop the agreed pairs AFTER pair_id assignment (like the judgeable filter) so
    # kept ids stay stable; excluded ids just leave gaps. Matched by URL (they train
    # the prompt -- see _agreed_url_pairs).
    n_excluded = 0
    if exclude_agreed:
        agreed = _agreed_url_pairs()
        if agreed:
            kept = [c for c in cases
                    if (_norm(c["p_url"]), _norm(c["q_url"])) not in agreed]
            n_excluded = len(cases) - len(kept)
            cases = kept

    out = Path(out_path) if out_path else population_cases_path(patch_cap, judgeable_only, p_side)
    out.write_text(json.dumps(cases))
    console.log(f"[green]wrote[/green] {out}  ({len(cases):,} cases"
                f"{' [judgeable-only]' if judgeable_only else ''}"
                f"{f' [-{n_excluded} agreed]' if n_excluded else ''})")
    console.print(f"  population {n_all:,} | with >=1 shared patch: {n_shared:,} | "
                  f"context_complete: {n_complete:,} | "
                  f"no net-diff overlap / dead PR: {n_all - n_shared:,}"
                  f"{f' | agreed-pairs excluded: {n_excluded}' if n_excluded else ''}")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Build RQ1 LLM-judge case files.")
    ap.add_argument("--mode", choices=["gold", "population"], default="gold",
                    help="gold = fixed 50-pair validation set; population = "
                         "config-sized agent-Q net-diff pool.")
    ap.add_argument("--out", default=None,
                    help="Output JSON path (default: derived from mode/patch-cap).")
    # gold knobs
    ap.add_argument("--full-diff", action="store_true",
                    help="[gold] Also include each PR's NON-shared changed files, "
                         "so the judge can see a fix outside the file overlap.")
    # population knobs
    ap.add_argument("--patch-cap", type=int, default=None,
                    help=f"Chars per file patch (gold default {PATCH_CAP}, "
                         f"population default {POP_PATCH_CAP}).")
    ap.add_argument("--judgeable-only", action="store_true",
                    help="[population] Emit only pairs with a real net-diff overlap "
                         "(n_shared>0).")
    ap.add_argument("--limit", type=int, default=None,
                    help="[population] Only the first N pairs (smoke test).")
    ap.add_argument("--concurrency", type=int, default=8,
                    help="[population] Parallel gh fetches.")
    ap.add_argument("--fetch-only", action="store_true",
                    help="[population] Only populate the diff snapshot; don't assemble.")
    ap.add_argument("--no-blame", action="store_true",
                    help="[population] Skip Method-B blame attribution (no extra gh "
                         "calls); shared files carry only the content-overlap signal.")
    ap.add_argument("--p-side", choices=["agent", "human"], default="agent",
                    help="[population] Whose merges form the P side: agent (default) "
                         "or human (the contemporaneous human cohort, for the "
                         "incidence baseline; writes ..._humanP.json).")
    ap.add_argument("--keep-agreed", action="store_true",
                    help="[population] Do NOT drop the annotator-agreed gold pairs "
                         "(by default they are excluded so the judged population is "
                         "disjoint from the pairs that trained the prompt).")
    args = ap.parse_args()

    if args.mode == "gold":
        build_gold(full_diff=args.full_diff, out_path=args.out,
                   patch_cap=args.patch_cap or PATCH_CAP)
    else:
        build_population(patch_cap=args.patch_cap or POP_PATCH_CAP,
                         judgeable_only=args.judgeable_only, limit=args.limit,
                         concurrency=args.concurrency, fetch_only=args.fetch_only,
                         blame=not args.no_blame, exclude_agreed=not args.keep_agreed,
                         out_path=args.out, p_side=args.p_side)
