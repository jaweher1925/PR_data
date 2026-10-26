"""Bulk-fetch GitHub net-diff filenames for the full RQ1 candidate-pair PR set.

Fetches the net (base->head) changed-file list for every PR that appears as a P
or candidate-Q in the linker's 30-day pairs (~5.8k PRs), so the linker overlap
can be recomputed on net-diff instead of the per-commit union. Filenames only
(no patches) -- the overlap test needs only names.

Resumable: each PR's result is appended to ``netdiff_cache.jsonl`` immediately,
and a re-run skips PRs already cached. Safe to kill/restart. Retries on rate
limits. Merged-PR net diffs are immutable, so results are stable.

Stage 1 of the net-diff repro chain: this fetch -> data/interim/netdiff_cache.jsonl
-> prfix.common.loaders.build_net_files -> pr_net_files table -> the RQ1 linker.

Run (background): uv run python -m prfix.common.loaders.fetch_netdiff
``--p-side {agent,human,both}`` picks whose merges form the P side of the pairs
(default ``agent``, the paper's cohort); the Q side always spans both cohorts.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time

from prfix.common.loaders.base import connect
from prfix.config import INTERIM_DIR, ensure_dirs
from prfix.rq1_fix_rates import linker as rb

CACHE = INTERIM_DIR / "netdiff_cache.jsonl"
_URL = re.compile(r"github\.com/([^/]+)/([^/]+)/pull/(\d+)")


def pr_list(p_side: str = "agent") -> list[tuple[int, str]]:
    """Distinct (pr_id, url) for every PR in the candidate pair set.

    ``p_side`` selects whose merges form the P side ("agent" = the paper's
    cohort, "human" = the contemporaneous human cohort for the incidence
    baseline); the Q side always spans both cohorts."""
    sql = rb.pairs_prelude_sql(p_side) + """
    , ids AS (SELECT DISTINCT p_id AS id FROM pair UNION SELECT q_id FROM pair)
    SELECT i.id AS id, COALESCE(a.html_url, h.html_url) AS url
    FROM ids i
    LEFT JOIN pull_request a       ON a.id = i.id
    LEFT JOIN human_pull_request h ON h.id = i.id
    """
    con = connect(read_only=True)
    df = con.execute(sql).df()
    con.close()
    return [(int(r.id), r.url) for r in df.itertuples() if isinstance(r.url, str)]


def fetch_filenames(url: str, tries: int = 5):
    """Net-diff filenames for a PR; (list, 'ok') or (None, reason)."""
    m = _URL.search(url)
    if not m:
        return None, "bad_url"
    owner, repo, num = m.groups()
    for attempt in range(tries):
        r = subprocess.run(
            ["gh", "api", "--paginate", f"repos/{owner}/{repo}/pulls/{num}/files",
             "-q", ".[].filename"],
            capture_output=True, text=True, timeout=120, check=False)
        if r.returncode == 0:
            return [ln for ln in r.stdout.splitlines() if ln], "ok"
        err = (r.stderr or "").lower()
        if "rate limit" in err or "was submitted too quickly" in err or "403" in err:
            time.sleep(60 * (attempt + 1))          # back off and retry
            continue
        if "404" in err or "not found" in err:
            return None, "not_found"                # deleted/private
        return None, "error"
    return None, "rate_limited"


def main() -> None:
    ap = argparse.ArgumentParser(description="Bulk-fetch net-diff filenames for linker pair PRs")
    ap.add_argument("--p-side", choices=["agent", "human", "both"], default="agent",
                    help="whose merges form the P side of the pairs (default: agent; "
                         "'human' adds the ~4.6k human-cohort merges for the incidence baseline)")
    args = ap.parse_args()

    ensure_dirs()
    done = set()
    if CACHE.exists():
        for ln in CACHE.read_text().splitlines():
            try:
                done.add(json.loads(ln)["pr_id"])
            except Exception:
                pass

    sides = ["agent", "human"] if args.p_side == "both" else [args.p_side]
    seen: set[int] = set()
    prs: list[tuple[int, str]] = []
    for s in sides:
        for i, u in pr_list(s):
            if i not in seen:
                seen.add(i)
                prs.append((i, u))
    todo = [(i, u) for i, u in prs if i not in done]
    print(f"total PRs: {len(prs)} | cached: {len(done)} | to fetch: {len(todo)}", flush=True)

    t0 = time.time()
    with CACHE.open("a") as fh:
        for k, (pid, url) in enumerate(todo, 1):
            files, status = fetch_filenames(url)
            fh.write(json.dumps({"pr_id": pid, "url": url, "status": status,
                                 "files": files or []}) + "\n")
            fh.flush()
            if k % 100 == 0 or k == len(todo):
                rate = k / max(time.time() - t0, 1e-9)
                eta = (len(todo) - k) / max(rate, 1e-9) / 60
                print(f"  {k}/{len(todo)}  ({rate*60:.0f}/min, ETA {eta:.0f} min)", flush=True)
    print(f"done: {len(todo)} fetched in {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    sys.exit(main())
