"""Attribution for the ``[P-overlap]`` note: does Q's change land on a line P
introduced? Two methods, cheapest to ground-truth.

**Method A -- content overlap (no network).** From the two case patches: does a
line Q removes / a context line in Q's hunk match a line P added, with Q's net
change more than cosmetic? The fallback: under-counts when P's line was reformatted
between P and Q, and can't tell P's line from an identical one authored in between.
Also drives the cosmetic-only provenance banner (the pair-#69 guard). Public:
``is_code``, ``file_provenance``, ``pair_provenance``.

**Method B -- git blame (ground truth).** Blames the shared file at Q's merge-base
on the lines Q removes and checks whether the authoring commit is P's (P's shas +
squash ``merge_commit_sha``). History-aware: sees through reformatting and
intermediate rewrites. A rebase-merge that rewrote P's commits to new shas can't be
mapped, so Method B reports 0 and the caller falls back to A. All fetches (PR meta,
merge-base, blame) are cached to JSON snapshots in ``INTERIM_DIR`` for a
reproducible, free rebuild. The pure functions (``removed_oldlines``,
``attribute_removed``) take no network and are unit-tested.
"""
from __future__ import annotations

import json
import re
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from prfix.config import INTERIM_DIR

# ============================================================================
# Method A -- content overlap (no network); feeds the note + provenance banner
# ============================================================================
# no-logic lines; a change confined to these is "cosmetic"
_COMMENT_PREFIXES = ("//", "#", "*", "/*", "*/", "<!--", "-->", '"""', "'''",
                     ";;", "--")
_PUNCT = set("{}()[];,:")
# doc/data files: an overlap here is cleanup, not a code-defect repair
_NONCODE_EXT = {"md", "mdx", "markdown", "rst", "txt", "json", "yaml", "yml",
                "toml", "lock", "csv", "tsv", "xml", "svg", "html", "htm",
                "ini", "cfg", "env", "properties", "gitignore", "lockb"}
_NONCODE_NAMES = {"changelog", "license", "notice", "authors", "codeowners"}


def is_code(name: str) -> bool:
    """A code file (vs docs/data/config) — only a code overlap counts as a fix."""
    base = name.rsplit("/", 1)[-1].lower()
    if base in _NONCODE_NAMES or base.split(".")[0] in _NONCODE_NAMES:
        return False
    ext = base.rsplit(".", 1)[-1] if "." in base else ""
    return ext not in _NONCODE_EXT


def _strip(line: str) -> str:
    return line.strip()


def _key(line: str) -> str:
    """Whitespace-insensitive identity of a code line, so brace/indent
    reformatting (``{ X }`` -> ``{X}``) doesn't defeat line matching."""
    return re.sub(r"\s+", "", line)


def _is_trivial(stripped: str) -> bool:
    """Whitespace-only, pure-bracket/punctuation, or a comment line."""
    if not stripped:
        return True
    if all(ch in _PUNCT for ch in stripped):
        return True
    return any(stripped.startswith(p) for p in _COMMENT_PREFIXES)


def _classify(patch: str | None) -> tuple[list[str], list[str], list[str]]:
    """Split a unified-diff patch into (added, removed, context) line bodies."""
    added: list[str] = []
    removed: list[str] = []
    context: list[str] = []
    for raw in (patch or "").splitlines():
        if not raw:
            continue
        if raw[:2] in ("@@", "++", "--") or raw.startswith(("diff ", "index ", "\\")):
            # hunk header / file header / "\ No newline at end of file"
            continue
        tag, body = raw[0], raw[1:]
        if tag == "+":
            added.append(body)
        elif tag == "-":
            removed.append(body)
        elif tag == " ":
            context.append(body)
    return added, removed, context


def _nontrivial(lines: list[str]) -> set[str]:
    """Whitespace-insensitive keys of the non-trivial lines."""
    return {_key(ln) for ln in lines if not _is_trivial(_strip(ln))}


def file_provenance(p_patch: str | None, q_patch: str | None) -> dict:
    """Per-shared-file provenance signals from the two one-file patches."""
    p_added_all, _, _ = _classify(p_patch)
    q_added, q_removed, q_context = _classify(q_patch)
    p_added = _nontrivial(p_added_all)
    q_add, q_rem, q_ctx = (_nontrivial(q_added), _nontrivial(q_removed),
                           _nontrivial(q_context))
    removed_hits = q_rem & p_added           # Q reverts/edits a line P added
    context_hits = q_ctx & p_added           # Q edits inside a region P authored
    return {
        "q_eff": len(q_add) + len(q_rem),    # Q's non-trivial changed lines here
        "p_added": len(p_added),
        "q_overlaps_p": bool(removed_hits or context_hits),
        "n_removed_hits": len(removed_hits),
        "n_context_hits": len(context_hits),
    }


def pair_provenance(shared: list[dict]) -> dict:
    """Roll per-file signals over the substantive (non-boilerplate) shared files.

    Returns ``suspect`` (the Direct-fix-over-credit guard) plus a ``warning``
    string to surface, and the per-file detail keyed by filename.
    """
    per = {}
    cosmetic, no_overlap, code_overlap = [], [], []
    for s in (shared or []):
        if s.get("boilerplate"):
            continue
        pr = file_provenance(s.get("p_patch"), s.get("q_patch"))
        per[s["name"]] = pr
        if not (s.get("q_patch") or "").strip() or not is_code(s["name"]):
            continue                                       # only CODE files gate a fix
        if pr["q_eff"] == 0:                               # comment/whitespace-only
            cosmetic.append(s["name"])
        elif pr["q_overlaps_p"]:                           # Q edits a code line P wrote
            code_overlap.append(s["name"])
        else:                                              # real code change, but not on P's lines
            no_overlap.append(s["name"])
    changed_code = cosmetic or no_overlap or code_overlap

    # SUSPECT (high precision): every code change is a comment/whitespace tweak —
    # the #69 disease (commit messages claim a fix the net diff lacks). 0/14 false
    # positives on gold Direct-fixes.
    suspect = bool(cosmetic) and not code_overlap and not no_overlap
    # WEAK (advisory only): real code changes but none touch a line P introduced.
    # Noisier (~3/14 gold Direct-fixes), so it never gates the judge — surfaced to
    # the human auditor, not the model.
    weak = bool(no_overlap) and not code_overlap

    warning = ""
    if suspect:
        warning = ("AUTOMATED PROVENANCE CHECK — a 'Direct fix' is not supported "
                   "by the shared code diffs: Q's net change is comment/whitespace-"
                   "only on " + ", ".join(cosmetic) + ". The substantive code here "
                   "appears only as unchanged context (likely authored by a PR "
                   "merged between P and Q). Credit Q only for code it actually "
                   "adds/removes, and weight Q's diff over its commit-message claims.")
    return {"suspect": suspect, "weak": weak, "warning": warning,
            "cosmetic_files": cosmetic, "no_overlap_files": no_overlap,
            "overlap_files": code_overlap, "changed_code": bool(changed_code),
            "per_file": per}


# ============================================================================
# Method B -- git blame at Q's merge-base (ground truth); cached to snapshots
# ============================================================================
_URL = re.compile(r"github\.com/([^/]+)/([^/]+)/pull/(\d+)")
_HUNK = re.compile(r"@@ -(\d+)(?:,\d+)? \+\d+")

# ---- snapshots (one JSON cache per call type) ------------------------------
PR_META_SNAP = INTERIM_DIR / "rq1_pr_meta_snapshot.json"
MERGE_BASE_SNAP = INTERIM_DIR / "rq1_merge_base_snapshot.json"
BLAME_SNAP = INTERIM_DIR / "rq1_blame_snapshot.json"

_BLAME_QL = (
    "query($o:String!,$r:String!,$oid:GitObjectID!,$p:String!){"
    "repository(owner:$o,name:$r){object(oid:$oid){... on Commit{"
    "blame(path:$p){ranges{startingLine endingLine commit{oid}}}}}}}"
)


class _Snap:
    """A lazily-loaded JSON cache that is only rewritten when it gains entries."""

    def __init__(self, path: Path):
        self.path = path
        self.data: dict | None = None
        self.dirty = False
        self.lock = threading.RLock()  # guards lazy load + save across prefetch threads

    def load(self) -> dict:
        # Double-checked lock: dict item-set (snap[key]=v) is atomic under the GIL so
        # writers need no lock; only lazy init must run once, else a second parse
        # replaces self.data and drops entries added meanwhile.
        if self.data is None:
            with self.lock:
                if self.data is None:
                    try:
                        self.data = json.loads(self.path.read_text())
                    except (FileNotFoundError, json.JSONDecodeError):
                        self.data = {}
        return self.data

    def save(self) -> None:
        with self.lock:
            if self.dirty and self.data is not None:
                self.path.write_text(json.dumps(self.data, sort_keys=True), encoding="utf-8")
                self.dirty = False


_meta = _Snap(PR_META_SNAP)
_mbase = _Snap(MERGE_BASE_SNAP)
_blame = _Snap(BLAME_SNAP)
# in-memory negatives (unparseable url / permanent miss) so a run doesn't re-call;
# NOT persisted, so a later rebuild retries transient failures.
_meta_miss: set[str] = set()


def save_all() -> None:
    """Persist any newly fetched meta/merge-base/blame entries."""
    _meta.save()
    _mbase.save()
    _blame.save()


def _norm(url: str) -> str:
    return (url or "").rstrip("/")


# ---- rate-limit handling (coordinated across the prefetch threads) ---------
# A rate limit must not crash the build nor silently drop calls (incomplete
# attribution). Instead a thread that hits one announces it once, sets a SHARED
# pause until the limit resets, and every gh call waits it out before retrying --
# so the threads back off together.
_RL_RETRIES = 6          # per-call retries before giving up (then caller falls back)
_RL_MAX_WAIT = 3600      # cap a single wait at 1h (the primary limit resets hourly)
_RL_SECONDARY_WAIT = 90  # backoff when GitHub reports no concrete reset (secondary limit)
_rl_lock = threading.Lock()
_rl_state: dict = {"resume_at": 0.0}      # epoch until which all gh calls pause
_console_state: dict = {"console": None}  # where to surface notices (set by callers)


def set_console(console) -> None:
    """Route rate-limit notices to the user's terminal (a rich Console or None)."""
    _console_state["console"] = console


def _notify(msg: str) -> None:
    c = _console_state["console"]
    (c.print if c is not None else print)(msg)


def _is_rate_limited(stderr: str) -> bool:
    s = (stderr or "").lower()
    return "rate limit" in s or "secondary rate" in s or "429" in s


def _reset_wait(kind: str) -> float:
    """Seconds until the ``kind`` (core/graphql) limit resets; 0 if not exhausted.
    The ``rate_limit`` endpoint itself does not count against the limit."""
    try:
        out = subprocess.run(["gh", "api", "rate_limit"], capture_output=True,
                             text=True, timeout=30, check=False)
        if out.returncode != 0:
            return 0.0
        res = (json.loads(out.stdout) or {}).get("resources", {})
        bucket = res.get(kind) or res.get("core") or {}
        if bucket.get("remaining", 1) == 0 and bucket.get("reset"):
            return max(0.0, float(bucket["reset"]) - time.time())
    except (subprocess.SubprocessError, json.JSONDecodeError, TypeError, ValueError):
        pass
    return 0.0


def _wait_if_paused() -> None:
    """Block while a rate-limit pause set by any thread is still in effect."""
    with _rl_lock:
        wait = _rl_state["resume_at"] - time.time()
    if wait > 0:
        time.sleep(min(wait, _RL_MAX_WAIT))


def _begin_pause(kind: str) -> None:
    """Record a shared pause until the limit resets, announcing it exactly once."""
    wait = min((_reset_wait(kind) or _RL_SECONDARY_WAIT) + 2.0, _RL_MAX_WAIT)
    with _rl_lock:
        now = time.time()
        fresh = _rl_state["resume_at"] <= now          # not already paused
        _rl_state["resume_at"] = max(_rl_state["resume_at"], now + wait)
        resume_at = _rl_state["resume_at"]
    if fresh:  # only the thread that opened the pause announces it
        mins = max(0.0, resume_at - time.time()) / 60.0
        when = time.strftime("%H:%M:%S", time.localtime(resume_at))
        _notify(f"[yellow]GitHub API rate limit reached -- pausing ~{mins:.1f} min "
                f"(auto-resume ~{when}); the run will wait and continue, not crash.[/yellow]")


def gh_json(args: list[str], timeout: int = 60, kind: str = "core"):
    """Run ``gh`` and parse stdout as JSON. On a rate-limit error, announce + wait
    for the reset (shared across threads) and retry, rather than failing. Returns
    None only on a genuine error or after exhausting retries (caller falls back)."""
    for _ in range(_RL_RETRIES + 1):
        _wait_if_paused()
        try:
            r = subprocess.run(["gh", *args], capture_output=True, text=True,
                               timeout=timeout, check=False)
        except subprocess.TimeoutExpired:
            return None
        if r.returncode == 0:
            try:
                return json.loads(r.stdout)
            except json.JSONDecodeError:
                return None
        if _is_rate_limited(r.stderr):
            _begin_pause(kind)
            continue
        return None
    return None


# ---- pure diff/blame logic (no network) ------------------------------------
def removed_oldlines(patch: str | None) -> list[int]:
    """Old-side (pre-image) line numbers of every '-' line in a unified diff.

    These index lines *at Q's merge-base*, which is exactly where we blame."""
    nums: list[int] = []
    old: int | None = None
    for ln in (patch or "").splitlines():
        m = _HUNK.match(ln)
        if m:
            old = int(m.group(1))
            continue
        if old is None:
            continue
        if ln.startswith("-") and not ln.startswith("---"):
            nums.append(old)
            old += 1
        elif ln.startswith("+"):
            continue            # added line: no old-side number
        else:
            old += 1            # context line advances the old-side cursor
    return nums


def _owner_of(line: int, ranges: list) -> str | None:
    """The commit oid blame attributes ``line`` to (ranges are [start, end, oid])."""
    for start, end, oid in ranges:
        if start <= line <= end:
            return oid
    return None


def attribute_removed(removed: list[int], ranges: list | None,
                      p_shas: set[str]) -> dict:
    """Map Q's removed line numbers to blame commits and count how many are P's.

    Returns ``status`` plus counts:
      * ``no_removed``  -- Q removes nothing here; blame has no target.
      * ``unavailable`` -- blame ranges missing (fetch failed / path renamed).
      * ``ok``          -- ``n_blame_to_p`` of ``n_resolved`` resolved removed lines
                           were authored by P (``n_removed`` total, the rest fell in
                           no blame range -- e.g. file shorter at merge-base)."""
    if not removed:
        return {"status": "no_removed", "n_removed": 0, "n_resolved": 0,
                "n_blame_to_p": 0}
    if not ranges:
        return {"status": "unavailable", "n_removed": len(removed), "n_resolved": 0,
                "n_blame_to_p": 0}
    oids = [_owner_of(ln, ranges) for ln in removed]
    resolved = [o for o in oids if o is not None]
    hits = sum(1 for o in resolved if o in p_shas)
    return {"status": "ok", "n_removed": len(removed), "n_resolved": len(resolved),
            "n_blame_to_p": hits}


def parse_pr_url(url: str) -> tuple[str, str, str] | None:
    m = _URL.search(url or "")
    return (m.group(1), m.group(2), m.group(3)) if m else None


# ---- cached fetchers -------------------------------------------------------
def pr_meta(url: str) -> dict | None:
    """``{owner, repo, num, base, head, merge_commit}`` for a PR url, cached.

    Returns None (and remembers the miss for this run) on an unparseable url or a
    fetch failure; transient failures are not persisted, so a rebuild retries."""
    key = _norm(url)
    if key in _meta_miss:
        return None
    snap = _meta.load()
    if key in snap:
        return snap[key]
    parsed = parse_pr_url(url)
    if not parsed:
        _meta_miss.add(key)
        return None
    owner, repo, num = parsed
    j = gh_json(["api", f"repos/{owner}/{repo}/pulls/{num}"])
    if not j:
        _meta_miss.add(key)
        return None
    meta = {"owner": owner, "repo": repo, "num": num,
            "base": (j.get("base") or {}).get("sha"),
            "head": (j.get("head") or {}).get("sha"),
            "merge_commit": j.get("merge_commit_sha")}
    snap[key] = meta
    _meta.dirty = True
    return meta


def merge_base(owner: str, repo: str, base: str, head: str) -> str | None:
    """The merge-base commit oid of ``base...head`` (Q's diff base), cached."""
    if not (base and head):
        return None
    key = f"{owner}/{repo}:{base}...{head}"
    snap = _mbase.load()
    if key in snap:
        return snap[key]
    j = gh_json(["api", f"repos/{owner}/{repo}/compare/{base}...{head}"])
    if not j:
        return None
    oid = (j.get("merge_base_commit") or {}).get("sha")
    if oid:
        snap[key] = oid
        _mbase.dirty = True
    return oid


def blame(owner: str, repo: str, oid: str, path: str) -> list | None:
    """Blame ``path`` at commit ``oid`` -> ``[[start, end, commit_oid], ...]``, cached.

    None when the path/oid is absent at that commit (rename, deleted) or the call
    fails -- the caller then falls back to Method A."""
    if not oid:
        return None
    key = f"{owner}/{repo}@{oid}:{path}"
    snap = _blame.load()
    if key in snap:
        return snap[key]
    j = gh_json(["api", "graphql", "-f", "query=" + _BLAME_QL,
                 "-f", "o=" + owner, "-f", "r=" + repo, "-f", "oid=" + oid,
                 "-f", "p=" + path], kind="graphql")
    if not j:
        return None
    try:
        raw = j["data"]["repository"]["object"]["blame"]["ranges"]
    except (KeyError, TypeError):
        return None
    ranges = [[r["startingLine"], r["endingLine"], r["commit"]["oid"]] for r in raw]
    snap[key] = ranges
    _blame.dirty = True
    return ranges


def q_merge_base(q_url: str) -> tuple[str, str, str] | None:
    """``(owner, repo, merge_base_oid)`` for Q, or None if meta/merge-base missing."""
    m = pr_meta(q_url)
    if not m:
        return None
    mb = merge_base(m["owner"], m["repo"], m["base"], m["head"])
    if not mb:
        return None
    return m["owner"], m["repo"], mb


def p_sha_set(p_commit_shas: set[str], p_url: str) -> set[str]:
    """P's commit shas (from ``pr_commits``) plus P's merge_commit_sha (squash)."""
    shas = set(p_commit_shas)
    m = pr_meta(p_url)
    if m and m.get("merge_commit"):
        shas.add(m["merge_commit"])
    return shas


# ---- concurrent prefetch (resumable, chunk-checkpointed) -------------------
def _prefetch(label: str, items: list, fn, concurrency: int, console) -> None:
    """Run ``fn`` over ``items`` concurrently, checkpointing snapshots per chunk."""
    if not items:
        return
    if console:
        set_console(console)  # so a rate-limit pause is announced to this terminal
        console.print(f"  blame: prefetching {len(items):,} {label} "
                      f"(concurrency={concurrency})")
    CHUNK = 200
    for start in range(0, len(items), CHUNK):
        chunk = items[start:start + CHUNK]
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            futs = [pool.submit(fn, it) for it in chunk]
            for fut in as_completed(futs):
                try:
                    fut.result()
                except Exception:  # noqa: BLE001 -- transient; not snapshotted, retried
                    pass
        save_all()
        if console:
            console.print(f"    {label}: {min(start + CHUNK, len(items)):,}/{len(items):,}")


def prefetch_meta(urls, concurrency: int, console=None) -> None:
    todo = [u for u in {_norm(u) for u in urls if u} if u not in _meta.load()]
    _prefetch("PR-meta", todo, pr_meta, concurrency, console)


def prefetch_merge_bases(q_urls, concurrency: int, console=None) -> None:
    """Merge-base for each distinct Q (needs meta already cached)."""
    tasks = []
    seen = set()
    for u in {_norm(u) for u in q_urls if u}:
        m = pr_meta(u)
        if not m or not (m["base"] and m["head"]):
            continue
        key = f"{m['owner']}/{m['repo']}:{m['base']}...{m['head']}"
        if key in _mbase.load() or key in seen:
            continue
        seen.add(key)
        tasks.append((m["owner"], m["repo"], m["base"], m["head"]))
    _prefetch("merge-base", tasks, lambda t: merge_base(*t), concurrency, console)


def prefetch_blame(targets, concurrency: int, console=None) -> None:
    """``targets``: iterable of ``(owner, repo, oid, path)``; deduped before fetch."""
    seen = set()
    todo = []
    snap = _blame.load()
    for owner, repo, oid, path in targets:
        key = f"{owner}/{repo}@{oid}:{path}"
        if key in snap or key in seen:
            continue
        seen.add(key)
        todo.append((owner, repo, oid, path))
    _prefetch("blame", todo, lambda t: blame(*t), concurrency, console)
