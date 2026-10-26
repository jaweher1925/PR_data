"""GitHub REST API helpers for human-cohort enrichment and the net-diff fetch.

* Token is loaded from ``.env`` (key ``GITHUB_TOKEN``) or the env var
  of the same name. Loaded once, cached in module state, never logged.
* All file fetches are cached on disk under ``data/github_cache/`` keyed
  by a blake2b hash of (owner, repo, ref, path). Reruns are free.
* Public API: ``get_token``, ``get_pr_base_sha``, ``get_file_at_ref``,
  ``parse_repo_url``, ``rate_limit_status``.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from functools import lru_cache
from pathlib import Path

from prfix.config import DATA_DIR, PROJECT_ROOT

CACHE_DIR = DATA_DIR / "github_cache"
USER_AGENT = "pr-fix-authorship/1.0 (research; +https://github.com/)"
API_BASE = "https://api.github.com"
RAW_BASE = "https://raw.githubusercontent.com"
HTTP_NOT_FOUND = 404
_MAX_ATTEMPTS = 3
_MAX_BACKOFF_S = 120  # don't block a run for more than 2 min on a rate-limit reset


# ---------------------------------------------------------------------------
# Token loading (.env-aware, never logged)
# ---------------------------------------------------------------------------

def _load_dotenv(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    if not path.exists():
        return out
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):]
        key, _, val = line.partition("=")
        out[key.strip()] = val.strip().strip('"').strip("'")
    return out


@lru_cache(maxsize=1)
def get_token() -> str:
    """Load ``GITHUB_TOKEN`` from env var or ``.env``; cached after first read."""
    tok = os.environ.get("GITHUB_TOKEN", "")
    if not tok:
        env = _load_dotenv(PROJECT_ROOT / ".env")
        tok = env.get("GITHUB_TOKEN", "")
    if not tok:
        raise RuntimeError(
            "GITHUB_TOKEN not found. Create a fine-grained PAT with read-only "
            "public-repository contents permission and add it to .env as "
            "`GITHUB_TOKEN=ghp_...` (or export it in your shell)."
        )
    return tok


def _auth_headers(accept: str = "application/vnd.github+json") -> dict[str, str]:
    return {
        "Authorization": f"Bearer {get_token()}",
        "User-Agent":    USER_AGENT,
        "Accept":        accept,
    }


# ---------------------------------------------------------------------------
# HTTP helpers with retry + rate-limit backoff
# ---------------------------------------------------------------------------

def _request(url: str, *, accept: str, timeout: int = 30) -> tuple[int, bytes, dict]:
    """Single authenticated request. Returns (status, body, headers).
    Raises on non-2xx / non-404 after retries."""
    last_exc: Exception | None = None
    for attempt in range(_MAX_ATTEMPTS):
        try:
            req = urllib.request.Request(url, headers=_auth_headers(accept))
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.status, r.read(), dict(r.headers)
        except urllib.error.HTTPError as e:
            if e.code == HTTP_NOT_FOUND:
                return HTTP_NOT_FOUND, b"", dict(e.headers)
            if e.code in (403, 429):
                # Rate-limit or abuse-detection -- back off.
                reset = int(e.headers.get("X-RateLimit-Reset", "0"))
                wait = max(1, reset - int(time.time()))
                if wait > _MAX_BACKOFF_S:
                    raise
                time.sleep(min(wait, 60))
                last_exc = e
                continue
            if attempt == _MAX_ATTEMPTS - 1:
                raise
            time.sleep(2 ** attempt)
            last_exc = e
        except (urllib.error.URLError, TimeoutError) as e:
            if attempt == _MAX_ATTEMPTS - 1:
                raise
            time.sleep(2 ** attempt)
            last_exc = e
    if last_exc is not None:
        raise last_exc
    raise RuntimeError("unreachable")


# ---------------------------------------------------------------------------
# Disk cache helpers
# ---------------------------------------------------------------------------

def _cache_path(*parts: str) -> Path:
    h = hashlib.blake2b("|".join(parts).encode("utf-8"), digest_size=12).hexdigest()
    return CACHE_DIR / h[:2] / f"{h}.json"


def _cache_read(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def _cache_write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def parse_repo_url(repo_url: str) -> tuple[str, str]:
    """Parse ``https://api.github.com/repos/{owner}/{name}`` -> ``(owner, name)``."""
    parts = repo_url.rstrip("/").split("/")
    return parts[-2], parts[-1]


def get_pr_base_sha(owner: str, repo: str, pr_number: int) -> str | None:
    """Return the merge-base SHA of a PR. ``None`` if PR / repo is inaccessible."""
    cache = _cache_path("base_sha", owner, repo, str(pr_number))
    cached = _cache_read(cache)
    if cached is not None:
        return cached.get("sha")

    url = f"{API_BASE}/repos/{owner}/{repo}/pulls/{pr_number}"
    status, body, _ = _request(url, accept="application/vnd.github+json")
    if status == HTTP_NOT_FOUND:
        sha: str | None = None
    else:
        sha = json.loads(body).get("base", {}).get("sha")
    _cache_write(cache, {"sha": sha})
    return sha


def get_file_at_ref(owner: str, repo: str, path: str, ref: str) -> str | None:
    """Return UTF-8 file content at a git ref. ``None`` if file does not exist there.

    Uses ``raw.githubusercontent.com`` to handle large files and to skip
    the contents-API base64 round trip. Authenticated requests count
    against the same hourly rate-limit budget as the JSON API.
    """
    cache = _cache_path("file", owner, repo, ref, path)
    cached = _cache_read(cache)
    if cached is not None:
        return cached.get("content")

    url = f"{RAW_BASE}/{owner}/{repo}/{ref}/{urllib.parse.quote(path)}"
    status, body, _ = _request(url, accept="*/*")
    if status == HTTP_NOT_FOUND:
        content: str | None = None
    else:
        content = body.decode("utf-8", errors="replace")
    _cache_write(cache, {"content": content})
    return content


def get_pr_commits(owner: str, repo: str, pr_number: int) -> list[dict]:
    """Return the list of commits on a PR branch.

    Each entry has at least ``sha``, ``commit.author``, ``commit.committer``,
    ``commit.message``. ``None`` if PR/repo is inaccessible. Disk-cached.

    The GitHub API caps this endpoint at 250 commits per PR; for our use
    cases (October-2020 human PRs, almost all <30 commits) one page is
    almost always enough, so we read just the first page.
    """
    cache = _cache_path("pr_commits", owner, repo, str(pr_number))
    cached = _cache_read(cache)
    if cached is not None:
        return cached.get("commits") or []

    url = f"{API_BASE}/repos/{owner}/{repo}/pulls/{pr_number}/commits?per_page=100"
    status, body, _ = _request(url, accept="application/vnd.github+json")
    commits: list[dict] = [] if status == HTTP_NOT_FOUND else json.loads(body)
    _cache_write(cache, {"commits": commits})
    return commits


def get_commit_detail(owner: str, repo: str, sha: str) -> dict | None:
    """Return the full commit detail (incl. ``files`` with per-file diffs).

    Returns ``None`` if not found. Disk-cached. Each commit costs 1 API
    call; this dominates Stage 4 wall-clock.
    """
    cache = _cache_path("commit_detail", owner, repo, sha)
    cached = _cache_read(cache)
    if cached is not None:
        return cached.get("commit")

    url = f"{API_BASE}/repos/{owner}/{repo}/commits/{sha}"
    status, body, _ = _request(url, accept="application/vnd.github+json")
    commit: dict | None = None if status == HTTP_NOT_FOUND else json.loads(body)
    _cache_write(cache, {"commit": commit})
    return commit


def count_pr_reviews(owner: str, repo: str, pr_number: int) -> int:
    """Return the number of formal PR reviews (approve / request-changes / comment).

    One paginated call. AIDev's ``pr_reviews`` table is per-review. Disk-cached.
    """
    cache = _cache_path("reviews", owner, repo, str(pr_number))
    cached = _cache_read(cache)
    if cached is not None:
        return int(cached.get("n", 0))

    url = f"{API_BASE}/repos/{owner}/{repo}/pulls/{pr_number}/reviews?per_page=100"
    status, body, _ = _request(url, accept="application/vnd.github+json")
    n = 0 if status == HTTP_NOT_FOUND else len(json.loads(body))
    _cache_write(cache, {"n": n})
    return n


def count_pr_review_comments(owner: str, repo: str, pr_number: int) -> int:
    """Return the number of inline review-comments on the PR diff.

    AIDev's ``pr_review_comments_v2`` table is per-comment. Disk-cached.
    """
    cache = _cache_path("review_comments", owner, repo, str(pr_number))
    cached = _cache_read(cache)
    if cached is not None:
        return int(cached.get("n", 0))

    url = f"{API_BASE}/repos/{owner}/{repo}/pulls/{pr_number}/comments?per_page=100"
    status, body, _ = _request(url, accept="application/vnd.github+json")
    n = 0 if status == HTTP_NOT_FOUND else len(json.loads(body))
    _cache_write(cache, {"n": n})
    return n


def count_pr_issue_comments(owner: str, repo: str, pr_number: int) -> int:
    """Return the number of general issue-style comments on the PR.

    AIDev's ``pr_comments`` table is per-comment. GitHub treats PRs as issues
    for non-inline comments, so we hit the issues endpoint. Disk-cached.
    """
    cache = _cache_path("issue_comments", owner, repo, str(pr_number))
    cached = _cache_read(cache)
    if cached is not None:
        return int(cached.get("n", 0))

    url = f"{API_BASE}/repos/{owner}/{repo}/issues/{pr_number}/comments?per_page=100"
    status, body, _ = _request(url, accept="application/vnd.github+json")
    n = 0 if status == HTTP_NOT_FOUND else len(json.loads(body))
    _cache_write(cache, {"n": n})
    return n


def rate_limit_status() -> dict:
    """Return current rate-limit info (does NOT count against the limit)."""
    _, body, _ = _request(f"{API_BASE}/rate_limit",
                          accept="application/vnd.github+json")
    return json.loads(body)
