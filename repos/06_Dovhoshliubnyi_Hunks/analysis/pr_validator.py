"""Binary validation pass for candidate performance PRs.

Uses enriched input (title + body + commit messages + repo URL) and an
exclusion-focused binary prompt to reduce false positives from the
initial 1-5 relevance classifier.
"""
import asyncio
import json
import logging

import anthropic
from tqdm import tqdm

from config import (
    ANTHROPIC_MODEL,
    BODY_TRUNCATE_LENGTH,
    CLASSIFY_CONCURRENCY,
    COMMIT_MSG_TRUNCATE_LENGTH,
)

logger = logging.getLogger(__name__)

VALIDATION_PROMPT = """You are a software engineering researcher. Decide whether this GitHub pull request is a **runtime performance improvement** of existing software.

Answer TRUE only if the PR makes already-existing, already-running code **faster or use less memory**. Answer FALSE for everything else.

ALWAYS answer FALSE for:
- Compiler/transpiler PRs that improve the *quality of code they generate* (better codegen ≠ the compiler running faster)
- Adding new features, new endpoints, new programs, new capabilities (adding ≠ optimizing)
- Bug fixes or correctness fixes (fixing broken behavior ≠ improving performance)
- Build system, CI, config, or tooling changes
- Code style, refactoring, or cleanup with no explicit performance intent
- Monitoring, observability, or metrics tooling (measuring ≠ improving)
- Dependency upgrades that happen to be faster (unless the PR explicitly targets performance)

ONLY answer TRUE if:
- The PR explicitly reduces CPU time, memory usage, I/O latency, or network calls of existing code
- OR adds caching/memoization to avoid redundant computation in existing code
- OR improves concurrency/parallelism in an existing code path
- OR replaces an existing algorithm/data structure with a faster one

Repo: {repo_url}
PR title: {title}
PR body:
<body>{body}</body>
Commit messages:
<commits>{commit_messages}</commits>

Respond ONLY with this JSON:
{{
  "is_performance_related": <true or false>,
  "reasoning": "<one sentence explaining the decision>"
}}
"""


def _build_enriched_input(conn, pr_ids):
    """Fetch title, body, commit messages, and repo_url for each PR."""
    placeholders = ",".join("?" * len(pr_ids))
    rows = conn.execute(
        f"""
        SELECT p.id, p.title, p.body, p.repo_url,
               GROUP_CONCAT(DISTINCT pcd.message) as commit_messages
        FROM pull_request p
        LEFT JOIN pr_commit_details pcd ON pcd.pr_id = p.id
        WHERE p.id IN ({placeholders})
        GROUP BY p.id
        """,
        pr_ids,
    ).fetchall()
    return {row[0]: row for row in rows}


def _parse_response(raw: str) -> tuple[bool, str]:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("```", 2)[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()
    result = json.loads(raw)
    is_perf = bool(result["is_performance_related"])
    reasoning = result.get("reasoning", "")
    return is_perf, reasoning


def _build_prompt(row) -> str:
    pr_id, title, body, repo_url, commit_messages = row
    truncated_body = (body or "")[:BODY_TRUNCATE_LENGTH]
    msgs = list(dict.fromkeys((commit_messages or "").split(",")))
    commit_str = " | ".join(m.strip() for m in msgs if m.strip())
    commit_str = commit_str[:COMMIT_MSG_TRUNCATE_LENGTH]
    return VALIDATION_PROMPT.format(
        repo_url=repo_url or "",
        title=title or "",
        body=truncated_body,
        commit_messages=commit_str,
    )


def validate_performance_prs(conn, model=ANTHROPIC_MODEL, concurrency=None):
    """Validate all score>=4 unvalidated PRs using direct async API calls."""
    cur = conn.cursor()
    cur.execute("""
        SELECT ppc.pr_id
        FROM pr_performance_classification ppc
        LEFT JOIN pr_validation pv ON ppc.pr_id = pv.pr_id
        WHERE ppc.relevance_score >= 4
          AND pv.pr_id IS NULL
    """)
    pr_ids = [row[0] for row in cur.fetchall()]

    if not pr_ids:
        print("No unvalidated PRs found.")
        return {"total": 0, "true_count": 0, "false_count": 0, "errors": 0}

    pr_data = _build_enriched_input(conn, pr_ids)
    effective_concurrency = concurrency or CLASSIFY_CONCURRENCY

    total = true_count = false_count = errors = 0

    def on_chunk(results):
        nonlocal total, true_count, false_count, errors
        for pr_id, is_perf, reasoning, error in results:
            if error is not None:
                errors += 1
                continue
            cur.execute(
                "INSERT OR REPLACE INTO pr_validation (pr_id, is_performance_related, reasoning) VALUES (?, ?, ?)",
                (pr_id, 1 if is_perf else 0, reasoning),
            )
            total += 1
            if is_perf:
                true_count += 1
            else:
                false_count += 1
        conn.commit()

    async def run():
        client = anthropic.AsyncAnthropic()
        semaphore = asyncio.Semaphore(effective_concurrency)

        async def validate_one(pr_id):
            row = pr_data.get(pr_id)
            if row is None:
                return (pr_id, None, None, ValueError("PR not found in DB"))
            prompt = _build_prompt(row)
            async with semaphore:
                try:
                    while True:
                        try:
                            message = await client.messages.create(
                                model=model,
                                max_tokens=256,
                                messages=[{"role": "user", "content": prompt}],
                            )
                            is_perf, reasoning = _parse_response(message.content[0].text)
                            return (pr_id, is_perf, reasoning, None)
                        except anthropic.RateLimitError as exc:
                            wait = None
                            if exc.response is not None:
                                wait = exc.response.headers.get("retry-after") or \
                                       exc.response.headers.get("x-ratelimit-reset-requests")
                            wait = float(wait) if wait is not None else 60.0
                            logger.warning("Rate limit hit for PR %d — waiting %.1fs", pr_id, wait)
                            await asyncio.sleep(wait)
                        except Exception as exc:
                            logger.warning("Skipping PR %d: %s", pr_id, exc)
                            return (pr_id, None, None, exc)
                finally:
                    progress_bar.update(1)

        with tqdm(total=len(pr_ids), desc="Validating PRs") as progress_bar:
            results = await asyncio.gather(*[validate_one(pr_id) for pr_id in pr_ids])
        on_chunk(results)
        await client.close()

    asyncio.run(run())
    return {"total": total, "true_count": true_count, "false_count": false_count, "errors": errors}
