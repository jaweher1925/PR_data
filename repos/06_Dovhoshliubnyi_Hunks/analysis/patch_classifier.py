"""Patch-level mutation category classification.

Classifies each patch from validated performance PRs into one or more
of the 18 SSBSE syntactic mutation categories. These patches come from
the MSR paper's validated dataset (324 agent perf PRs).
"""
import asyncio
import csv
import json
import logging
import re

import anthropic
import json5
from tqdm import tqdm

from config import ANTHROPIC_MODEL, CLASSIFY_CONCURRENCY, MIN_CONFIDENCE, PATCH_TRUNCATE_LENGTH

logger = logging.getLogger(__name__)

MUTATION_CATEGORIES = {
    0:  ("added_code_from_github",   "Added (some arbitrary) code from GitHub"),
    1:  ("no_change",                "No change"),
    2:  ("comment_modification",     "Modified a comment (add/remove/edit)"),
    3:  ("deleted_blocks",           "Deleted blocks in a method (all/most/some)"),
    4:  ("duplicate_code",           "Duplicate code"),
    5:  ("return_statement_mods",    "Modifications to return statements (add/remove/edit)"),
    6:  ("method_name_change",       "Changes to method names"),
    7:  ("type_change",              "Changed data types or type usage and generics"),
    8:  ("inlining",                 "Inlining of implementations (sort, methods...)"),
    9:  ("exception_handling",       "Added exception-handling constructs (unreachable or reachable)"),
    10: ("extra_brackets",           "Added extra brackets"),
    11: ("synchronization",          "Added synchronization logic"),
    12: ("name_modification",        "Modified Variable/Class/Object Name"),
    13: ("control_flow",             "Modified Control Flow Structure"),
    14: ("object_creation",          "Modified Object/Primitive Creation or Initialization"),
    15: ("statement_splitting",      "Split a statement into multiple lines"),
    16: ("arithmetic_manipulation",  "Arithmetic manipulation (boolean var. or expr. manipulations)"),
    17: ("dead_code",                "Added dead code"),
}

VALID_IDS = set(MUTATION_CATEGORIES.keys())

PATCH_CLASSIFICATION_PROMPT = """You are a software engineering researcher analyzing a code diff from a known performance-improvement pull request.

Your task: identify which syntactic code mutation patterns are present in this diff.

## Mutation categories (use IDs 0–17):

 0  added_code_from_github     — Added (some arbitrary) code from GitHub
 1  no_change                  — No meaningful code change
 2  comment_modification       — Modified a comment (add/remove/edit)
 3  deleted_blocks             — Deleted blocks in a method (all/most/some)
 4  duplicate_code             — Duplicate code
 5  return_statement_mods      — Modifications to return statements (add/remove/edit: a return keyword must be added, removed, or have its returned expression changed)
 6  method_name_change         — Changes to method/function names (the method being defined or called is renamed)
 7  type_change                — The declared data type of a variable, parameter, return value, field, or collection changes (e.g. int→long, HashMap→FxHashMap, String→StringBuilder, List<T>→List<T>?, Array→Vector). Swapping one library for another with the same API (e.g. chalk→picocolors) is name_modification, NOT type_change, unless the actual type signature changes.
 8  inlining                   — Replacing a function/method CALL with the function's body (expanding code at the call site). Consider following: 1. extracting inline code into a new function is not inlining. If code moves FROM a call site INTO a new function, that is extraction 2. The function/method should be removed from hunk to prove the inlining has happened
 9  exception_handling         — Added exception-handling constructs
10  extra_brackets             — Added extra brackets
11  synchronization            — Added synchronization logic. For example implementation of Locks (Mutexes), Semaphores, multi-threading, Atomic Operations, etc
12  name_modification          — Modified variable, function, class, Object or module name.
13  control_flow               — Modified Control Flow Structure (if/else, loops, switch, ternary, Jump Statements)
14  object_creation            — Modified Object/Primitive Creation or Initialization (new constructor call, changed initializer, new collection literal). Caching a value into a local variable is statement_splitting, not object_creation. Importing new modules/libraries are object_creation. Change in the same imported modules/libraries with new object imported also counts as object_creation.
15  statement_splitting        — Split a statement into multiple lines, or extracted sub-expressions into intermediate variables
16  arithmetic_manipulation    — An arithmetic (+,-,*,/), comparison (<,>,==,!=), or boolean (&&,||,!) OPERATOR is added, removed, or changed, OR a numeric/boolean LITERAL is changed. NOT: adding/removing function arguments, null-coalescing (??/??=), method call substitution, quote style changes, or reordering method arguments.
17  dead_code                  — Added dead code

## Rules:
- A patch can match MULTIPLE categories (multi-label)
- Only include categories clearly present in the diff — when in doubt, leave it out
- If the patch has no meaningful code change, use category 1 (no_change)
- When the declared data type changes (e.g. HashMap→FxHashMap, String→StringBuilder, List<T>→List<T>?), ALWAYS include type_change even if name_modification also applies
- Whitespace-only changes (trailing spaces, blank lines, indentation, BOM markers) are no_change
- Build config files (Cargo.toml, package.json, .csproj, .xlf) that only add/remove dependencies or change versions are no_change
- Documentation-only files (.md, .txt, .rst) added or changed are no_change ONLY if there are no code changes in the same patch. A new .md file in an otherwise code-changing PR is still no_change for that specific file patch.
- Do NOT use category 5 (return_statement_mods) unless a return/yield keyword is explicitly added, removed, or its returned expression is directly modified
- Do NOT confuse extraction with inlining: if code is moved FROM inline INTO a new named function, that is NOT category 8
- Do NOT speculate about the intent of a change — classify based on the actual code changes in the diff, not what you think the developer's motivation might have been

File: {filename}

```diff
{patch}
```

For each category, include `start_line` and `end_line` integer fields indicating the line number range in the diff where this pattern occurs. Count lines starting from 1, where line 1 is the `@@` header line itself. Do NOT use the file line numbers shown inside the `@@ -X,Y +A,B @@` header — those are source offsets, not diff line numbers.

Respond ONLY with this JSON (no markdown fences):
{{"categories": [{{"id": <int 0-17>, "name": "<category_name>", "confidence": <float 0.0-1.0>, "start_line": <int>, "end_line": <int>}}], "reasoning": "<one sentence>"}}
"""


def _parse_response(raw: str) -> tuple[list, str]:
    raw = raw.strip()
    # Extract JSON from markdown fences anywhere in the response
    fence_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', raw, re.DOTALL)
    if fence_match:
        raw = fence_match.group(1).strip()
    # Find first '{' and use raw_decode to tolerate trailing text
    idx = raw.find('{')
    if idx == -1:
        raise ValueError(f"No JSON object found in response: {raw[:120]}")
    try:
        result, _ = json.JSONDecoder().raw_decode(raw, idx)
    except json.JSONDecodeError:
        # Fallback: json5 handles JS-style JSON (single quotes, unquoted keys, trailing commas)
        result = json5.loads(raw[idx:])
    categories = [
        {
            "id": c["id"],
            "name": c.get("name", ""),
            "confidence": float(c.get("confidence", 0.0)),
            "start_line": c.get("start_line"),
            "end_line": c.get("end_line"),
        }
        for c in result.get("categories", [])
        if isinstance(c.get("id"), int) and c["id"] in VALID_IDS
           and float(c.get("confidence", 0.0)) >= MIN_CONFIDENCE
    ]
    reasoning = result.get("reasoning", "")
    return categories, reasoning


def _build_prompt(filename: str, patch: str) -> str:
    truncated_patch = (patch or "")[:PATCH_TRUNCATE_LENGTH]
    return PATCH_CLASSIFICATION_PROMPT.format(
        filename=filename or "",
        patch=truncated_patch,
    )


def classify_patches(conn, model=ANTHROPIC_MODEL, concurrency=None):
    """Classify all unclassified patches in commit_details."""
    rows = conn.execute("""
        SELECT cd.pr_id, cd.sha, cd.filename, cd.patch
        FROM commit_details cd
        JOIN perf_pr p ON p.pr_id = cd.pr_id
        LEFT JOIN patch_classification pc
            ON pc.pr_id = cd.pr_id AND pc.sha = cd.sha AND pc.filename = cd.filename
        WHERE pc.pr_id IS NULL
          AND cd.patch IS NOT NULL
          AND LENGTH(cd.patch) > 0
          AND p.pr_url NOT LIKE '%mochilang/mochi%'
    """).fetchall()

    if not rows:
        print("No unclassified patches found.")
        return {"total": 0, "perf_count": 0, "non_perf_count": 0, "errors": 0}

    effective_concurrency = concurrency or CLASSIFY_CONCURRENCY
    total = perf_count = non_perf_count = errors = 0

    def on_chunk(results):
        nonlocal total, perf_count, non_perf_count, errors
        for pr_id, sha, filename, categories, reasoning, error in results:
            if error is not None:
                errors += 1
                continue
            # is_performance = has at least one non-trivial category
            trivial = {1, 2}  # no_change, comment_modification
            has_meaningful = any(c["id"] not in trivial for c in categories)
            conn.execute(
                """
                INSERT OR REPLACE INTO patch_classification
                    (pr_id, sha, filename, is_performance, categories, confidence, reasoning)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (pr_id, sha, filename, 1 if has_meaningful else 0,
                 json.dumps(categories),
                 max((c["confidence"] for c in categories), default=0.0),
                 reasoning),
            )
            total += 1
            if has_meaningful:
                perf_count += 1
            else:
                non_perf_count += 1
        conn.commit()

    async def run():
        client = anthropic.AsyncAnthropic()
        semaphore = asyncio.Semaphore(effective_concurrency)

        async def classify_one(pr_id, sha, filename, patch):
            prompt = _build_prompt(filename, patch)
            async with semaphore:
                try:
                    while True:
                        try:
                            msg = await client.messages.create(
                                model=model,
                                max_tokens=256,
                                messages=[{"role": "user", "content": prompt}],
                            )
                            categories, reasoning = _parse_response(msg.content[0].text)
                            return (pr_id, sha, filename, categories, reasoning, None)
                        except anthropic.RateLimitError as exc:
                            wait = None
                            if exc.response is not None:
                                wait = exc.response.headers.get("retry-after") or \
                                       exc.response.headers.get("x-ratelimit-reset-requests")
                            wait = float(wait) if wait is not None else 60.0
                            logger.warning("Rate limit — waiting %.1fs", wait)
                            await asyncio.sleep(wait)
                        except Exception as exc:
                            logger.warning("Skipping %s/%s: %s", sha[:8], filename, exc)
                            return (pr_id, sha, filename, None, None, exc)
                finally:
                    progress_bar.update(1)

        with tqdm(total=len(rows), desc="Classifying patches") as progress_bar:
            results = await asyncio.gather(*[classify_one(*row) for row in rows])
        on_chunk(results)
        await client.close()

    asyncio.run(run())

    # Export CSV
    export_csv(conn)

    return {
        "total": total,
        "perf_count": perf_count,
        "non_perf_count": non_perf_count,
        "errors": errors,
    }


def export_csv(conn, path="classification_results.csv"):
    """Export patch_classification joined with perf_pr to CSV."""
    rows = conn.execute("""
        SELECT pc.pr_id, p.agent, p.pattern, p.sub_pattern,
               pc.sha, pc.filename, pc.is_performance,
               pc.categories, pc.confidence, pc.reasoning
        FROM patch_classification pc
        JOIN perf_pr p ON p.pr_id = pc.pr_id
        ORDER BY pc.pr_id, pc.sha, pc.filename
    """).fetchall()

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "pr_id", "agent", "msr_pattern", "msr_sub_pattern",
            "sha", "filename", "is_performance",
            "categories", "confidence", "reasoning",
        ])
        for row in rows:
            writer.writerow(row)

    print(f"Exported {len(rows)} rows to {path}")
