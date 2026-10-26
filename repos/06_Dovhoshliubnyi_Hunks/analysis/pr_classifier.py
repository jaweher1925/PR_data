import asyncio
import json
import logging

import anthropic
from tqdm import tqdm

from config import (
    ANTHROPIC_MODEL, ANTHROPIC_BATCH_MODEL, BODY_TRUNCATE_LENGTH, CLASSIFY_CONCURRENCY,
    CLASSIFICATION_BATCH_SIZE, RELEVANCE_THRESHOLD,
    LOCAL_MODEL_BASE_URL, LOCAL_MODEL_NAME,
)

logger = logging.getLogger(__name__)

CLASSIFICATION_PROMPT = """You are a software engineering researcher analyzing GitHub pull requests. Classify whether this PR directly improves the runtime performance of existing software.

INCLUDE (score 4-5):
- Optimizing algorithms or data structures for speed/memory
- Adding or improving caching mechanisms
- Reducing query complexity or database round-trips
- Improving concurrency, parallelism, or async efficiency
- Reducing memory allocations, copies, or leaks
- Optimizing I/O, serialization, or network calls
- Batch processing to reduce overhead
- Lazy loading or deferred computation
- Resource pooling (connections, threads, buffers)

EXCLUDE (score 1-2):
- Adding benchmarks, profiling tools, or performance monitoring (measuring != improving)
- CI/build checks like bundle size limits or lint rules
- New features motivated by scalability (adding a queue, new async pipeline)
- Compiler/transpiler features that improve generated code (unless the compiler itself is optimized)
- Dependency upgrades that happen to be faster (unless the PR explicitly targets performance)
- Test-only performance improvements
- Code style or readability refactors with no performance intent
- Bug fixes that restore previously broken performance or caching behavior (fixing != improving)
- New API methods or utilities that happen to use caching/pooling patterns (adding != optimizing)
- Removing dead code, deprecating features, or cleanup (maintenance != optimization)
- Architecture or infrastructure changes where performance is a side effect, not the stated goal

SCORING GUIDE (use only integers 1-5):
- 5: Entirely focused on runtime performance improvement with clear intent or measurements
- 4: Primary goal is performance, may include related cleanup
- 3: Performance is a secondary benefit, not the main goal
- 2: Incidental performance effect, not intentional
- 1: No relation to performance

CATEGORIES (use only from this list):
algorithmic_complexity, memory_data_locality, parallelism, loop_transformations,
build_compilation, network_db_access, io, concurrency, other

Respond ONLY with a JSON object in this exact format:
{{
  "relevance_score": <integer 1-5>,
  "categories_detected": [<categories from the list above>],
  "reasoning": "<one sentence>"
}}

PR title: <title>{title}</title>
PR body:
<body>{body}</body>
"""


def _parse_response(raw: str) -> tuple[float, str, str]:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("```", 2)[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()
    result = json.loads(raw)
    relevance_score = int(result["relevance_score"])
    categories_detected = json.dumps(result.get("categories_detected", []))
    reasoning = result.get("reasoning", "")
    return relevance_score, categories_detected, reasoning


async def _run_anthropic(prs, model, progress_bar, on_chunk, concurrency):
    client = anthropic.AsyncAnthropic()
    semaphore = asyncio.Semaphore(concurrency)

    async def classify_one(pr_id, title, body):
        truncated_body = (body or "")[:BODY_TRUNCATE_LENGTH]
        prompt = CLASSIFICATION_PROMPT.format(title=title or "", body=truncated_body)
        async with semaphore:
            try:
                while True:
                    try:
                        message = await client.messages.create(
                            model=model,
                            max_tokens=256,
                            messages=[{"role": "user", "content": prompt}],
                        )
                        relevance_score, categories_detected, reasoning = _parse_response(
                            message.content[0].text
                        )
                        return (pr_id, relevance_score, categories_detected, reasoning, None)
                    except anthropic.RateLimitError as exc:
                        wait = None
                        if exc.response is not None:
                            wait = exc.response.headers.get("retry-after") or \
                                   exc.response.headers.get("x-ratelimit-reset-requests")
                        wait = float(wait) if wait is not None else 60.0
                        logger.warning("Rate limit hit for PR %d — waiting %.1fs", pr_id, wait)
                        await asyncio.sleep(wait)
                    except Exception as exc:
                        logger.warning("Skipping PR %d due to error: %s", pr_id, exc)
                        return (pr_id, None, None, None, exc)
            finally:
                progress_bar.update(1)

    try:
        for chunk_start in range(0, len(prs), CLASSIFICATION_BATCH_SIZE):
            chunk = prs[chunk_start:chunk_start + CLASSIFICATION_BATCH_SIZE]
            results = await asyncio.gather(*[classify_one(*pr) for pr in chunk])
            on_chunk(results)
    finally:
        await client.close()


async def _run_local(prs, model, progress_bar, on_chunk, concurrency):
    import openai
    client = openai.AsyncOpenAI(base_url=LOCAL_MODEL_BASE_URL, api_key="not-needed")
    semaphore = asyncio.Semaphore(concurrency)

    async def classify_one(pr_id, title, body):
        truncated_body = (body or "")[:BODY_TRUNCATE_LENGTH]
        prompt = CLASSIFICATION_PROMPT.format(title=title or "", body=truncated_body)
        async with semaphore:
            try:
                response = await client.chat.completions.create(
                    model=model,
                    max_tokens=256,
                    messages=[{"role": "user", "content": prompt}],
                )
                relevance_score, categories_detected, reasoning = _parse_response(
                    response.choices[0].message.content
                )
                return (pr_id, relevance_score, categories_detected, reasoning, None)
            except Exception as exc:
                logger.warning("Skipping PR %d due to error: %s", pr_id, exc)
                return (pr_id, None, None, None, exc)
            finally:
                progress_bar.update(1)

    try:
        for chunk_start in range(0, len(prs), CLASSIFICATION_BATCH_SIZE):
            chunk = prs[chunk_start:chunk_start + CLASSIFICATION_BATCH_SIZE]
            results = await asyncio.gather(*[classify_one(*pr) for pr in chunk])
            on_chunk(results)
    finally:
        await client.aclose()


def submit_batch(conn, model=ANTHROPIC_BATCH_MODEL):
    """Submit all unclassified PRs to the Anthropic Batch API. Prints the batch ID."""
    cur = conn.cursor()
    cur.execute("""
        SELECT p.id, p.title, p.body
        FROM pull_request p
        LEFT JOIN pr_performance_classification ppc ON p.id = ppc.pr_id
        WHERE p.merged_at IS NOT NULL
          AND ppc.pr_id IS NULL
    """)
    prs = cur.fetchall()

    if not prs:
        print("No unclassified PRs found.")
        return

    client = anthropic.Anthropic()
    requests = []
    for pr_id, title, body in prs:
        truncated_body = (body or "")[:BODY_TRUNCATE_LENGTH]
        prompt = CLASSIFICATION_PROMPT.format(title=title or "", body=truncated_body)
        requests.append({
            "custom_id": str(pr_id),
            "params": {
                "model": model,
                "max_tokens": 256,
                "messages": [{"role": "user", "content": prompt}],
            },
        })

    print(f"Submitting {len(requests)} PRs to Anthropic Batch API ...")
    batch = client.messages.batches.create(requests=requests)
    print(f"Batch submitted successfully.")
    print(f"  Batch ID : {batch.id}")
    print(f"  Status   : {batch.processing_status}")
    print(f"  PRs      : {len(requests)}")
    print(f"\nSave this batch ID — run with --batch-collect {batch.id} once processing ends.")
    return batch.id


def collect_batch(conn, batch_id):
    """Download results from a completed Anthropic Batch job and save to DB."""
    client = anthropic.Anthropic()

    batch = client.messages.batches.retrieve(batch_id)
    if batch.processing_status != "ended":
        print(f"Batch {batch_id} is not finished yet (status: {batch.processing_status}).")
        print("Check back later.")
        return {"total_classified": 0, "relevant_count": 0, "skipped_errors": 0}

    cur = conn.cursor()
    total_classified = 0
    relevant_count = 0
    skipped_errors = 0
    chunk = []
    CHUNK_SIZE = 500

    def flush():
        nonlocal total_classified, relevant_count
        for pr_id, relevance_score, categories_detected, reasoning in chunk:
            cur.execute(
                """
                INSERT OR REPLACE INTO pr_performance_classification
                    (pr_id, relevance_score, categories_detected, reasoning)
                VALUES (?, ?, ?, ?)
                """,
                (pr_id, relevance_score, categories_detected, reasoning),
            )
            total_classified += 1
            if relevance_score >= RELEVANCE_THRESHOLD:
                relevant_count += 1
        conn.commit()
        chunk.clear()

    print(f"Downloading results for batch {batch_id} ...")
    for result in client.messages.batches.results(batch_id):
        if result.result.type == "succeeded":
            try:
                text = result.result.message.content[0].text
                relevance_score, categories_detected, reasoning = _parse_response(text)
                chunk.append((int(result.custom_id), relevance_score, categories_detected, reasoning))
                if len(chunk) >= CHUNK_SIZE:
                    flush()
            except Exception as exc:
                logger.warning("Failed to parse result for custom_id %s: %s", result.custom_id, exc)
                skipped_errors += 1
        else:
            logger.warning("Non-success result for custom_id %s: type=%s", result.custom_id, result.result.type)
            skipped_errors += 1

    if chunk:
        flush()

    return {"total_classified": total_classified, "relevant_count": relevant_count, "skipped_errors": skipped_errors}


def classify_performance_prs(conn, model=ANTHROPIC_MODEL, local=False, concurrency=None):
    cur = conn.cursor()

    cur.execute("""
        SELECT p.id, p.title, p.body
        FROM pull_request p
        LEFT JOIN pr_performance_classification ppc ON p.id = ppc.pr_id
        WHERE p.merged_at IS NOT NULL
          AND ppc.pr_id IS NULL
    """)
    prs = cur.fetchall()

    total_classified = 0
    relevant_count = 0
    skipped_errors = 0

    def on_chunk(results):
        nonlocal total_classified, relevant_count, skipped_errors
        for pr_id, relevance_score, categories_detected, reasoning, error in results:
            if error is not None:
                skipped_errors += 1
                continue
            cur.execute(
                """
                INSERT OR REPLACE INTO pr_performance_classification
                    (pr_id, relevance_score, categories_detected, reasoning)
                VALUES (?, ?, ?, ?)
                """,
                (pr_id, relevance_score, categories_detected, reasoning),
            )
            total_classified += 1
            if relevance_score >= RELEVANCE_THRESHOLD:
                relevant_count += 1
        conn.commit()

    if local:
        effective_model = model if model != ANTHROPIC_MODEL else LOCAL_MODEL_NAME
        run_fn = _run_local
    else:
        effective_model = model
        run_fn = _run_anthropic

    effective_concurrency = concurrency if concurrency is not None else CLASSIFY_CONCURRENCY
    with tqdm(total=len(prs), desc="Classifying PRs") as progress_bar:
        asyncio.run(run_fn(prs, effective_model, progress_bar, on_chunk, effective_concurrency))

    return {
        "total_classified": total_classified,
        "relevant_count": relevant_count,
        "skipped_errors": skipped_errors,
    }
