# SSBSE 2026: Performance-Related Patch Classification from AIDev Dataset

## Overview

This project builds a Python pipeline that mines the [AIDev dataset](https://huggingface.co/datasets/hao-li/AIDev) to identify GitHub issues related to software performance improvement, collects the code patches contributed to resolve those issues, and classifies each patch into a taxonomy of 18 change categories.

**Challenge:** 2026 SSBSE Challenge
**Dataset:** hao-li/AIDev on HuggingFace (AIDev-pop subset: repos with >100 stars)
**Language:** Python
**LLM:** Cloud API (OpenAI or Anthropic)
**Output:** SQLite database + CSV exports

## Goal

Produce a table of classified patches with the schema:

| Column | Type | Description |
|--------|------|-------------|
| issue_id | int | ID of the performance-related issue |
| pr_id | int | ID of the pull request that addressed the issue |
| sha | str | Commit SHA containing the patch |
| filename | str | File path of the changed file |
| patch | str | Unified diff of the change |
| categories | list[{id, confidence}] | JSON array of {id: int, confidence: float} per applicable category |
| reasoning | str | LLM explanation of categorization |

## Data Model & Join Path

### Tables Used (AIDev-pop subset)

- **issue** (4.61k rows): id, number, title, body, user, state, created_at, closed_at, html_url
- **related_issue** (4.92k rows): pr_id, issue_id, source
- **pull_request** (33.6k rows): id, number, title, body, agent, user_id, user, state, created_at, closed_at, merged_at, repo_id, repo_url, html_url
- **pr_commit_details** (712k rows): sha, pr_id, author, committer, message, commit_stats_total, commit_stats_additions, commit_stats_deletions, filename, status, additions, deletions, changes, patch

### Join Chain

```
issue (id)
  -> related_issue (issue_id -> pr_id)
    -> pull_request (id)
      -> pr_commit_details (pr_id -> sha, filename, patch)
```

### Filtering Constraints

1. **issue.state = 'closed'** — only resolved issues
2. **pull_request.merged_at IS NOT NULL** — only PRs that were actually merged (closed-without-merge PRs didn't land code)
3. **issue must have at least one related PR** — via related_issue join
4. **pr_commit_details.filename must be business logic** — non-test, non-docs, non-config (see exclusion patterns below)

## Pipeline Architecture

Four sequential stages, each producing intermediate output persisted in SQLite:

```
Stage 1: Load & Join Data
  Input:  HuggingFace parquet files
  Output: SQLite DB with raw tables + filtered join view

Stage 2: Issue Classification (AI)
  Input:  Issues with related closed PRs (~1-2k after filtering)
  Output: issue_classification table (issue_id, relevance_score, categories_detected, reasoning)

Stage 3: Patch Collection & Filtering
  Input:  Issues scoring >= threshold + pr_commit_details
  Output: business_logic_patches table (issue_id, pr_id, sha, filename, patch)

Stage 4: Patch Categorization (AI)
  Input:  Filtered patches from Stage 3
  Output: patch_classification table (final output)
  Export: SQLite + CSV
```

### Stage 1: Load & Join Data

Downloads parquet files from HuggingFace using the `datasets` library. Loads into SQLite for efficient querying. Creates a filtered view joining issue -> related_issue -> pull_request where both issue and PR are closed.

### Stage 2: Issue Classification

An LLM classifies each issue by title and body for performance relevance. Wide-net approach: casts broadly for anything related to optimization, refactoring, complexity reduction, caching, memory, concurrency, algorithmic improvements, latency, throughput, scalability, efficiency.

**Output per issue:**
- `relevance_score`: 0.0 to 1.0 float
- `categories_detected`: JSON array of tags (e.g., ["optimization", "refactoring", "caching"])
- `reasoning`: one-sentence explanation

Issues processed individually (not batched) since body text can be large. Body truncated to 2000 tokens max. Results cached in SQLite — re-running skips already-classified issues.

### Stage 3: Patch Collection & Filtering

Joins issues scoring above the configurable threshold (default 0.5) through to pr_commit_details. Filters out non-business-logic files using exclusion patterns:

```python
EXCLUDED_PATTERNS = [
    r"(^|/)tests?/",
    r"(^|/)spec/",
    r"_test\.\w+$",
    r"_spec\.\w+$",
    r"\.test\.\w+$",
    r"\.spec\.\w+$",
    r"(^|/)__(tests|test)__/",
    r"(^|/)(README|CHANGELOG|CHANGES|LICENSE|CONTRIBUTING)",
    r"(^|/)\.github/",
    r"(^|/)(ci|\.circleci|\.travis)/",
    r"(package-lock\.json|yarn\.lock|Pipfile\.lock|poetry\.lock)$",
    r"(^|/)docs?/",
]
```

Patches with null/empty `patch` field are excluded (the GitHub API omits content for very large patches).

### Stage 4: Patch Categorization

An LLM classifies each patch into one or more of 18 categories. This taxonomy is externally defined (from prior SBSE/patch classification literature) and includes categories like "No change" and "Added dead code" that may seem unusual in a performance context but are retained for completeness:

| # | Description |
|---|-------------|
| 0 | Added (some arbitrary) code from GitHub |
| 1 | No change |
| 2 | Modified a comment (add/remove/edit) |
| 3 | Deleted blocks in a method (all/most/some) |
| 4 | Duplicate code |
| 5 | Modifications to return statements (add/remove/edit) |
| 6 | Changes to method names |
| 7 | Changed data types or type usage and generics |
| 8 | Includes inlining of implementations (sort, methods...) |
| 9 | Added exception-handling constructs (unreachable or reachable) |
| 10 | Added extra brackets |
| 11 | Added synchronization logic |
| 12 | Modified Variable/Class/Object Name |
| 13 | Modified Control Flow Structure |
| 14 | Modified Object/Primitive Creation or Initialization |
| 15 | Split a statement into multiple lines |
| 16 | Arithmetic manipulation (boolean var. or expr. manipulations) |
| 17 | Added dead code |

**Output per patch:**
- `categories`: list of {id, confidence} objects
- `reasoning`: brief explanation

Patches processed individually. Patch text truncated to 4000 tokens max. Results cached for resumability.

## LLM Prompt Design

### Issue Classification Prompt

```
You are analyzing GitHub issues to determine if they relate to
software performance improvement. This includes: optimization,
refactoring, complexity reduction, caching, memory optimization,
concurrency improvements, algorithmic improvements, latency reduction,
throughput improvement, scalability, and efficiency gains.

Given the following GitHub issue title and body, score how relevant
this issue is to performance improvement on a scale of 0.0 to 1.0.

Title: {title}
Body: {body_truncated}

Respond in JSON:
{
  "relevance_score": <float 0.0-1.0>,
  "categories_detected": [<list of tags>],
  "reasoning": "<one sentence>"
}
```

### Patch Categorization Prompt

```
You are classifying a code patch into one or more of these categories:

0: Added arbitrary code from GitHub
1: No change
2: Modified a comment (add/remove/edit)
3: Deleted blocks in a method (all/most/some)
4: Duplicate code
5: Modifications to return statements (add/remove/edit)
6: Changes to method names
7: Changed data types or type usage and generics
8: Includes inlining of implementations (sort, methods...)
9: Added exception-handling constructs (unreachable or reachable)
10: Added extra brackets
11: Added synchronization logic
12: Modified Variable/Class/Object Name
13: Modified Control Flow Structure
14: Modified Object/Primitive Creation or Initialization
15: Split a statement into multiple lines
16: Arithmetic manipulation (boolean var. or expr. manipulations)
17: Added dead code

Given this patch from file "{filename}":

```diff
{patch}
```

Assign ALL applicable categories. Respond in JSON:
{
  "categories": [{"id": <int>, "confidence": <float 0.0-1.0>}],
  "reasoning": "<brief explanation>"
}
```

## Configuration

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `LLM_PROVIDER` | `openai` | `openai` or `anthropic` |
| `LLM_MODEL` | `gpt-4o-mini` | Model for both classification tasks |
| `RELEVANCE_THRESHOLD` | `0.5` | Minimum score to consider an issue performance-related |
| `MAX_BODY_TOKENS` | `2000` | Truncation limit for issue body |
| `MAX_PATCH_TOKENS` | `4000` | Truncation limit for patch text |
| `BATCH_DELAY_MS` | `100` | Delay between API calls |
| `DB_PATH` | `data/pipeline.db` | SQLite database path |
| `CSV_OUTPUT_DIR` | `output/` | CSV export directory |

## Project Structure

```
ssbse-challenge-2026/
  src/
    config.py              # Configuration and constants
    stage1_load.py         # Download parquet, load into SQLite
    stage2_classify.py     # AI classification of issues
    stage3_patches.py      # Join to patches, filter business logic
    stage4_categorize.py   # AI categorization of patches (0-17)
    export.py              # Export final results to CSV
    prompts.py             # All LLM prompt templates
    llm_client.py          # API wrapper with retry/backoff/caching
    filters.py             # File path exclusion patterns
    db.py                  # SQLite helpers
  run_pipeline.py          # Orchestrator: runs stages 1-4
  requirements.txt
  .env.example
  docs/
    specs/               # This design doc
    plans/               # Implementation plan
```

## SQLite Schema

```sql
-- Raw tables (loaded from parquet, columns match dataset)
CREATE TABLE issue (
  id INTEGER PRIMARY KEY, number INTEGER, title TEXT, body TEXT,
  user TEXT, state TEXT, created_at TEXT, closed_at TEXT, html_url TEXT
);

CREATE TABLE related_issue (
  pr_id INTEGER, issue_id INTEGER, source TEXT
);

CREATE TABLE pull_request (
  id INTEGER PRIMARY KEY, number INTEGER, title TEXT, body TEXT,
  agent TEXT, user_id INTEGER, user TEXT, state TEXT,
  created_at TEXT, closed_at TEXT, merged_at TEXT,
  repo_id INTEGER, repo_url TEXT, html_url TEXT
);

CREATE TABLE pr_commit_details (
  sha TEXT, pr_id INTEGER, author TEXT, committer TEXT, message TEXT,
  commit_stats_total INTEGER, commit_stats_additions INTEGER,
  commit_stats_deletions INTEGER, filename TEXT, status TEXT,
  additions INTEGER, deletions INTEGER, changes INTEGER, patch TEXT,
  PRIMARY KEY (sha, pr_id, filename)
);

-- Indexes for join performance
CREATE INDEX idx_related_issue_issue_id ON related_issue(issue_id);
CREATE INDEX idx_related_issue_pr_id ON related_issue(pr_id);
CREATE INDEX idx_pr_commit_details_pr_id ON pr_commit_details(pr_id);

-- Stage 2 output
CREATE TABLE issue_classification (
  issue_id INTEGER PRIMARY KEY,
  title TEXT,
  relevance_score REAL,
  categories_detected TEXT,  -- JSON array
  reasoning TEXT
);

-- Stage 4 output (final)
CREATE TABLE patch_classification (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  issue_id INTEGER,
  pr_id INTEGER,
  sha TEXT,
  filename TEXT,
  patch TEXT,
  categories TEXT,    -- JSON array of {id, confidence} objects
  reasoning TEXT
);
```

## Error Handling & Resilience

- **API errors:** Exponential backoff with jitter on 429/500/502/503 errors, max 5 retries
- **Response validation:** LLM JSON responses validated for required fields, correct types, and score ranges (0.0-1.0). Malformed responses trigger a single retry with a stricter prompt; if still invalid, the item is logged as failed and skipped.
- **Resumability:** Results cached in SQLite per-item. Re-running any stage skips already-processed items.
- **Dry run:** `--dry-run` flag estimates API costs (counts items, token estimates) without making calls
- **Logging:** Python logging module, INFO level by default, DEBUG for troubleshooting

## Cost Estimates

- **Stage 2:** ~1-2k issue classifications at ~500-1000 tokens per call = ~$1-3 (gpt-4o-mini)
- **Stage 4:** Depends on patch count after filtering. Estimated hundreds to low thousands of patches at ~1000-2000 tokens per call = ~$5-15 (gpt-4o-mini)
- **Total estimated cost:** $6-18

## Risks

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| Few performance-related issues found | Medium | Wide-net approach + adjustable threshold. Even small datasets are publishable. |
| Large patches truncated, losing context | Medium | Truncation with note to LLM. Can increase token limit for specific cases. |
| LLM misclassifies patches | Medium | Multi-label approach, confidence scores, manual spot-check of sample. |
| API rate limits slow pipeline | Low | Configurable delay, exponential backoff, resumability. |
| pr_commit_details missing patches for large diffs | Known | Dataset limitation documented by authors. We exclude null patches. |
