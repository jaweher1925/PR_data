# SSBSE Project Explanation

## Project Overview

SSBSE is a data mining pipeline for the SSBSE 2026 Challenge Track. It mines the HuggingFace [`hao-li/AIDev`](https://huggingface.co/datasets/hao-li/AIDev) dataset — 24,014 merged pull requests authored by five AI coding agents — to identify performance improvement patches and classify them by optimization strategy.

**Research question:** What code change patterns do AI agents use when improving software performance, and what does this tell SBSE?

**Dataset:** 24,014 merged PRs across 5 agents:

| Agent | Merged PRs |
|-------|-----------|
| OpenAI_Codex | 18,004 |
| Devin | 2,595 |
| Copilot | 2,139 |
| Cursor | 1,005 |
| Claude_Code | 271 |

The end product is a `patch_classification` table containing confirmed performance improvement patches, each classified into one or more of 10 optimization strategies, suitable for cross-agent comparison and SBSE research.

---

## Project Structure

```
SSBSE/
├── pipeline.py                        # Entry point; orchestrates all steps
├── config.py                          # Constants and tunable parameters
├── requirements.txt                   # Python dependencies
├── aidev.db                           # SQLite database (created at runtime)
│
├── db/
│   ├── connection.py                  # Context-manager wrapper for sqlite3
│   └── schema.py                      # CREATE TABLE statements for all 16 tables
│
├── loaders/
│   └── hf_loader.py                   # Downloads HuggingFace dataset configs into SQLite
│
├── analysis/
│   ├── pr_classifier.py               # Step 1 — score PRs for performance relevance (1-5)
│   ├── pr_validator.py                # Optional — binary validation pass on candidates
│   ├── commit_collector.py            # Step 2 — join + file-filter commit patches
│   └── patch_classifier.py            # Step 3 — patch-level performance classification
│
├── scripts/
│   ├── calibrate_validator.py         # Test PR validation prompt against ground truth
│   ├── calibrate_patch_classifier.py  # Test patch classifier against ground truth
│   ├── sample_prs.py                  # Sample PRs for manual validation
│   ├── evaluate_prompt.py             # Evaluate PR classification prompt quality
│   └── collect_results.py            # Aggregate manual validation agent results
│
├── validation/                        # Hand-labeled ground truth (60 PRs)
│   ├── sampled_prs.csv
│   └── validation_results.csv
│
└── docs/
    ├── explanation.md                 # This file
    └── specs/                         # Original design specification
```

---

## Pipeline Steps

### Data Loading (`loaders/hf_loader.py`)

Downloads four dataset configurations from `hao-li/AIDev` and inserts them into SQLite:

| HuggingFace config | SQLite table |
|--------------------|--------------|
| `issue` | `issue` |
| `pull_request` | `pull_request` |
| `related_issue` | `related_issue` |
| `pr_commit_details` | `pr_commit_details` |

Each table is populated only if empty (idempotent). Rows are streamed in batches of `BATCH_SIZE` (5,000) with `INSERT OR IGNORE`. Foreign-key enforcement is disabled during bulk load and re-enabled afterwards.

---

### Step 1 — Classify PRs for Performance Relevance (`analysis/pr_classifier.py`)

**Goal:** Score all merged PRs for performance relevance using an LLM.

`classify_performance_prs()` selects all merged PRs not yet classified and sends them to the Anthropic API asynchronously with a concurrency cap of `CLASSIFY_CONCURRENCY` (30).

The prompt scores each PR **1–5** and assigns performance categories:

| Score | Meaning |
|-------|---------|
| 5 | Entirely focused on runtime performance improvement |
| 4 | Primary goal is performance, may include related cleanup |
| 3 | Performance is a secondary benefit |
| 2 | Incidental performance effect |
| 1 | No relation to performance |

**9 performance categories:**

| Category | Covers |
|----------|--------|
| `algorithmic_complexity` | Recursion elimination, DP injection, complexity reduction (O(n²) → O(n log n)) |
| `memory_data_locality` | Cache-friendly access patterns, data structure layout, prefetching, avoiding false sharing |
| `parallelism` | Thread-level parallelism, SIMD/vectorization, GPU offload, task parallelism |
| `loop_transformations` | Loop unrolling, tiling/blocking, fusion, invariant hoisting |
| `build_compilation` | Compiler hint flags, link-time optimization, inlining directives, PGO |
| `network_db_access` | Query optimization, index usage, batching requests, connection pooling |
| `io` | Buffered reading/writing, async I/O, reducing syscall frequency |
| `concurrency` | Lock contention reduction, lock-free structures, synchronization elimination |
| `other` | Miscellaneous optimizations not fitting the above categories |

Results are stored in `pr_performance_classification`. PRs with `relevance_score >= 4` (629 of 24,014) feed into Step 2.

**Batch API support:** `--batch-submit` submits all unclassified PRs to the Anthropic Batch API (cheaper, asynchronous). `--batch-collect BATCH_ID` downloads results once processing completes.

**Local model support:** `--local` uses a local llama-server (OpenAI-compatible API) instead of Anthropic.

---

### Step 2 — Collect Commit Details (`analysis/commit_collector.py`)

**Goal:** Gather business-logic code patches for performance-candidate PRs.

`collect_commit_details()` joins performance-candidate PRs to `pr_commit_details` and filters to relevant source files.

**Source selection priority:**
1. If `pr_validation` table has rows, use only PRs where `is_performance_related = 1` (higher precision).
2. Otherwise fall back to `pr_performance_classification` where `relevance_score >= 4`.

**File filtering — two-pass:**
1. **Exclusion patterns** (12 regex): strips test files, docs, CI configs, lock files, and generated files.
2. **Extension whitelist** (21 extensions): keeps only business-logic source files (`.py`, `.go`, `.rs`, `.ts`, `.java`, `.cpp`, `.cs`, etc.).

Results are written to `performance_commit_details`. Returns stats: `total_prs`, `total_commits`, `total_files`, `filtered_files`.

**Current dataset:** 5,448 rows (5,237 with non-null patch text) from 579 PRs.

---

### Step 3 — Classify Patches (`analysis/patch_classifier.py`)

**Goal:** Classify each code diff as a performance improvement and assign optimization strategies.

`classify_patches()` processes all unclassified rows in `performance_commit_details`. Unlike Step 1, the input is the **actual diff** — not the PR title or body. The patch is the evidence.

For each patch, the LLM answers:
- `is_performance` (true/false): does this diff make existing code faster or use less memory?
- `categories`: which of the 9 performance categories apply?
- `confidence`: 0.0–1.0
- `reasoning`: one sentence

**Explicit exclusions in the prompt:** new features, bug fixes that restore behavior, compiler/transpiler codegen improvements, refactoring, test/config/build changes.

Results are stored in `patch_classification`. Only patches with `is_performance = 1` are used for the research analysis.

---

### Optional — Binary Validation Pass (`analysis/pr_validator.py`)

An optional extra filtering step for the 629 score≥4 PR candidates. Uses an enriched prompt (title + body + commit messages + repo URL) with explicit binary TRUE/FALSE output.

Designed to reduce false positives before commit collection. Results stored in `pr_validation`; `commit_collector.py` uses this table automatically when populated.

**Calibration:** `scripts/calibrate_validator.py` tests the validation prompt against 60 hand-labeled ground-truth PRs and reports precision/recall/F1 before committing to the full run.

**Known limitation:** PR-level classification achieved ~43% precision on score=4 candidates (main false-positive source: `mochilang/mochi`, a compiler repo where "improve" refers to generated code quality, not runtime performance). Patch-level classification (Step 3) compensates by evaluating the diff directly.

---

## Database Schema

The database has **16 tables** in two groups.

### Source tables (loaded from HuggingFace)

| Table | Description |
|-------|-------------|
| `user` | GitHub user profiles |
| `repository` | Repository metadata |
| `issue` | GitHub issues |
| `pull_request` | Pull requests (includes `agent` field: Codex/Devin/Copilot/Cursor/Claude_Code) |
| `related_issue` | Many-to-many: issues a PR references |
| `pr_timeline` | Timeline events on PRs |
| `pr_commits` | Commit SHAs per PR |
| `pr_commit_details` | Per-file diff details (patch, additions, deletions) |
| `pr_reviews` | PR review records |
| `pr_review_comments` | Inline review comments |
| `pr_comments` | General PR comments |
| `pr_task_type` | Task-type labels on PRs |

### Derived tables (pipeline-generated)

| Table | Description |
|-------|-------------|
| `pr_performance_classification` | LLM scores (1–5) for all 24,014 merged PRs |
| `pr_validation` | Binary validation results for score≥4 candidates |
| `performance_commit_details` | Filtered business-logic patches from candidate PRs |
| `patch_classification` | Final output: per-patch performance classification with categories |

---

## Configuration (`config.py`)

| Constant | Default | Description |
|----------|---------|-------------|
| `DB_PATH` | `"aidev.db"` | SQLite file path |
| `HF_DATASET` | `"hao-li/AIDev"` | HuggingFace dataset identifier |
| `BATCH_SIZE` | `5000` | Insert batch size for HF loading |
| `RELEVANCE_THRESHOLD` | `4` | Minimum PR score (1–5) to collect patches |
| `ANTHROPIC_MODEL` | `claude-haiku-4-5-20251001` | Model for classification |
| `ANTHROPIC_BATCH_MODEL` | `claude-3-5-haiku-20241022` | Model for Batch API |
| `BODY_TRUNCATE_LENGTH` | `2000` | Max PR body chars sent to model |
| `PATCH_TRUNCATE_LENGTH` | `4000` | Max patch chars sent to model |
| `COMMIT_MSG_TRUNCATE_LENGTH` | `1000` | Max commit message chars for validation |
| `CLASSIFY_CONCURRENCY` | `30` | Async concurrency cap for API calls |
| `EXCLUDED_FILE_PATTERNS` | (12 patterns) | Regex patterns for files to exclude |
| `BUSINESS_LOGIC_EXTENSIONS` | (21 extensions) | Source file extension whitelist |
| `LOCAL_MODEL_BASE_URL` | `http://localhost:8080/v1` | Local llama-server endpoint |
| `LOCAL_MODEL_NAME` | `Llama-3.1-8B-Instruct-Q6_K` | Local model name |

---

## Usage

### Install dependencies

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY="sk-ant-..."
```

### Run the full pipeline

```bash
python pipeline.py
```

### CLI flags

```
usage: pipeline.py [-h] [--db-path DB_PATH] [--skip-load] [--skip-classify]
                   [--skip-collect] [--anthropic-model MODEL] [--local]
                   [--concurrency N] [--batch-submit] [--batch-collect BATCH_ID]
                   [--validate] [--classify-patches]

options:
  --db-path DB_PATH          SQLite file path (default: aidev.db)
  --skip-load                Skip HuggingFace data loading
  --skip-classify            Skip Step 1 PR classification
  --skip-collect             Skip Step 2 commit detail collection
  --anthropic-model MODEL    Anthropic model override
  --local                    Use local llama-server instead of Anthropic API
  --concurrency N            Max concurrent API requests
  --batch-submit             Submit unclassified PRs to Anthropic Batch API
  --batch-collect BATCH_ID   Download completed Batch API results
  --validate                 Run binary validation pass on score>=4 PRs
  --classify-patches         Run Step 3: patch-level performance classification
```

### Example: data already loaded, run Steps 1–2 only

```bash
python pipeline.py --skip-load
```

### Example: calibrate and run patch classification

```bash
# Calibrate first (~40 API calls, ~$0.50)
python scripts/calibrate_patch_classifier.py

# Run patch classification on all 5,237 patches (~$8–10)
python pipeline.py --skip-load --skip-classify --skip-collect --classify-patches
```

### Example: submit to Batch API (cheaper, async)

```bash
# Submit — prints a BATCH_ID
python pipeline.py --skip-load --skip-collect --batch-submit

# Later, once batch is complete:
python pipeline.py --batch-collect msgbatch_abc123
```

### Example: use local model

```bash
# Start llama-server first, then:
python pipeline.py --skip-load --local
```

### Output

After a full run:

```
=== Step 1 Summary ===
Classified:     24014
Relevant:       629
Skipped errors: 0

=== Step 2 Summary ===
PRs:            579
Commits:        1264
Files changed:  5448
Filtered out:   ...

=== Stage 4 Summary ===
Total classified:  5237
Performance:       ~800–1500
Non-performance:   ~3700–4400
Errors:            0
```

The final research dataset is the `patch_classification` table filtered to `is_performance = 1`, joined with `pull_request.agent` for cross-agent analysis.
