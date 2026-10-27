# 13_Linker_v5 – Mohamed's linker on AIDev v5 (agent follow-ups)

**Goal:** for every merged AI PR in AIDev v5 (Python, 500+ stars), find the later **agent** PRs in the same
repository that change the same files within 30 / 60 days.

```
AIDev v5 agent PRs  ->  linker_v5.py (Mohamed's branch, adapted)  ->  candidate pairs  ->  manual analysis
```

## Why a v5 version of the linker
AIDev v5 (`hao-li/AIDev-7.6M`, 7.7M agent PRs up to March 2026) has **no** `pr_task_type`,
`human_pull_request`, `human_pr_task_type` (README of v5: *"pr_task_type is not shipped since v4,
still available for v3"*). `linker_v5.py` = Mohamed's branch `linker.py` with only these changes
(full list in `CHANGES_linker_v5.md`, made by `_build/make_linker_v5.py`):

| Change | Why |
|---|---|
| follow-ups (Q) = agent PRs only | v5 has no human PRs (we agreed: we need the agents) |
| task type from `pr_task_type_v5` (step 2) | v5 has no task type: title prefix (`fix:` …) or a classifier trained on v3 |
| changed files from v5 `pr_commit_details` (`pr_files`) | v5 has no net diffs; boilerplate is still excluded |
| reads `data/duckdb/aidev_v5.duckdb` | separate from the v3 lake |
| removed the RQ4 contingency / report() | they need human-commit data |

Everything else is Mohamed's code unchanged: P population, `all_changes` / `fix_only`, windows 30/60,
censoring, `candidate_pairs` export, `run_metadata.json`.

**Checked:** on Mohamed's own test lake (agent PRs only), `linker_v5.py` gives exactly the same pairs and
per-PR flags as his branch in all 8 settings (all_changes / fix_only × 30-60 / 14-30 × with / without boilerplate).
**Task type rebuilt for v5:** tested on v3 labels it doesn't see: 85.8% accuracy, `fix` precision 91.8 %, recall 88.6 %.

## Steps (CMD, in this folder)
```
set PATH=%APPDATA%\Python\Python314\Scripts;%PATH%
set EA=C:\Users\GIGABYTE\Desktop\code_theis\evolution-assurance
cd C:\Users\GIGABYTE\Desktop\Thesis_Data_Analysis\code-data_Analysis\4_Linker_Test\13_Linker_v5
```
| Step | Command | What it does | Time |
|---|---|---|---|
| 1 | `uv run --project %EA% python step1_download_v5.py` | downloads v5 pull_request, repository, pr_commit_details (pr_id + filename only) | 10–20 min |
| 2 | `uv run --project %EA% python step2_task_type_v5.py` | rebuilds the task type (tests it on v3 first) | ~2 min |
| 3 | `uv run --project %EA% python step3_build_lake_v5.py` | builds the v5 lake, Python repos (`--language all` for all) | ~5 min |
| 4 | `uv run --project %EA% python step4_run_linker_v5.py` | runs linker_v5: `all_changes` and `fix_only`, 30/60 days | minutes |
| 5 | open `step5_results_v5.ipynb` → Run All | tables, charts, 50 pairs to annotate → `tables/linker_v5_results.xlsx` | 1 min |

Step 2 needs the v3 labels from `12_Run_Linker` step 1 (`evolution-assurance/data/raw_linker`).

## Limits (say them when showing results)
- **Agent follow-ups only**: changes made by humans are not seen.
- **Candidates**: same files + later date ≠ proof the AI PR was wrong (code-level analysis needed).
- v5 `pr_commit_details` may miss files for commits after the 30th in very large PRs (v5 README).
