# 12_Run_Linker – run linker.py step by step

**Goal:** take the AI-generated PRs (AIDev-pop) and let linker.py find, for each merged AI PR,
a later PR (AI or human) that fixed it. Code: Mohamed's repo `evolution-assurance`, branch **main**, unchanged.

```
AI-generated PRs  ->  linker.py  ->  candidate fix PRs  ->  manual analysis (Q1, Q2, Q3)
```

## Data we use: AIDev-pop (5 tables, one AIDev version)
| Table | What it gives |
|---|---|
| pull_request | the AI PRs (agent, repo, dates, title) |
| repository | repos (stars: linker keeps 500+) |
| pr_task_type | label of each AI PR (fix, feature, ...) |
| human_pull_request | human PRs in the same repos (many fixes are by humans) |
| human_pr_task_type | label of each human PR |
| + changed files (GitHub) | files changed by each candidate PR (fix must touch the same files) |

AIDev v4 removed pr_task_type and the human tables, so step 1 downloads them from the last version that has them.

## Before you start (once)
- `uv` installed, and `uv sync --extra dev` done in `code_theis\evolution-assurance` (already done).
- GitHub CLI: `winget install --id GitHub.cli`, then close and reopen the terminal.
- `GITHUB_TOKEN` in `evolution-assurance\.env` (already there).

Open a terminal **in this folder** and set a short name for the repo:
```
cd C:\Users\GIGABYTE\Desktop\Thesis_Data_Analysis\code-data_Analysis\4_Linker_Test\12_Run_Linker
set EA=C:\Users\GIGABYTE\Desktop\code_theis\evolution-assurance
```

## Steps
| Step | Command | What it does | Time |
|---|---|---|---|
| 1 | `uv run --project %EA% python step1_download_data.py` | downloads the 5 AIDev tables to `evolution-assurance\data\raw_linker` | ~10 min |
| 2 | `uv run --project %EA% python step2_load_tables.py` | loads them in linker.py's database, prints merged AI PRs per agent | 1 min |
| 3 | `uv run --project %EA% python step3_fetch_changed_files.py` | gets the changed files of ~6,000 PRs from GitHub (can be stopped and restarted) | 1–2 h |
| 4 | `uv run --project %EA% python step4_run_linker.py` | runs linker.py, saves `results\` (per_pr.csv, per_agent.csv, pairs.csv) | few min |
| 5 | open `step5_results.ipynb` → Run All | tables + charts + comparison with the paper + sample of 50 pairs to annotate → `tables\linker_results.xlsx` | 1 min |

## What we get
- % of merged AI PRs with a candidate fix within 14 / 30 days, per agent
- every pair AI PR -> fix PR with links, who fixed it (AI agent or human), days between
- check: same pairs as the paper (2,510 candidates, 334 direct fixes)? our known fixes (323 clean pairs, 948 fixed later)?
- 50 pairs ready for the manual analysis (columns: why fixed, what was wrong, how fixed)
