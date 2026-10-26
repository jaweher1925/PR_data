# ENRICH – data mined from public GitHub

The papers' released files miss some columns. We filled them by downloading the public repositories
(git clone --bare --filter=blob:none, no API, no token) and reading each commit with `git show`.

| What | How | Result |
|---|---|---|
| Sawada + Watanabe fix commits (318) | `git show <sha>` in 122 repos (PR refs fetched for Watanabe) | message, date, author, files, **code patch** for 316 |
| Who Finishes PRs (464) | `refs/pull/<n>/head` in 83 repos | date of the last commit of the AI PR and of the fix PR |
| Xia & Miller AI commits (7,822) | `Co-authored-by:` line of each AI commit in 173 repos | **agent name for 7,810** (Claude Code 4,980, Copilot 2,256, Cursor 229, Devin 175, Jules 77, Gemini 49, Codex 44) |
| Xia & Miller first fix commits (3,522 unique) | `git show <sha>` | message, date, author, **code patch** for 3,515 |

Files:
- `FINAL_AI_FIXES_ENRICHED.csv` – 9,619 fixes (with Khemissi), 20 columns (used by `99_ALL_PAPERS_summary.ipynb`)
- `patches_<paper>.parquet` – the code of every fix (fix_id, repo, fix_change, patch) – open with `pd.read_parquet`
- `miner.py` – the git miner (`python miner.py requests.csv out.jsonl [--no-patch]`)
- `merge_enrich.py` – merges the mined data into the final dataset (paths inside point to my work folder – change W/G at the top)

Limits: Xia & Miller give the fix commit only when the FIRST later change was a fix (52% of rows);
Yu 2024 keeps no repo owner, so it cannot be mined.
