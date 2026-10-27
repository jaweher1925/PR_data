# linker_v5.py – changes vs Mohamed's branch linker.py

Source: `malmukhtar/evolution-assurance`, branch `feature/broader-maintenance-linker`, `src/prfix/rq1_fix_rates/linker.py`.

1. Docstring: explain what changed for v5
2. Connection: open the v5 lake (aidev_v5.duckdb); drop the RQ4 import (it needs human tables)
3. P_POP: remove the human cohort (no human PRs in v5)
4. Q pool: task type from pr_task_type_v5 (rebuilt for v5 by step2: Conventional-Commits title prefix, else a classifier trained on the v3 labels) instead of AIDev's pr_task_type
5. Q pool: remove the human follow-up PRs (UNION with human_pull_request)
6. SQL_COHORT: remove the human cohort
7. SQL_SHARED_CUTOFF: agent table only (same as 'own')
8. FILES_SOURCE: commit_sub / commit_any from v5 pr_commit_details (no net-diff, no human files)
9. file_source_coverage: count PRs without any file row in pr_files
10. Remove net_sub_excluding (net-diff only)
11. Default file-match mode: commit_sub (was net_sub)
12. SQL_PR_SOURCE: agent table only
13. Remove low_touch_followup_correlation (needs RQ4 human-commit signals)
14. Remove _render_contingency and report() (paper RQ1 report on v3)
15. CLI: default output folder
16. CLI: program name
17. CLI: default file-match mode commit_sub
18. CLI: only the 'own' cut-off (one cohort)
19. CLI: coverage message for v5
20. run_metadata: record the dataset and lake

Unchanged: P population (merged agent PRs in 500+ star repos), candidate modes (`all_changes`, `fix_only`), windows and window rules, censoring, `followup_status`, `candidate_pairs` export, `per_agent_summary`, CLI options.
