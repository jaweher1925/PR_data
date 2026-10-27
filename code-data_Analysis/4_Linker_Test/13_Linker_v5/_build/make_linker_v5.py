"""Build linker_v5.py from Mohamed's branch linker.py with small, explicit edits.

Every edit is a (old, new, why) triple; the script fails if an `old` text is not
found exactly once, so nothing is changed silently. The list of edits is also
written to CHANGES_linker_v5.md.
"""
import re
import sys
from pathlib import Path

SRC = Path(sys.argv[1])
OUT = Path(sys.argv[2])
CHANGES = Path(sys.argv[3])
s = SRC.read_text(encoding="utf-8")
log = []


def edit(old, new, why, count=1):
    global s
    n = s.count(old)
    if n != count:
        raise SystemExit(f"edit failed ({n} matches, expected {count}): {why}\n---\n{old[:300]}")
    s = s.replace(old, new)
    log.append(why)


def cut(start, end, why):
    """Remove the text from `start` (included) up to `end` (excluded)."""
    global s
    i = s.index(start)
    j = s.index(end, i)
    s = s[:i] + s[j:]
    log.append(why)


# 0. header
edit('"""RQ1 behavioural follow-up: did a merged agent PR get a subsequent\n',
     '"""linker_v5 -- Mohamed\'s maintenance linker (branch feature/broader-maintenance-linker)\n'
     'adapted to AIDev v5 (hao-li/AIDev-7.6M), AGENT follow-ups only.\n\n'
     'AIDev v5 has no pr_task_type, human_pull_request or human_pr_task_type, so:\n'
     '  * the follow-up pool (Q) is agent PRs only (pull_request);\n'
     '  * the task type is read from the PR title (Conventional Commits prefix, e.g. "fix:");\n'
     '  * file overlap uses v5 pr_commit_details (table pr_files, boilerplate flagged);\n'
     '  * the lake is data/duckdb/aidev_v5.duckdb (built by step2_build_lake_v5.py).\n'
     'Everything else (P population, all_changes / fix_only, windows, censoring, pair export)\n'
     'is the branch code unchanged. See CHANGES_linker_v5.md.\n\n'
     'Original docstring:\n\n'
     'RQ1 behavioural follow-up: did a merged agent PR get a subsequent\n',
     "Docstring: explain what changed for v5")

# 1. imports / connection
edit("from prfix.common.loaders.base import connect\n"
     "from prfix.config import DIGESTS_DIR, PROCESSED_DIR, PROJECT_ROOT, ensure_dirs\n"
     "from prfix.rq4_signals.per_pr_signals import compute as per_pr_compute\n",
     "import os\n\n"
     "import duckdb\n\n"
     "from prfix.config import DATA_DIR, DIGESTS_DIR, PROCESSED_DIR, PROJECT_ROOT, ensure_dirs\n\n"
     "# v5 lake (built by step2_build_lake_v5.py); override with the AIDEV_V5_DB variable.\n"
     "V5_DB = Path(os.environ.get(\"AIDEV_V5_DB\", DATA_DIR / \"duckdb\" / \"aidev_v5.duckdb\"))\n\n\n"
     "def connect(read_only: bool = False) -> duckdb.DuckDBPyConnection:\n"
     "    \"\"\"Open the AIDev v5 lake (instead of the v3 lake prfix.duckdb).\"\"\"\n"
     "    if not V5_DB.exists():\n"
     "        raise FileNotFoundError(f\"{V5_DB} not found - run step2_build_lake_v5.py first\")\n"
     "    return duckdb.connect(str(V5_DB), read_only=read_only)\n",
     "Connection: open the v5 lake (aidev_v5.duckdb); drop the RQ4 import (it needs human tables)")

# 2. P side: agent only
cut('    "human": """\n    -- All merged contemporaneous human-cohort PRs (same >=500-star repos;',
    '}\n\nSQL_FOLLOWUP_PAIRS',
    "P_POP: remove the human cohort (no human PRs in v5)")

# 3. Q pool: task type from the title, no human union
TITLE_TYPE = (r"CASE WHEN lower(q.title) LIKE 'revert%' THEN 'revert' ELSE COALESCE(NULLIF(regexp_extract(lower(q.title), "
              r"'^\\s*\\[?(feat|fix|docs|style|refactor|perf|test|build|ci|chore|revert)\\]?"
              r"(\\([^)]*\\))?!?\\s*:', 1), ''), '(none)') END AS q_type")
edit("        COALESCE(t.type, '(none)') AS q_type,\n"
     "        (q.title LIKE 'Revert %' OR q.title LIKE 'revert %') AS q_is_revert\n"
     "    FROM pull_request q\n"
     "    JOIN repository   r ON r.id = q.repo_id\n"
     "    LEFT JOIN pr_task_type t ON t.id = q.id\n",
     "        COALESCE(t5.type, '(none)') AS q_type,  -- v5: type rebuilt by step2 (title prefix or classifier)\n"
     "        (q.title LIKE 'Revert %' OR q.title LIKE 'revert %') AS q_is_revert\n"
     "    FROM pull_request q\n"
     "    JOIN repository   r ON r.id = q.repo_id\n"
     "    LEFT JOIN pr_task_type_v5 t5 ON t5.id = q.id\n",
     "Q pool: task type from pr_task_type_v5 (rebuilt for v5 by step2: Conventional-Commits title "
     "prefix, else a classifier trained on the v3 labels) instead of AIDev's pr_task_type")
cut("\n    UNION ALL\n\n    -- Contemporaneous *human* follow-up PRs in the same AIDev-pop repos.",
    "\n),\nq_fix AS (",
    "Q pool: remove the human follow-up PRs (UNION with human_pull_request)")

# 4. cohort + shared cut-off: agent only
cut('    "human": """\nSELECT\n    p.id        AS pr_id,\n    \'Human\'     AS agent,',
    '}\n\n\n# Shared observation cut-off date',
    "SQL_COHORT: remove the human cohort")
i = s.index('SQL_SHARED_CUTOFF = """')
j = s.index('"""', i + len('SQL_SHARED_CUTOFF = """')) + 3
s = s[:i] + ('SQL_SHARED_CUTOFF = """\n'
             '-- v5: agent PRs only, so the shared cut-off is the agent table\'s last merge.\n'
             'SELECT MAX(TRY_CAST(q.merged_at AS TIMESTAMP)) AS shared_cutoff\n'
             'FROM pull_request q JOIN repository r ON r.id = q.repo_id\n'
             'WHERE q.merged_at IS NOT NULL AND r.stars >= 500\n"""') + s[j:]
log.append("SQL_SHARED_CUTOFF: agent table only (same as 'own')")

# 5. file sources: v5 pr_commit_details (pr_files)
i = s.index("FILES_SOURCE = {")
j = s.index("}\n", i) + 2
s = s[:i] + ('FILES_SOURCE = {\n'
             '    # v5: per-PR files from pr_commit_details (table pr_files, step2), boilerplate flagged\n'
             '    # with prfix.rq1_fix_rates.boilerplate.is_boilerplate. v5 has no net diffs.\n'
             '    "commit_sub": "SELECT pr_id, filename FROM pr_files WHERE NOT is_boilerplate",\n'
             '    "commit_any": "SELECT pr_id, filename FROM pr_files",\n'
             '}\n') + s[j:]
log.append("FILES_SOURCE: commit_sub / commit_any from v5 pr_commit_details (no net-diff, no human files)")
edit('    source = FILES_SOURCE["net_any"] if linker.startswith("net") else FILES_SOURCE[linker]\n',
     '    source = FILES_SOURCE["commit_any"] if linker.startswith("commit") else FILES_SOURCE[linker]\n',
     "file_source_coverage: count PRs without any file row in pr_files")
cut("def net_sub_excluding(filenames) -> str:", "def pairs_prelude_sql(",
    "Remove net_sub_excluding (net-diff only)")
s = s.replace('linker: str = "net_sub"', 'linker: str = "commit_sub"')
log.append('Default file-match mode: commit_sub (was net_sub)')

# 6. pair export: authors are agents
edit("""    SELECT id, 'agent' AS src, agent FROM pull_request
    UNION ALL
    SELECT id, 'human' AS src, CAST(NULL AS VARCHAR) AS agent FROM human_pull_request
""", """    SELECT id, 'agent' AS src, agent FROM pull_request
""", "SQL_PR_SOURCE: agent table only")

# 7. remove the RQ4 contingency + report() (need human commit signals)
cut("# ---------------------------------------------------------------------------\n"
    "# Low-touch merge vs follow-up correlation",
    "# ---------------------------------------------------------------------------\n# Reporting",
    "Remove low_touch_followup_correlation (needs RQ4 human-commit signals)")
cut("def _render_contingency(", "# ---------------------------------------------------------------------------\n"
    "# Command line", "Remove _render_contingency and report() (paper RQ1 report on v3)")

# 8. CLI
edit('DEFAULT_OUTPUT_DIR = PROCESSED_DIR / "maintenance_linker"',
     'DEFAULT_OUTPUT_DIR = PROCESSED_DIR / "maintenance_linker_v5"', "CLI: default output folder")
edit('        prog="python -m prfix.rq1_fix_rates.linker",', '        prog="python linker_v5.py",',
     "CLI: program name")
edit('    ap.add_argument("--file-match-mode", choices=sorted(FILES_SOURCE), default="net_sub",',
     '    ap.add_argument("--file-match-mode", choices=sorted(FILES_SOURCE), default="commit_sub",',
     "CLI: default file-match mode commit_sub")
edit('    ap.add_argument("--cutoff", choices=["own", "shared"], default="own",',
     '    ap.add_argument("--cutoff", choices=["own"], default="own",',
     "CLI: only the 'own' cut-off (one cohort)")
edit('            "their pairs cannot be file-matched. For net-diff modes, run "\n'
     '            f"`python -m prfix.common.loaders.fetch_netdiff --p-side {args.p_side} "\n'
     '            f"--candidate-mode {args.candidate_mode} --max-window {windows[-1]}` and "\n'
     '            "`python -m prfix.common.loaders.build_net_files` first.[/yellow]")',
     '            "their pairs cannot be file-matched (no rows in v5 pr_commit_details).[/yellow]")',
     "CLI: coverage message for v5")
edit('        "code_revision": _git_revision(),',
     '        "code_revision": _git_revision(),\n'
     '        "dataset": "AIDev v5 (hao-li/AIDev-7.6M), agent follow-ups only",\n'
     '        "lake": str(V5_DB),', "run_metadata: record the dataset and lake")

# 9. sanity: no reference to the removed tables is left
for t in ("pr_task_type", "human_pull_request", "human_pr_task_type", "pr_net_files",
          "human_pr_commit_details", "per_pr_compute"):
    code_lines = [ln for ln in s.splitlines() if t in ln and not ln.strip().startswith(("#", "--", "*", "``"))
                  and '"""' not in ln]
    left = [ln for ln in code_lines if re.search(r"\b(FROM|JOIN)\s+" + t + r"\b", ln) or t + "(" in ln]
    if left:
        raise SystemExit(f"still uses {t}: {left[:3]}")

OUT.write_text(s, encoding="utf-8")
CHANGES.write_text("# linker_v5.py – changes vs Mohamed's branch linker.py\n\n"
                   "Source: `malmukhtar/evolution-assurance`, branch `feature/broader-maintenance-linker`, "
                   "`src/prfix/rq1_fix_rates/linker.py`.\n\n"
                   + "\n".join(f"{k}. {w}" for k, w in enumerate(log, 1)) + "\n\n"
                   "Unchanged: P population (merged agent PRs in 500+ star repos), candidate modes "
                   "(`all_changes`, `fix_only`), windows and window rules, censoring, `followup_status`, "
                   "`candidate_pairs` export, `per_agent_summary`, CLI options.\n", encoding="utf-8")
print(f"wrote {OUT} ({len(log)} edits)")
