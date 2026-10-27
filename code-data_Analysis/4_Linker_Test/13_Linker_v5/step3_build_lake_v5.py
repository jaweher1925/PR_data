"""
step3_build_lake_v5.py - build the DuckDB lake linker_v5.py reads (data/duckdb/aidev_v5.duckdb).

Tables: pull_request, repository, pr_task_type_v5 and pr_files (pr_id, filename, is_boilerplate).
Scope: Python repositories by default (Mohamed: focus on Python); linker_v5 itself keeps the
paper's 500+ star filter.  Use  --language all  for every language.

Run:  uv run --project %EA% python step3_build_lake_v5.py [--language all]
"""
import argparse, os
import duckdb
import pandas as pd
from prfix.rq1_fix_rates.boilerplate import is_boilerplate        # Mohamed's repo
from paths import DB, RAW_V5

ap = argparse.ArgumentParser()
ap.add_argument("--language", default="Python", help="repository language, or 'all' (default: Python)")
args = ap.parse_args()
f = lambda t: os.path.join(RAW_V5, f"{t}.parquet").replace("\\", "/")
for t in ("pull_request", "repository", "pr_commit_details", "pr_task_type_v5"):
    if not os.path.exists(f(t)):
        raise SystemExit(f"Missing {f(t)} - run step1 and step2 first.")

os.makedirs(os.path.dirname(DB), exist_ok=True)
if os.path.exists(DB):
    os.remove(DB)
con = duckdb.connect(DB)
lang = "" if args.language.lower() == "all" else f"WHERE language = '{args.language}'"
con.execute(f"CREATE TABLE repository AS SELECT * FROM read_parquet('{f('repository')}') {lang}")
con.execute(f"CREATE VIEW pull_request AS SELECT * FROM read_parquet('{f('pull_request')}')")
con.execute(f"CREATE VIEW pr_task_type_v5 AS SELECT * FROM read_parquet('{f('pr_task_type_v5')}')")

# changed files, only for PRs of in-scope repositories with 500+ stars (keeps the lake small)
files = con.execute(f"""
    SELECT DISTINCT d.pr_id, d.filename
    FROM read_parquet('{f('pr_commit_details')}') d
    JOIN pull_request p ON p.id = d.pr_id
    JOIN repository r ON r.id = p.repo_id
    WHERE r.stars >= 500 AND d.filename IS NOT NULL""").df()
names = pd.Series(files.filename.unique())
bp = dict(zip(names, names.map(is_boilerplate)))
files["is_boilerplate"] = files.filename.map(bp).astype(bool)
con.register("files_df", files)
con.execute("CREATE TABLE pr_files AS SELECT * FROM files_df")

print(f"Lake: {DB}  (language: {args.language})")
print(f"pr_files: {len(files):,} rows, {files.pr_id.nunique():,} PRs, "
      f"{files.is_boilerplate.mean():.1%} boilerplate")
print("\nWhat linker_v5 will use (merged agent PRs, 500+ star repos):")
print(con.execute("""
    SELECT p.agent, COUNT(*) AS merged_ai_prs, COUNT(DISTINCT p.repo_id) AS repos,
           MIN(p.merged_at) AS first, MAX(p.merged_at) AS latest
    FROM pull_request p JOIN repository r ON r.id = p.repo_id
    WHERE p.merged_at IS NOT NULL AND r.stars >= 500
    GROUP BY 1 ORDER BY 2 DESC""").df().to_string(index=False))
con.close()
print("\nNext: step4_run_linker_v5.py")
