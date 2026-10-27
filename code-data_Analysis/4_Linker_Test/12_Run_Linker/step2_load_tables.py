"""
step2_load_tables.py - register the 5 AIDev tables from step 1 in linker.py's DuckDB database.
No change to Mohamed's code: we only point the database views to the right files.

Run:  uv run --project "C:\\Users\\GIGABYTE\\Desktop\\code_theis\\evolution-assurance" python step2_load_tables.py
"""
import os
import duckdb
from paths import RAW_LINKER, DUCKDB, TABLES

os.makedirs(os.path.dirname(DUCKDB), exist_ok=True)
con = duckdb.connect(DUCKDB)
for t in TABLES:
    f = os.path.join(RAW_LINKER, f"{t}.parquet")
    if not os.path.exists(f):
        raise SystemExit(f"Missing {f} - run step1_download_data.py first.")
    con.execute(f"CREATE OR REPLACE VIEW {t} AS SELECT * FROM read_parquet('{f.replace(chr(92), '/')}')")
    n = con.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
    print(f"  {t:20s} {n:>9,} rows")

print("\nWhat linker.py will use (repos with 500+ stars, merged PRs):")
print(con.execute("""
    SELECT p.agent, COUNT(*) AS merged_ai_prs, MAX(p.merged_at) AS latest
    FROM pull_request p JOIN repository r ON r.id = p.repo_id
    WHERE p.merged_at IS NOT NULL AND r.stars >= 500
    GROUP BY 1 ORDER BY 2 DESC""").df().to_string(index=False))
n_h = con.execute("""SELECT COUNT(*) FROM human_pull_request h JOIN repository r ON r.url = h.repo_url
                     WHERE h.merged_at IS NOT NULL AND r.stars >= 500""").fetchone()[0]
print(f"merged human PRs in the same repos: {n_h:,}")
con.close()
print("\nDone. Next: step3_fetch_changed_files.py")
