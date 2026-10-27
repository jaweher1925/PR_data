"""
step4_run_linker.py - run linker.py (Mohamed's main, unchanged) and save the results.

1. builds the changed-files table (prfix.common.loaders.build_net_files)
2. linker.compute()            -> one row per merged AI PR: fix within 14 / 30 days?
3. linker.per_agent_summary()  -> % of AI PRs with a candidate fix, per agent
4. linker.pairs_sql()          -> every candidate pair  AI PR -> fix PR  (with links)

Run:  uv run --project "C:\\Users\\GIGABYTE\\Desktop\\code_theis\\evolution-assurance" python step4_run_linker.py
Output: results/ (per_pr.csv, per_agent.csv, pairs.csv, linker_results.xlsx)
"""
import os, subprocess, sys
import pandas as pd
from paths import REPO, RESULTS

subprocess.check_call([sys.executable, "-m", "prfix.common.loaders.build_net_files"], cwd=REPO)

from prfix.rq1_fix_rates import linker            # Mohamed's linker.py
from prfix.common.loaders.base import connect

os.makedirs(RESULTS, exist_ok=True)
print("Running linker.compute() ...")
per_pr = linker.compute()                         # defaults: net_sub linker, agent P side
summary = linker.per_agent_summary(per_pr)

con = connect(read_only=True)
pairs = con.execute(linker.pairs_sql()).df()
urls = con.execute("""SELECT id, html_url, 'agent' AS cohort, agent, title FROM pull_request
                      UNION ALL
                      SELECT id, html_url, 'human', 'Human', title FROM human_pull_request""").df()
con.close()
u = urls.drop_duplicates("id").set_index("id")
pairs["ai_pr_url"] = pairs["p_id"].map(u["html_url"])
pairs["ai_pr_title"] = pairs["p_id"].map(u["title"])
pairs["fix_pr_url"] = pairs["q_id"].map(u["html_url"])
pairs["fixed_by"] = pairs["q_id"].map(u["cohort"]).map({"agent": "AI agent", "human": "Human"})
pairs["fix_by_agent"] = pairs["q_id"].map(u["agent"])
pairs = pairs.rename(columns={"p_id": "ai_pr_id", "agent": "ai_agent", "q_id": "fix_pr_id",
                              "q_title": "fix_pr_title", "q_type": "fix_type", "day_gap": "days_between"})
pairs = pairs[["ai_agent", "ai_pr_id", "ai_pr_url", "ai_pr_title", "fix_pr_id", "fix_pr_url", "fix_pr_title",
               "fix_type", "fixed_by", "fix_by_agent", "days_between"]]

per_pr.to_csv(os.path.join(RESULTS, "per_pr.csv"), index=False)
summary.to_csv(os.path.join(RESULTS, "per_agent.csv"), index=False)
pairs.to_csv(os.path.join(RESULTS, "pairs.csv"), index=False)
try:
    with pd.ExcelWriter(os.path.join(RESULTS, "linker_results.xlsx")) as xw:
        summary.to_excel(xw, sheet_name="Per agent", index=False)
        pairs.to_excel(xw, sheet_name="Pairs AI PR - fix PR", index=False)
        per_pr.to_excel(xw, sheet_name="Per AI PR", index=False)
except ImportError:
    print("(openpyxl not installed in this environment: Excel skipped, CSV files saved)")

print(f"\nMerged AI PRs checked : {len(per_pr):,}")
print(f"With a candidate fix  : {per_pr.has_followup_30d.sum():,} within 30 days, "
      f"{per_pr.has_followup_14d.sum():,} within 14 days")
print(f"Candidate pairs       : {len(pairs):,}   (paper: 2,510)")
print(f"Fixed by              : {pairs.fixed_by.value_counts().to_dict()}")
print("\n" + summary.round(1).to_string(index=False))
print(f"\nSaved in {RESULTS}. Next: open step5_results.ipynb")
