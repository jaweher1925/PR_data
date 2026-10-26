"""
add_dates.py - add the dates of the AI PR and of the fix PR to the 323 clean pairs.

Run from anywhere:   python code-data_Analysis/tools/add_dates.py
Reads (all inside Thesis_Data_Analysis/data):
  CLEAN_PR_DATASET/CLEAN_AI_PR_FIXES.csv                         the 323 clean pairs
  08_Alam_AI_Fix_PRs/Agents for Fixing/dataset/pull_request.parquet   AIDev-pop: created_at / merged_at of agent PRs
  ENRICH/FINAL_AI_FIXES_ENRICHED.csv                              last-commit dates mined from GitHub (for human fix PRs)
Writes: data/CLEAN_PR_DATASET/CLEAN_AI_PR_FIXES_with_dates.xlsx (+ .csv)
"""
import os
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.normpath(os.path.join(HERE, "..", "..", "data"))
CLEAN = os.path.join(DATA, "CLEAN_PR_DATASET")

pairs = pd.read_csv(os.path.join(CLEAN, "CLEAN_AI_PR_FIXES.csv"), dtype={"ai_pr_id": str, "fix_pr_id": str})
print("pairs:", len(pairs))

# 1. AIDev-pop dates (agent PRs) - from Alam's package
pr = pd.read_parquet(os.path.join(DATA, "08_Alam_AI_Fix_PRs", "Agents for Fixing", "dataset", "pull_request.parquet"),
                     columns=["id", "created_at", "merged_at"])
pr["id"] = pr["id"].astype(str)
d = pr.set_index("id")
for side in ["ai_pr", "fix_pr"]:
    pairs[f"{side}_created_at"] = pairs[f"{side}_id"].map(d["created_at"])
    pairs[f"{side}_merged_at"] = pairs[f"{side}_id"].map(d["merged_at"])
pairs["date_source"] = "AIDev-pop"

# 2. fix PRs written by humans are not in AIDev -> date of the last commit of the fix PR (mined from GitHub)
enr = pd.read_csv(os.path.join(DATA, "ENRICH", "FINAL_AI_FIXES_ENRICHED.csv"), low_memory=False)
wf = enr[enr["paper"] == "Who Finishes 2026"].drop_duplicates("fix_change").set_index("fix_change")["fix_date"]
miss = pairs["fix_pr_created_at"].isna()
pairs.loc[miss, "fix_pr_merged_at"] = pairs.loc[miss, "fix_pr_url"].map(wf)
pairs.loc[miss, "date_source"] = "AIDev-pop (AI PR) + GitHub last commit (fix PR)"

for c in ["ai_pr_created_at", "ai_pr_merged_at", "fix_pr_created_at", "fix_pr_merged_at"]:
    pairs[c] = pd.to_datetime(pairs[c], utc=True, errors="coerce")
pairs["days_ai_merge_to_fix_merge"] = ((pairs["fix_pr_merged_at"] - pairs["ai_pr_merged_at"]).dt.total_seconds() / 86400).round(2)

print("AI PRs with date :", pairs["ai_pr_created_at"].notna().sum(), "/", len(pairs))
print("fix PRs with date:", pairs["fix_pr_merged_at"].notna().sum(), "/", len(pairs))
print("AI PRs from", pairs["ai_pr_created_at"].min().date(), "to", pairs["ai_pr_created_at"].max().date())
print("latest fix     :", pairs["fix_pr_merged_at"].max().date())

out = pairs.copy()
for c in ["ai_pr_created_at", "ai_pr_merged_at", "fix_pr_created_at", "fix_pr_merged_at"]:
    out[c] = out[c].dt.strftime("%Y-%m-%d %H:%M")
out.to_csv(os.path.join(CLEAN, "CLEAN_AI_PR_FIXES_with_dates.csv"), index=False)
with pd.ExcelWriter(os.path.join(CLEAN, "CLEAN_AI_PR_FIXES_with_dates.xlsx")) as xw:
    out.to_excel(xw, sheet_name="pairs", index=False)
    pairs.groupby("agent").agg(pairs=("pair_id", "size"), first_ai_pr=("ai_pr_created_at", "min"), latest_ai_pr=("ai_pr_created_at", "max"),
                               median_days_to_fix=("days_ai_merge_to_fix_merge", "median")).astype(str).to_excel(xw, sheet_name="per agent")
print("Saved:", os.path.join(CLEAN, "CLEAN_AI_PR_FIXES_with_dates.xlsx"))
