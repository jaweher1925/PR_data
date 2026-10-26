"""
build_clean_pr_dataset.py - the CLEAN dataset of AI pull requests and the pull requests that fix them.

Source: Who Finishes the Job? (Takerngsaksiri 2026) - 2,510 AI PR -> later PR pairs labelled by an LLM judge.
Cleaning rules:
  1. keep only pairs labelled "Direct fix"
  2. keep only pairs where the fix PR has a real code patch on a non-boilerplate file
  3. keep only pairs with judge confidence >= 0.70
  4. one row per (AI PR, fix PR) pair; flag fix PRs that fix several AI PRs
Output: CLEAN_AI_PR_FIXES.xlsx / .csv (one row per pair) and CLEAN_AI_PR_FIX_FILES.csv (one row per file, with both patches)
Run from anywhere:  python code-data_Analysis/tools/build_clean_pr_dataset.py
Reads  data/01_WhoFinishes/pr-fix-authorship/data/judge/   ->   writes data/CLEAN_PR_DATASET/
"""
import os, json, sys, gzip
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.normpath(os.path.join(HERE, "..", "..", "data"))
J = sys.argv[1] if len(sys.argv) > 1 else os.path.join(DATA, "01_WhoFinishes", "pr-fix-authorship", "data", "judge")
OUT = os.path.join(DATA, "CLEAN_PR_DATASET"); os.makedirs(OUT, exist_ok=True)
def load_json(p):
    p = p if os.path.exists(p) else p + ".gz"
    with (gzip.open if p.endswith(".gz") else open)(p, "rt", encoding="utf-8") as f:
        return json.load(f)
MIN_CONF = 0.70
AGENT = {"OpenAI_Codex": "Codex", "Claude_Code": "Claude Code"}
EXT = {".py": "Python", ".ts": "TypeScript", ".tsx": "TypeScript", ".js": "JavaScript", ".jsx": "JavaScript", ".go": "Go",
       ".rs": "Rust", ".java": "Java", ".cs": "C#", ".rb": "Ruby", ".php": "PHP", ".cpp": "C++", ".c": "C", ".kt": "Kotlin",
       ".swift": "Swift", ".vue": "Vue", ".svelte": "Svelte", ".dart": "Dart", ".scala": "Scala", ".sh": "Shell"}
lang = lambda f: EXT.get(os.path.splitext(str(f))[1].lower(), "Other (config, docs, data)")

v = pd.read_csv(os.path.join(J, "rq1_llm_judge_pop_anthropic_claude-opus-4-8_low.csv"))
c = pd.DataFrame(load_json(os.path.join(J, "rq1_judge_cases_population_12k.json")))
v["pair_id"] = v["pair_id"].astype(str); c["pair_id"] = c["pair_id"].astype(str)
d = v.merge(c[["pair_id", "p_id", "q_id", "p_title", "q_title", "p_commits", "q_commits", "day_gap", "n_intervening", "q_references_p", "shared"]], on="pair_id")
log = [("All candidate pairs (AI PR -> later PR)", len(d))]

d = d[d["label"] == "Direct fix"]; log.append(("1. Labelled 'Direct fix' by the LLM judge", len(d)))

# one row per shared file
f = d[["pair_id", "shared"]].explode("shared").dropna(subset=["shared"])
f = pd.DataFrame({"pair_id": f["pair_id"].values, "file": [s.get("name") for s in f["shared"]],
                  "boilerplate": [bool(s.get("boilerplate")) for s in f["shared"]],
                  "ai_patch": [s.get("p_patch") for s in f["shared"]], "fix_patch": [s.get("q_patch") for s in f["shared"]]})
f["language"] = f["file"].map(lang)
f["is_code"] = ~f["language"].str.startswith("Other")
f["fix_lines_added"] = f["fix_patch"].fillna("").map(lambda p: sum(1 for l in p.splitlines() if l.startswith("+") and not l.startswith("+++")))
f["fix_lines_removed"] = f["fix_patch"].fillna("").map(lambda p: sum(1 for l in p.splitlines() if l.startswith("-") and not l.startswith("---")))
good = f[~f["boilerplate"] & f["fix_patch"].fillna("").str.len().gt(0)]
d = d[d["pair_id"].isin(good["pair_id"])]; log.append(("2. Fix PR has a real code patch (not boilerplate)", len(d)))

d = d[d["confidence"] >= MIN_CONF]; log.append((f"3. Judge confidence >= {MIN_CONF}", len(d)))
d = d.drop_duplicates(["p_id", "q_id"]); log.append(("4. One row per (AI PR, fix PR) pair", len(d)))
good = good[good["pair_id"].isin(d["pair_id"])]

per = good.groupby("pair_id").agg(files=("file", "size"), code_files=("is_code", "sum"),
        main_language=("language", lambda s: s[~s.str.startswith("Other")].mode().iloc[0] if (~s.str.startswith("Other")).any() else "Other (config, docs, data)"),
        file_list=("file", lambda s: "; ".join(s)), fix_lines_added=("fix_lines_added", "sum"), fix_lines_removed=("fix_lines_removed", "sum"))
d = d.merge(per, left_on="pair_id", right_index=True)
q_h = d["q_is_human"].astype(str).str.lower().eq("true")
out = pd.DataFrame({
    "pair_id": d["pair_id"], "agent": d["agent"].replace(AGENT), "repo": d["p_url"].str.extract(r"github\.com/([^/]+/[^/]+)")[0],
    "main_language": d["main_language"], "is_python": d["main_language"].eq("Python"),
    "ai_pr_id": d["p_id"].astype(str), "ai_pr_url": d["p_url"], "ai_pr_title": d["p_title"],
    "ai_pr_created_at": None, "ai_pr_merged_at": None,
    "fix_pr_id": d["q_id"].astype(str), "fix_pr_url": d["q_url"], "fix_pr_title": d["q_title"],
    "fix_pr_created_at": None, "fix_pr_merged_at": None,
    "fixed_by": q_h.map({True: "Human", False: "AI agent"}), "days_between_merges": pd.to_numeric(d["day_gap"], errors="coerce"),
    "prs_in_between": pd.to_numeric(d["n_intervening"], errors="coerce"), "fix_mentions_ai_pr": d["q_references_p"].astype(str).str.lower().eq("true"),
    "files_fixed": d["files"], "code_files_fixed": d["code_files"], "fix_lines_added": d["fix_lines_added"], "fix_lines_removed": d["fix_lines_removed"],
    "file_list": d["file_list"], "judge_confidence": d["confidence"], "judge_reason": d["rationale"]})
out["fix_pr_fixes_n_ai_prs"] = out.groupby("fix_pr_id")["pair_id"].transform("size")
out = out.sort_values(["agent", "repo", "ai_pr_id"]).reset_index(drop=True)
files = good.drop(columns=["boilerplate"]).merge(out[["pair_id", "agent", "repo", "ai_pr_url", "fix_pr_url"]], on="pair_id")

log = pd.DataFrame(log, columns=["step", "pairs left"])
print(log.to_string(index=False))
by_agent = out.groupby("agent").agg(pairs=("pair_id", "size"), repos=("repo", "nunique"), ai_prs=("ai_pr_id", "nunique"), fix_prs=("fix_pr_id", "nunique"),
            fixed_by_human=("fixed_by", lambda s: (s == "Human").sum()), fixed_by_AI=("fixed_by", lambda s: (s == "AI agent").sum()),
            python_pairs=("is_python", "sum"), median_days=("days_between_merges", "median")).sort_values("pairs", ascending=False)
print(by_agent.to_string())

out.to_csv(os.path.join(OUT, "CLEAN_AI_PR_FIXES.csv"), index=False)
files.to_csv(os.path.join(OUT, "CLEAN_AI_PR_FIX_FILES.csv.gz"), index=False)
short = files.copy()
for col in ("ai_patch", "fix_patch"):
    short[col] = short[col].fillna("").str[:30000]          # Excel cell limit
with pd.ExcelWriter(os.path.join(OUT, "CLEAN_AI_PR_FIXES.xlsx")) as xw:
    pd.DataFrame({"sheet": ["PR pairs", "files + patches", "per agent", "python only", "cleaning steps"],
                  "content": ["one row per AI PR -> fix PR (clean)", "one row per file changed by the fix, with the AI patch and the fix patch",
                              "numbers per AI agent", "pairs whose main language is Python", "how the dataset was cleaned"]}).to_excel(xw, sheet_name="README", index=False)
    out.to_excel(xw, sheet_name="PR pairs", index=False)
    short.to_excel(xw, sheet_name="files + patches", index=False)
    by_agent.reset_index().to_excel(xw, sheet_name="per agent", index=False)
    out[out.is_python].to_excel(xw, sheet_name="python only", index=False)
    log.to_excel(xw, sheet_name="cleaning steps", index=False)
print(f"\nSaved in {OUT}: {len(out)} clean PR pairs, {len(files)} file rows")
