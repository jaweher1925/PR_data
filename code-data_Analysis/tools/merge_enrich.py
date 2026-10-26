"""merge_enrich.py - HOW data/ENRICH/FINAL_AI_FIXES_ENRICHED.csv was made (kept for the record).
It needs the mined_*.jsonl files from miner.py (not in this folder), so you do not need to run it."""
import json, os, re, sys
import pandas as pd
sys.exit("merge_enrich.py is only a record of how data/ENRICH/FINAL_AI_FIXES_ENRICHED.csv was built.\n"
         "The result is already in data/ENRICH/ and data/FINAL_DATASET/ - nothing to run.\n"
         "To mine new commits use: python code-data_Analysis/tools/miner.py <requests.csv> <out.jsonl>")
W = "/home/claude/work/Paper_Datasets"; G = "/home/claude/gh"; OUT = "/home/claude/gh/out"; os.makedirs(OUT, exist_ok=True)
J = os.path.join(W, "01_WhoFinishes/data/pr-fix-authorship/data/judge")
load = lambda f: pd.DataFrame([json.loads(l) for l in open(os.path.join(G, f))]) if os.path.exists(os.path.join(G, f)) else None
AG = [("claude", "Claude Code"), ("anthropic", "Claude Code"), ("copilot", "Copilot"), ("devin", "Devin"), ("cursor", "Cursor"),
      ("codex", "Codex"), ("openai", "Codex"), ("chatgpt", "Codex"), ("jules", "Jules"), ("google-labs-jules", "Jules"), ("gemini", "Gemini"),
      ("aider", "Aider"), ("windsurf", "Windsurf"), ("codeium", "Windsurf"), ("amazon q", "Amazon Q"), ("amazonq", "Amazon Q"),
      ("codegen", "Codegen"), ("sweep", "Sweep"), ("openhands", "OpenHands"), ("all-hands", "OpenHands"), ("roo", "Roo Code"),
      ("cline", "Cline"), ("coderabbit", "CodeRabbit"), ("kiro", "Kiro"), ("opencode", "OpenCode"), ("factory", "Factory"),
      ("qwen", "Qwen Code"), ("amp", "Amp"), ("augment", "Augment"), ("lovable", "Lovable"), ("bolt", "Bolt"), ("v0", "v0")]
def agent_in(text):
    t = (text or "").lower()
    for k, n in AG:
        if re.search(r"(?<![a-z])" + re.escape(k) + r"(?![a-z])", t):
            return n
    return None
def agent_of_commit(r):
    trailers = " ".join(re.findall(r"co-authored-by:([^\n]+)", str(r.get("message", "")).lower()))
    return agent_in(trailers) or agent_in(f'{r.get("author", "")} {r.get("author_email", "")} {r.get("committer", "")}') or agent_in(str(r.get("message", ""))[-400:])
def stats(files):
    if not isinstance(files, list): return (None, None, None)
    n = lambda v: int(v) if str(v).isdigit() else 0
    return (len(files), sum(n(f["add"]) for f in files), sum(n(f["del"]) for f in files))

final = pd.read_csv(os.path.join(W, "FINAL_AI_FIXES_DATASET.csv"))
for c in ["ai_change_date", "fix_date", "fix_author", "fix_files", "fix_lines_added", "fix_lines_removed"]:
    final[c] = None
patches = []

# --- Sawada + Watanabe fix commits
sw = load("mined_sw.jsonl").set_index(["repo", "sha"])
for i, r in final[final.paper.isin(["Sawada 2026", "Watanabe 2025"])].iterrows():
    k = (r.repo, str(r.fix_change))
    if k not in sw.index or isinstance(sw.loc[k].get("error"), str): continue
    m = sw.loc[k]
    final.at[i, "fix_reason"] = m["message"].split("\n")[0]
    final.at[i, "fix_date"] = m["date"]; final.at[i, "fix_author"] = m["author"]
    final.at[i, "fix_files"], final.at[i, "fix_lines_added"], final.at[i, "fix_lines_removed"] = stats(m["files"])
    if isinstance(m.get("patch"), str) and m["patch"]:
        final.at[i, "has_code_patch"] = True
        patches.append({"fix_id": r.fix_id, "paper": r.paper, "agent": r.agent, "repo": r.repo, "fix_change": m["full_sha"], "patch": m["patch"]})
s = final.paper.eq("Sawada 2026") & final.fix_date.notna()
# Sawada: ai change = file creation date -> days_to_fix already known; ai_change_date = fix_date - days
final.loc[s, "ai_change_date"] = (pd.to_datetime(final.loc[s, "fix_date"], utc=True) - pd.to_timedelta(final.loc[s, "days_to_fix"].astype(float), unit="D")).dt.strftime("%Y-%m-%d")
w = final.paper.eq("Watanabe 2025") & final.fix_date.notna()
final.loc[w, "ai_change_date"] = (pd.to_datetime(final.loc[w, "fix_date"], utc=True) - pd.to_timedelta(final.loc[w, "days_to_fix"].astype(float), unit="D")).dt.strftime("%Y-%m-%d")

# --- Who Finishes: last-commit date of the AI PR and of the fix PR + fix patches from the paper's json
wf = load("mined_wf.jsonl"); wf["pr"] = wf["sha"].str.extract(r"pull/(\d+)/")[0]
wfd = {(r.repo, r.pr): (r.date, r.author) for r in wf.itertuples() if isinstance(r.date, str)}
def pr_key(u):
    p = str(u).rstrip("/").split("/"); return (p[3] + "/" + p[4], p[6]) if len(p) > 6 else None
cases = {str(c["pair_id"]): c for c in json.load(open(os.path.join(J, "rq1_judge_cases_population_12k.json"), encoding="utf-8"))}
v = pd.read_csv(os.path.join(J, "rq1_llm_judge_pop_anthropic_claude-opus-4-8_low.csv")); v = v[v.label == "Direct fix"]
url2pair = {(a, b): str(p) for a, b, p in zip(v.p_url, v.q_url, v.pair_id)}
for i, r in final[final.paper.eq("Who Finishes 2026")].iterrows():
    a, f = wfd.get(pr_key(r.ai_change)), wfd.get(pr_key(r.fix_change))
    if a: final.at[i, "ai_change_date"] = a[0]
    if f: final.at[i, "fix_date"], final.at[i, "fix_author"] = f
    c = cases.get(url2pair.get((r.ai_change, r.fix_change)), {})
    sh = c.get("shared") or []
    final.at[i, "fix_files"] = len(sh)
    txt = "".join(f"--- {s.get('name')}\n{s.get('q_patch') or ''}\n" for s in sh)
    final.at[i, "fix_lines_added"] = sum(1 for l in txt.splitlines() if l.startswith("+") and not l.startswith("+++"))
    final.at[i, "fix_lines_removed"] = sum(1 for l in txt.splitlines() if l.startswith("-") and not l.startswith("---"))
    patches.append({"fix_id": r.fix_id, "paper": r.paper, "agent": r.agent, "repo": r.repo, "fix_change": r.fix_change, "patch": txt})

# --- Xia & Miller (if mined)
xo, xf = load("mined_xia_origin.jsonl"), load("mined_xia_fix.jsonl")
if xo is not None and "--xia" in sys.argv:
    xi = pd.read_csv(os.path.join(W, "02_XiaMiller_PostMerge/tables/fixes.csv"))
    xo = xo[xo.get("error").isna()] if "error" in xo else xo
    oinfo = {(r["repo"], r["sha"]): r for r in xo.to_dict("records")}
    finfo = {(r["repo"], r["sha"]): r for r in (xf.to_dict("records") if xf is not None else [])}
    rows = []
    for r in xi.itertuples():
        o = oinfo.get((r.repo, r.ai_change)); f = finfo.get((r.repo, r.fix_change)) if isinstance(r.fix_change, str) else None
        if f is not None and isinstance(f.get("error"), str): f = None
        row = {"paper": "Xia & Miller 2026", "fix_id": r.fix_id, "agent": (agent_of_commit(o) if o else None) or "AI (not recognised)",
               "repo": r.repo, "language": r.language, "level": "commit", "ai_change": r.ai_change, "fix_change": r.fix_change,
               "fixed_by": "Unknown", "days_to_fix": r.days_to_fix, "fix_label": r.fix_label, "fix_reason": None,
               "has_code_patch": False, "is_python": r.language == "Python", "ai_change_date": o["date"] if o else None}
        if f:
            fa = agent_of_commit(f)
            row.update(fixed_by="AI agent" if fa else "Human", fix_reason=f["message"].split("\n")[0], fix_date=f["date"], fix_author=f["author"])
            row["fix_files"], row["fix_lines_added"], row["fix_lines_removed"] = stats(f.get("files"))
            if f.get("patch"):
                row["has_code_patch"] = True
                patches.append({"fix_id": r.fix_id, "paper": row["paper"], "agent": row["agent"], "repo": r.repo, "fix_change": f["full_sha"], "patch": f["patch"]})
        rows.append(row)
    final = pd.concat([final, pd.DataFrame(rows)], ignore_index=True)

final["fixed_by"] = final["fixed_by"].fillna("Unknown")
cols = ["paper", "fix_id", "agent", "repo", "language", "is_python", "level", "ai_change", "ai_change_date", "fix_change", "fix_date",
        "fixed_by", "fix_author", "days_to_fix", "fix_label", "fix_reason", "fix_files", "fix_lines_added", "fix_lines_removed", "has_code_patch"]
final = final[cols].sort_values(["agent", "paper", "repo"]).reset_index(drop=True)
p = pd.DataFrame(patches)
final.to_csv(f"{OUT}/FINAL_AI_FIXES_DATASET.csv", index=False)
p.to_csv(f"{OUT}/FINAL_FIX_PATCHES.csv", index=False)
cov = final.groupby("paper").agg(lambda s: f"{s.notna().mean() * 100:.0f}%").T
print(len(final), "fixes |", int(final.has_code_patch.sum()), "with code |", len(p), "patch rows")
print(cov.to_string())
print(final.groupby(["paper", "agent"]).size().to_string())
