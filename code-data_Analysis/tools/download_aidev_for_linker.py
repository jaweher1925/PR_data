"""
download_aidev_for_linker.py - get the AIDev tables that linker.py needs but AIDev v4 removed.

linker.py (Who Finishes / evolution-assurance, branch main) needs:
    pull_request, repository, pr_task_type, human_pull_request, human_pr_task_type
AIDev v4 (current main on Hugging Face) no longer has pr_task_type and the human_* tables,
so this script finds the newest older version that still has them and downloads the
whole set from THAT version (so all tables match).

Run on Windows (needs internet to huggingface.co):
    python download_aidev_for_linker.py
Files go to:  C:\\Users\\GIGABYTE\\Desktop\\code_theis\\evolution-assurance\\data\\raw_linker
"""
import os, shutil
from huggingface_hub import HfApi, hf_hub_download

REPO = "hao-li/AIDev"
OUT = r"C:\Users\GIGABYTE\Desktop\code_theis\evolution-assurance\data\raw_linker"
NEED = ["pull_request.parquet", "repository.parquet", "pr_task_type.parquet",
        "human_pull_request.parquet", "human_pr_task_type.parquet"]

api = HfApi()
refs = api.list_repo_refs(REPO, repo_type="dataset")
cands = [t.name for t in refs.tags] + [b.name for b in refs.branches]
print("Versions found:", cands)

chosen = None
for rev in sorted(cands, reverse=True):              # newest tag first (v3 before v2 ...)
    files = set(api.list_repo_files(REPO, repo_type="dataset", revision=rev))
    missing = [f for f in NEED if f not in files]
    print(f"  {rev}: {'has all tables' if not missing else 'missing ' + ', '.join(missing)}")
    if not missing and chosen is None:
        chosen = rev

if chosen is None:
    # fall back to the commit history: newest commit that still has all tables
    for c in api.list_repo_commits(REPO, repo_type="dataset"):
        files = set(api.list_repo_files(REPO, repo_type="dataset", revision=c.commit_id))
        if all(f in files for f in NEED):
            chosen = c.commit_id
            print(f"  using commit {c.commit_id[:10]} from {c.created_at:%Y-%m-%d}")
            break

if chosen is None:
    raise SystemExit("No version of AIDev with all 5 tables was found.")

print(f"\nDownloading from version: {chosen}")
os.makedirs(OUT, exist_ok=True)
for f in NEED:
    p = hf_hub_download(REPO, f, repo_type="dataset", revision=chosen)
    shutil.copy(p, os.path.join(OUT, f))
    print(f"  {f}: {os.path.getsize(os.path.join(OUT, f)) / 1e6:.1f} MB")
with open(os.path.join(OUT, "VERSION.txt"), "w") as fh:
    fh.write(f"hao-li/AIDev revision: {chosen}\n")
print("\nDone ->", OUT)
