"""
check_v5_tables.py - are pr_task_type / human PRs really gone in AIDev v5, or only renamed / moved?

For AIDev v3 (what linker.py uses) and AIDev v5 (hao-li/AIDev-7.6M) it prints:
  1. every table and its columns (reads only the file footer, no big download)
  2. columns that look like a task type or a human/bot flag
  3. the values of the 'agent' column (are human PRs inside the PR tables?)
  4. lines of the v5 README / scripts that mention task type or human PRs
Run:  uv run --project %EA% python check_v5_tables.py   (output also saved to check_v5_tables.txt)
"""
import io
import re

import pyarrow.parquet as pq
from huggingface_hub import HfApi, HfFileSystem

OUT = io.StringIO()


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    OUT.write(s + "\n")


api, fs = HfApi(), HfFileSystem()
SOURCES = {"v3 (what linker.py uses)": ("hao-li/AIDev", "v3"),
           "v5 (AIDev-7.6M)": ("hao-li/AIDev-7.6M", "main")}
HINT = re.compile(r"type|task|label|categor|convention|class|human|bot|author_assoc|agent", re.I)

for name, (repo, rev) in SOURCES.items():
    say("\n" + "=" * 90 + f"\n{name}: {repo} @ {rev}\n" + "=" * 90)
    files = [f for f in api.list_repo_files(repo, repo_type="dataset", revision=rev) if f.endswith(".parquet")]
    for f in sorted(files):
        path = f"datasets/{repo}@{rev}/{f}"
        try:
            with fs.open(path, "rb") as fh:
                pf = pq.ParquetFile(fh)
                cols = pf.schema_arrow.names
                say(f"\n{f}  ({pf.metadata.num_rows:,} rows)")
                say("   columns:", ", ".join(cols))
                hits = [c for c in cols if HINT.search(c)]
                if hits:
                    say("   -> interesting columns:", ", ".join(hits))
                if "agent" in cols:
                    vc = pf.read(columns=["agent"]).column("agent").to_pandas().value_counts(dropna=False)
                    say("   -> agent values:", dict(vc))
                for c in ("type", "task_type", "pr_type", "label", "category"):
                    if c in cols:
                        vc = pf.read(columns=[c]).column(c).to_pandas().value_counts(dropna=False).head(15)
                        say(f"   -> {c} values:", dict(vc))
        except Exception as e:  # keep going on any file
            say(f"\n{f}: could not read ({e})")

say("\n" + "=" * 90 + "\nv5 README / scripts: lines about task type or human PRs\n" + "=" * 90)
for f in api.list_repo_files("hao-li/AIDev-7.6M", repo_type="dataset"):
    if f.endswith((".md", ".py")):
        with fs.open(f"datasets/hao-li/AIDev-7.6M/{f}", "r", encoding="utf-8") as fh:
            text = fh.read()
        for i, line in enumerate(text.splitlines(), 1):
            if re.search(r"task.?type|human|pr_type|conventional|classif|removed|deprecat", line, re.I):
                say(f"{f}:{i}: {line.strip()[:200]}")

with open("check_v5_tables.txt", "w", encoding="utf-8") as fh:
    fh.write(OUT.getvalue())
print("\nSaved check_v5_tables.txt")
