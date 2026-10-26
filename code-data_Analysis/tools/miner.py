"""miner.py - read commit info (message, date, author, files) + code patches from PUBLIC GitHub repos with git only.
No API and no token: it makes small "blobless" clones and runs `git show <sha>`.

usage:  python code-data_Analysis/tools/miner.py <requests.csv> <out.jsonl> [--no-patch] [--workers N]
requests.csv columns: repo (owner/name), sha (commit id, or refs/pull/<n>/head), pr (optional PR number)
Clones go into Thesis_Data_Analysis/repos/bare/. You only need this to mine NEW commits (e.g. after linker.py).
"""
import csv, json, os, subprocess, sys, threading
from concurrent.futures import ThreadPoolExecutor
HERE = os.path.dirname(os.path.abspath(__file__))
BARE = os.path.normpath(os.path.join(HERE, "..", "..", "repos", "bare"))
os.makedirs(BARE, exist_ok=True)
if len(sys.argv) < 3:
    sys.exit(__doc__)
req, out = sys.argv[1], sys.argv[2]
no_patch = "--no-patch" in sys.argv
workers = int(sys.argv[sys.argv.index("--workers") + 1]) if "--workers" in sys.argv else 6
MAXP = 300_000
done = set()
if os.path.exists(out):
    for l in open(out):
        j = json.loads(l); done.add((j["repo"], j["sha"]))
lock = threading.Lock()
fo = open(out, "a")
def git(d, *a, timeout=600):
    return subprocess.run(["git", "-C", d, *a], capture_output=True, text=True, timeout=timeout, errors="replace")
by_repo = {}
for r in csv.DictReader(open(req)):
    if (r["repo"], r["sha"]) not in done:
        by_repo.setdefault(r["repo"], []).append(r)
print(f"{sum(map(len, by_repo.values()))} commits in {len(by_repo)} repos to mine", flush=True)
def work(item):
    repo, rows = item
    d = os.path.join(BARE, repo.replace("/", "__"))
    if not os.path.exists(d):
        p = subprocess.run(["git", "clone", "-q", "--bare", "--filter=blob:none", f"https://github.com/{repo}", d],
                           capture_output=True, text=True, timeout=3600, env={**os.environ, "GIT_LFS_SKIP_SMUDGE": "1", "GIT_TERMINAL_PROMPT": "0"})
        if p.returncode:
            with lock:
                for r in rows: fo.write(json.dumps({"repo": repo, "sha": r["sha"], "error": "clone: " + p.stderr[-200:]}) + "\n")
                fo.flush()
            return
    prs = sorted({r.get("pr") for r in rows if r.get("pr")})
    if prs:
        git(d, "fetch", "-q", "--filter=blob:none", "origin", *[f"+refs/pull/{n}/head:refs/pull/{n}/head" for n in prs])
    for r in rows:
        res = {"repo": repo, "sha": r["sha"]}
        m = git(d, "show", "-s", "--format=%H%x1f%aI%x1f%an%x1f%ae%x1f%cn%x1f%B", r["sha"])
        if m.returncode:
            res["error"] = "not found"
        else:
            full, date, an, ae, cn, body = m.stdout.split("\x1f", 5)
            res.update(full_sha=full, date=date, author=an, author_email=ae, committer=cn, message=body.strip())
            ns = git(d, "show", "--format=", "--numstat", full).stdout.strip().splitlines()
            res["files"] = [{"file": x.split("\t")[2], "add": x.split("\t")[0], "del": x.split("\t")[1]} for x in ns if x.count("\t") >= 2]
            if not no_patch:
                pt = git(d, "show", "--format=", "-p", "--no-color", "--no-ext-diff", full, timeout=900)
                res["patch"] = pt.stdout[:MAXP]; res["patch_truncated"] = len(pt.stdout) > MAXP
        with lock:
            fo.write(json.dumps(res) + "\n"); fo.flush()
    print(f"  done {repo} ({len(rows)})", flush=True)
with ThreadPoolExecutor(workers) as ex:
    list(ex.map(work, by_repo.items()))
print("ALL DONE", flush=True)
