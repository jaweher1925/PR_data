"""
step3_fetch_changed_files.py - get from GitHub the files changed by every candidate PR
(about 6,000 PRs, 1-2 hours). Uses Mohamed's own script (prfix.common.loaders.fetch_netdiff).

Needs:  GitHub CLI 'gh'  (Windows: winget install --id GitHub.cli, then reopen the terminal)
        GITHUB_TOKEN in evolution-assurance/.env (already there)
Run:    uv run --project "C:\\Users\\GIGABYTE\\Desktop\\code_theis\\evolution-assurance" python step3_fetch_changed_files.py
It can be stopped and restarted: PRs already fetched are skipped.
"""
import os, shutil, subprocess, sys
from paths import REPO

# 1. token from .env -> GH_TOKEN (what gh uses)
env = dict(os.environ)
envfile = os.path.join(REPO, ".env")
if os.path.exists(envfile):
    for line in open(envfile, encoding="utf-8"):
        if line.strip().startswith("GITHUB_TOKEN=") and line.split("=", 1)[1].strip():
            env["GH_TOKEN"] = line.split("=", 1)[1].strip().strip('"')
if "GH_TOKEN" not in env:
    raise SystemExit("No GITHUB_TOKEN found in .env")

# 2. gh must be installed (also accept ~/bin/gh on Linux)
home_bin = os.path.expanduser("~/bin")
env["PATH"] = home_bin + os.pathsep + env.get("PATH", "")
if not shutil.which("gh", path=env["PATH"]):
    raise SystemExit("GitHub CLI 'gh' not found. Windows: winget install --id GitHub.cli")

# 3. run Mohamed's fetch (unchanged)
sys.exit(subprocess.call([sys.executable, "-m", "prfix.common.loaders.fetch_netdiff"], cwd=REPO, env=env))
