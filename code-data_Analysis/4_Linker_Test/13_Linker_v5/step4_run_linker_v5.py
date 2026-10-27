"""
step4_run_linker_v5.py - run linker_v5.py (Mohamed's maintenance linker adapted to v5) twice:

  results/all_changes_30_60   every later agent PR touching the same files (Mohamed's default)
  results/fix_only_30_60      only later PRs typed 'fix' or 'Revert ...'

Run:  uv run --project %EA% python step4_run_linker_v5.py
"""
import os, subprocess, sys
from paths import DB, HERE, RESULTS

env = dict(os.environ, AIDEV_V5_DB=DB)
runs = {"all_changes_30_60": ["--candidate-mode", "all_changes"],
        "fix_only_30_60": ["--candidate-mode", "fix_only"]}
for name, opts in runs.items():
    out = os.path.join(RESULTS, name)
    print(f"\n=== {name} ===")
    subprocess.check_call([sys.executable, os.path.join(HERE, "linker_v5.py"), *opts,
                           "--windows", "30", "60", "--window-rule", "elapsed",
                           "--pairs-format", "both", "--output-dir", out], env=env)
print(f"\nDone. Results in {RESULTS}. Next: open step5_results_v5.ipynb")
