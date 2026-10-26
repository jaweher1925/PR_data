"""
clone_all.py - download the code of every paper and tool we use (GitHub repositories).

Run from anywhere:   python code-data_Analysis/tools/clone_all.py
Needs git. Repositories go into  Thesis_Data_Analysis/repos/<folder>  (already-downloaded ones are skipped).
RefactoringMiner is big (about 3.5 GB) - skipped unless you run:  python code-data_Analysis/tools/clone_all.py --with-rm
"""
import csv, os, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.normpath(os.path.join(HERE, "..", "..", "repos"))
os.makedirs(OUT, exist_ok=True)
with_rm = "--with-rm" in sys.argv

for row in csv.DictReader(open(os.path.join(HERE, "repos.csv"), encoding="utf-8")):
    folder, url = row["folder"], row["url"]
    target = os.path.join(OUT, folder)
    if folder == "09_RefactoringMiner" and not with_rm:
        print(f"skip  {folder}  (big; use --with-rm)")
        continue
    if os.path.exists(target):
        print(f"ok    {folder}  (already here)")
        continue
    print(f"get   {folder}  <- {url}")
    subprocess.run(["git", "clone", "--depth", "1", url, target], check=False)

print(f"\nDone. Repositories are in: {OUT}")
print("Also download by hand: AIDev (Hugging Face hao-li/AIDev), DECODE data (Hugging Face jtliang/decode).")
