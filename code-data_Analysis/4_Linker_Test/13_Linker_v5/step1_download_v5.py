"""
step1_download_v5.py - download the AIDev v5 tables linker_v5.py needs (only the needed columns).

  pull_request       361,296 agent PRs (curated part, with commit details)
  repository         25,402 repos (stars, language)
  pr_commit_details  only pr_id + filename (the full table is 12M rows with patches)

Run:  uv run --project %EA% python step1_download_v5.py      (~10-20 min, can be re-run)
"""
import os
import pyarrow.parquet as pq
from huggingface_hub import HfApi, HfFileSystem
from paths import RAW_V5, V5_REPO

COLUMNS = {
    "pull_request": ["id", "number", "title", "agent", "state", "created_at", "closed_at",
                     "merged_at", "repo_id", "repo_url", "html_url"],
    "repository": None,                                   # all columns (small)
    "pr_commit_details": ["pr_id", "filename"],
}
os.makedirs(RAW_V5, exist_ok=True)
fs = HfFileSystem()
sha = HfApi().dataset_info(V5_REPO).sha
print(f"AIDev v5 = {V5_REPO} @ {sha[:10]}")

for table, cols in COLUMNS.items():
    out = os.path.join(RAW_V5, f"{table}.parquet")
    if os.path.exists(out):
        print(f"  {table}: already there, skipped"); continue
    tmp = out + ".part"
    with fs.open(f"datasets/{V5_REPO}@{sha}/{table}.parquet", "rb", block_size=32 * 2**20) as fh:
        pf = pq.ParquetFile(fh)
        writer, n = None, 0
        for batch in pf.iter_batches(columns=cols, batch_size=500_000):   # stream, low memory
            if writer is None:
                writer = pq.ParquetWriter(tmp, batch.schema)
            writer.write_batch(batch); n += batch.num_rows
            print(f"\r  {table}: {n:,} / {pf.metadata.num_rows:,} rows", end="", flush=True)
        writer.close()
    os.replace(tmp, out)
    print(f"  -> {os.path.getsize(out) / 1e6:.0f} MB")

with open(os.path.join(RAW_V5, "VERSION.txt"), "w") as fh:
    fh.write(f"{V5_REPO} revision {sha}\n")
print(f"\nDone -> {RAW_V5}\nNext: step2_task_type_v5.py")
