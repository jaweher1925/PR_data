"""Load MSR paper's validated performance PR data into the database."""

import csv
import sqlite3
from pathlib import Path

import pyarrow.parquet as pq

from config import MSR_DATA_DIR


def load_msr_data(conn: sqlite3.Connection, verbose: bool = True) -> dict:
    """Load Agent_PRs.csv and ai_pr_commit_details.parquet into the database.

    Returns stats dict with counts.
    """
    data_dir = Path(MSR_DATA_DIR)
    agent_prs_path = data_dir / "Agent_PRs.csv"
    parquet_path = data_dir / "ai_pr_commit_details.parquet"
    perf_csv_path = data_dir / "performance_prs_ai_vs_human.csv"

    # --- Load agent info from the full CSV (has 'agent' column) ---
    agent_map = {}
    if perf_csv_path.exists():
        with open(perf_csv_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                pr_id = int(row["id"])
                agent_map[pr_id] = row.get("agent", "")

    # --- Load perf_pr table from Agent_PRs.csv ---
    existing = conn.execute("SELECT COUNT(*) FROM perf_pr").fetchone()[0]
    if existing > 0:
        if verbose:
            print(f"  [perf_pr] already has {existing} rows, skipping.")
    else:
        with open(agent_prs_path) as f:
            reader = csv.DictReader(f)
            rows = []
            for row in reader:
                pr_id = int(row["PR ID"])
                rows.append((
                    pr_id,
                    row["PR URL"],
                    agent_map.get(pr_id, ""),
                    row["Final Pattern"],
                    row["Final Sub Pattern"],
                    row["Label Source"],
                ))
            conn.executemany(
                "INSERT OR IGNORE INTO perf_pr (pr_id, pr_url, agent, pattern, sub_pattern, label_source) VALUES (?,?,?,?,?,?)",
                rows,
            )
            conn.commit()
            if verbose:
                print(f"  [perf_pr] inserted {len(rows)} rows.")

    # --- Load commit_details from parquet, with file filtering ---
    existing = conn.execute("SELECT COUNT(*) FROM commit_details").fetchone()[0]
    if existing > 0:
        if verbose:
            print(f"  [commit_details] already has {existing} rows, skipping.")
        return {"perf_prs": conn.execute("SELECT COUNT(*) FROM perf_pr").fetchone()[0],
                "commit_details": existing}

    pr_ids = set(r[0] for r in conn.execute("SELECT pr_id FROM perf_pr").fetchall())
    table = pq.read_table(str(parquet_path))
    df = table.to_pandas()

    def safe_int(val):
        if val is None:
            return None
        try:
            import math
            if math.isnan(val):
                return None
        except (TypeError, ValueError):
            pass
        return int(val)

    def safe_str(val):
        if val is None:
            return None
        s = str(val)
        return None if s == "nan" else s

    total = 0
    filtered = 0
    null_patch = 0
    batch = []
    for _, row in df.iterrows():
        pr_id = int(row["pr_id"])
        if pr_id not in pr_ids:
            filtered += 1
            continue
        filename = safe_str(row["filename"]) or ""
        if not filename:
            filtered += 1
            continue
        patch = row["patch"]
        if patch is None or (isinstance(patch, float)) or str(patch) == "nan":
            null_patch += 1
            continue
        batch.append((
            str(row["sha"]),
            pr_id,
            safe_int(row["pr_number"]),
            safe_int(row["commit_stats_total"]),
            safe_int(row["commit_stats_additions"]),
            safe_int(row["commit_stats_deletions"]),
            filename,
            safe_str(row["status"]),
            safe_int(row["additions"]),
            safe_int(row["deletions"]),
            safe_int(row["changes"]),
            str(patch),
            safe_str(row["repo_owner"]),
            safe_str(row["repo_name"]),
            safe_str(row["commit_sha"]),
        ))
        total += 1

    conn.executemany(
        """INSERT OR IGNORE INTO commit_details
           (sha, pr_id, pr_number, commit_stats_total, commit_stats_additions,
            commit_stats_deletions, filename, status, additions, deletions, changes,
            patch, repo_owner, repo_name, commit_sha)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        batch,
    )
    conn.commit()

    stats = {
        "perf_prs": len(pr_ids),
        "commit_details": total,
        "filtered_out": filtered,
        "null_patches": null_patch,
    }
    if verbose:
        print(f"  [commit_details] inserted {total} rows (filtered {filtered}, null patches {null_patch}).")
    return stats
