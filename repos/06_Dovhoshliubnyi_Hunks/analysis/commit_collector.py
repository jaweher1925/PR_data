import os
import re
import sqlite3

from config import BUSINESS_LOGIC_EXTENSIONS, EXCLUDED_FILE_PATTERNS, RELEVANCE_THRESHOLD


def collect_commit_details(conn: sqlite3.Connection) -> dict:
    """
    Join performance-relevant PRs -> pr_commit_details and
    insert results into performance_commit_details.

    Prefers the validated set (pr_validation.is_performance_related = 1).
    Falls back to relevance_score >= threshold when no validated rows exist.

    Returns stats: total_prs, total_commits, total_files.
    """
    validated_count = conn.execute(
        "SELECT COUNT(*) FROM pr_validation WHERE is_performance_related = 1"
    ).fetchone()[0]

    if validated_count > 0:
        query = """
            SELECT
                pv.pr_id,
                pcd.sha,
                pcd.filename,
                pcd.author,
                pcd.committer,
                pcd.message,
                pcd.commit_stats_total,
                pcd.commit_stats_additions,
                pcd.commit_stats_deletions,
                pcd.status,
                pcd.additions,
                pcd.deletions,
                pcd.changes,
                pcd.patch
            FROM pr_validation pv
            JOIN pr_commit_details pcd ON pcd.pr_id = pv.pr_id
            WHERE pv.is_performance_related = 1
        """
        rows = conn.execute(query).fetchall()
    else:
        query = """
            SELECT
                ppc.pr_id,
                pcd.sha,
                pcd.filename,
                pcd.author,
                pcd.committer,
                pcd.message,
                pcd.commit_stats_total,
                pcd.commit_stats_additions,
                pcd.commit_stats_deletions,
                pcd.status,
                pcd.additions,
                pcd.deletions,
                pcd.changes,
                pcd.patch
            FROM pr_performance_classification ppc
            JOIN pr_commit_details pcd ON pcd.pr_id = ppc.pr_id
            WHERE ppc.relevance_score >= ?
        """
        rows = conn.execute(query, (RELEVANCE_THRESHOLD,)).fetchall()

    total_before = len(rows)
    rows = [
        row for row in rows
        if not any(re.search(pattern, row[2]) for pattern in EXCLUDED_FILE_PATTERNS)
        and os.path.splitext(row[2])[1] in BUSINESS_LOGIC_EXTENSIONS
    ]
    filtered_files = total_before - len(rows)

    conn.execute("DELETE FROM performance_commit_details")
    conn.executemany(
        """
        INSERT OR IGNORE INTO performance_commit_details (
            pr_id, sha, filename,
            author, committer, message,
            commit_stats_total, commit_stats_additions, commit_stats_deletions,
            status, additions, deletions, changes, patch
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )
    conn.commit()

    total_prs = conn.execute(
        "SELECT COUNT(DISTINCT pr_id) FROM performance_commit_details"
    ).fetchone()[0]
    total_commits = conn.execute(
        "SELECT COUNT(DISTINCT sha) FROM performance_commit_details"
    ).fetchone()[0]
    total_files = conn.execute(
        "SELECT COUNT(*) FROM performance_commit_details"
    ).fetchone()[0]

    return {
        "total_prs": total_prs,
        "total_commits": total_commits,
        "total_files": total_files,
        "filtered_files": filtered_files,
    }
