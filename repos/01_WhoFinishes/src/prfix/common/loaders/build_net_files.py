"""Bake the ``pr_net_files`` table from the GitHub net-diff cache (reproducible).

The RQ1 follow-up linker historically used the per-commit *union* of files
(``pr_commit_details``), which over-links: it counts files touched only in
intermediate or merge-from-main commits that never reach the merged diff. This
materialises each PR's NET (base->head) changed-file list -- exactly the GitHub
"Files changed" tab -- so the linker can require a net-diff co-edit instead.

Source: ``data/interim/netdiff_cache.jsonl`` (produced by the net-diff fetch).
``is_boilerplate`` is precomputed here (see :mod:`prfix.rq1_fix_rates.boilerplate`)
so the linker stays pure SQL. Table schema: ``pr_net_files(pr_id, filename,
is_boilerplate)``. Regenerable; the DuckDB lives under the gitignored ``data/``.

Run: uv run python -m prfix.common.loaders.build_net_files
"""
from __future__ import annotations

import json

import pandas as pd

from prfix.common.loaders.base import connect
from prfix.config import INTERIM_DIR
from prfix.rq1_fix_rates.boilerplate import is_boilerplate

CACHE = INTERIM_DIR / "netdiff_cache.jsonl"


def build() -> None:
    if not CACHE.exists():
        raise FileNotFoundError(
            f"{CACHE} not found -- run the net-diff fetch first "
            "(fetch_netdiff_all) to populate the cache.")
    rows, ok, unavailable = [], 0, 0
    for ln in CACHE.read_text().splitlines():
        d = json.loads(ln)
        if d["status"] == "ok":
            ok += 1
            for f in d["files"]:
                rows.append((int(d["pr_id"]), f))
        else:
            unavailable += 1
    df = pd.DataFrame(rows, columns=["pr_id", "filename"]).drop_duplicates()
    df["is_boilerplate"] = df["filename"].map(is_boilerplate)

    con = connect(read_only=False)
    try:
        con.register("net_files_df", df)
        con.execute("DROP TABLE IF EXISTS pr_net_files")
        con.execute("CREATE TABLE pr_net_files AS SELECT * FROM net_files_df")
        n = con.execute("SELECT count(*) FROM pr_net_files").fetchone()[0]
        npr = con.execute("SELECT count(DISTINCT pr_id) FROM pr_net_files").fetchone()[0]
        nb = con.execute("SELECT count(*) FROM pr_net_files WHERE is_boilerplate").fetchone()[0]
    finally:
        con.close()
    print(f"pr_net_files built: {n} file-rows, {npr} PRs ok "
          f"({unavailable} PRs unavailable/empty), {nb} boilerplate rows "
          f"({100*nb/max(n,1):.1f}%)")


if __name__ == "__main__":
    build()
