"""Shared DuckDB connection helper."""

from __future__ import annotations

import duckdb

from prfix.config import (
    DUCKDB_MEMORY_LIMIT,
    DUCKDB_PATH,
    DUCKDB_THREADS,
    ensure_dirs,
)


def connect(read_only: bool = False) -> duckdb.DuckDBPyConnection:
    """Open a DuckDB connection at the project's database path with tuned defaults.

    DuckDB is used out-of-core: views over the raw Parquet files mean the
    database file stays small and queries stream directly from disk.
    """
    ensure_dirs()
    con = duckdb.connect(str(DUCKDB_PATH), read_only=read_only)
    con.execute(f"PRAGMA memory_limit='{DUCKDB_MEMORY_LIMIT}'")
    con.execute(f"PRAGMA threads={DUCKDB_THREADS}")
    return con
