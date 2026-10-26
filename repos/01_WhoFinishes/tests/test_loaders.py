"""Smoke tests for the data-loading plumbing.

These do not download AIDev. They prove the core DuckDB-over-Parquet
capability we rely on and that the package's modules import cleanly.
"""

from __future__ import annotations

import duckdb
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


def test_package_imports():
    """Top-level package modules import without errors."""
    from prfix import config
    from prfix.common import metrics
    from prfix.common.loaders import aidev, base

    assert hasattr(config, "AIDEV_TABLES")
    assert hasattr(aidev, "ingest")
    assert hasattr(base, "connect")
    assert hasattr(metrics, "km_cuminc")


def test_duckdb_view_over_parquet(tmp_path):
    """DuckDB can create a view over a Parquet file and query it."""
    df = pd.DataFrame({"agent": ["Codex", "Devin"], "n": [1, 2]})
    p = tmp_path / "sample.parquet"
    pq.write_table(pa.Table.from_pandas(df), p)

    con = duckdb.connect(":memory:")
    con.execute(f"CREATE VIEW t AS SELECT * FROM read_parquet('{p}')")
    rows = con.execute("SELECT agent, n FROM t ORDER BY agent").fetchall()
    con.close()

    assert rows == [("Codex", 1), ("Devin", 2)]
