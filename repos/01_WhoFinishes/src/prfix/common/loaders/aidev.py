"""Download AIDev Parquet files from Hugging Face and register them as DuckDB views."""

from __future__ import annotations

import shutil
from pathlib import Path

from huggingface_hub import hf_hub_download
from rich.console import Console

from prfix.common.loaders.base import connect
from prfix.config import AIDEV_HF_REPO, AIDEV_TABLES, RAW_DIR, ensure_dirs

console = Console()


def download_table(filename: str) -> Path:
    """Download a single AIDev Parquet to data/raw/ if not already present."""
    target = RAW_DIR / filename
    if target.exists():
        console.log(f"[dim]exists[/dim]   {filename}")
        return target
    console.log(f"[bold]download[/bold] {filename}")
    cached = hf_hub_download(
        repo_id=AIDEV_HF_REPO,
        filename=filename,
        repo_type="dataset",
    )
    # Prefer a symlink to keep the HF cache canonical; fall back to copy.
    try:
        target.symlink_to(cached)
    except OSError:
        shutil.copy(cached, target)
    return target


def register_views(con) -> int:
    """Register a DuckDB view over each downloaded AIDev Parquet file.

    Returns the number of views registered.
    """
    registered = 0
    for table, filename in AIDEV_TABLES.items():
        path = RAW_DIR / filename
        if not path.exists():
            console.log(f"[yellow]skip[/yellow]    {table}: {filename} not downloaded")
            continue
        con.execute(
            f"CREATE OR REPLACE VIEW {table} AS "
            f"SELECT * FROM read_parquet('{path}')"
        )
        console.log(f"[green]view[/green]    {table}")
        registered += 1
    return registered


def ingest(only: list[str] | None = None) -> None:
    """Download AIDev Parquets and register DuckDB views.

    Pass `only=[...]` to limit to a subset of friendly table names.
    """
    ensure_dirs()
    tables = AIDEV_TABLES if not only else {k: v for k, v in AIDEV_TABLES.items() if k in only}
    if not tables:
        console.print(f"[red]no matching tables in --only filter:[/red] {only}")
        return

    console.rule("[bold]Downloading AIDev tables[/bold]")
    for filename in tables.values():
        try:
            download_table(filename)
        except Exception as exc:
            console.log(f"[red]error[/red]   {filename}: {exc}")

    console.rule("[bold]Registering DuckDB views[/bold]")
    con = connect()
    try:
        n_views = register_views(con)
        console.print(f"\n[bold]Views registered:[/bold] {n_views}")
        # Smoke check on the headline table when present.
        headline = (RAW_DIR / AIDEV_TABLES["all_pull_request"])
        if headline.exists():
            n_rows = con.execute("SELECT COUNT(*) FROM all_pull_request").fetchone()[0]
            console.print(f"[bold green]all_pull_request[/bold green]: {n_rows:,} rows")
    finally:
        con.close()


if __name__ == "__main__":  # pragma: no cover
    ingest()
