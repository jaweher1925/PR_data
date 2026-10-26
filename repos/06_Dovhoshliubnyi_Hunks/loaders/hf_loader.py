import sqlite3
from datasets import load_dataset
from tqdm import tqdm
from config import HF_DATASET, HF_CONFIGS, BATCH_SIZE


def _table_has_rows(conn: sqlite3.Connection, table: str) -> bool:
    cur = conn.execute(f"SELECT COUNT(*) FROM {table}")
    return cur.fetchone()[0] > 0


def _get_columns(conn: sqlite3.Connection, table: str) -> list[str]:
    cur = conn.execute(f"PRAGMA table_info({table})")
    return [row[1] for row in cur.fetchall()]


def _insert_batch(conn: sqlite3.Connection, table: str, columns: list[str], rows: list[tuple]):
    placeholders = ", ".join("?" * len(columns))
    col_list = ", ".join(columns)
    sql = f"INSERT OR IGNORE INTO {table} ({col_list}) VALUES ({placeholders})"
    conn.executemany(sql, rows)
    conn.commit()


def _row_to_tuple(record: dict, columns: list[str]) -> tuple:
    return tuple(record.get(col) for col in columns)


def load_all(conn: sqlite3.Connection, verbose: bool = True) -> None:
    conn.execute("PRAGMA foreign_keys=OFF")
    for config_name, table_name in HF_CONFIGS.items():
        if _table_has_rows(conn, table_name):
            if verbose:
                print(f"  [{table_name}] already populated, skipping.")
            continue

        if verbose:
            print(f"  Loading {config_name} -> {table_name} ...")

        columns = _get_columns(conn, table_name)

        try:
            dataset = load_dataset(HF_DATASET, config_name, split="train")
        except Exception as e:
            print(f"  WARNING: could not load {config_name}: {e}")
            continue

        batch: list[tuple] = []
        total = 0
        for record in tqdm(dataset, desc=table_name, disable=not verbose):
            batch.append(_row_to_tuple(record, columns))
            if len(batch) >= BATCH_SIZE:
                _insert_batch(conn, table_name, columns, batch)
                total += len(batch)
                batch = []

        if batch:
            _insert_batch(conn, table_name, columns, batch)
            total += len(batch)

        if verbose:
            print(f"  [{table_name}] inserted {total} rows.")

    conn.execute("PRAGMA foreign_keys=ON")
