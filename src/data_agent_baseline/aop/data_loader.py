import csv
import json
import sqlite3
from pathlib import Path


def load_records(context_dir: str | Path) -> dict[str, list[dict]]:
    """
    Load all supported files in context_dir into a dict of {key: [records]}.

    CSV  -> {filename: [{"col": val, ...}, ...]}
    JSON -> {filename: [record, ...]}
    SQLite (.db) -> {"filename::tablename": [record, ...]}
    Other extensions (.txt, .md, etc.) are ignored.
    """
    context_dir = Path(context_dir)
    result: dict[str, list[dict]] = {}

    for file_path in sorted(context_dir.iterdir()):
        if not file_path.is_file():
            continue

        suffix = file_path.suffix.lower()

        if suffix == ".csv":
            result[file_path.name] = _load_csv(file_path)
        elif suffix == ".json":
            result[file_path.name] = _load_json(file_path)
        elif suffix == ".db":
            result.update(_load_sqlite(file_path))

    return result


def _load_csv(file_path: Path) -> list[dict]:
    with open(file_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return [dict(row) for row in reader]


def _load_json(file_path: Path) -> list[dict]:
    with open(file_path, encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        if "records" in data:
            return data["records"]
        return [data]
    return [data]


def _load_sqlite(file_path: Path) -> dict[str, list[dict]]:
    result: dict[str, list[dict]] = {}
    conn = sqlite3.connect(file_path)
    conn.row_factory = sqlite3.Row
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
        tables = [row[0] for row in cursor.fetchall()]
        for table in tables:
            cursor.execute(f"SELECT * FROM {table}")  # noqa: S608
            rows = cursor.fetchall()
            key = f"{file_path.name}::{table}"
            result[key] = [dict(row) for row in rows]
    finally:
        conn.close()
    return result
