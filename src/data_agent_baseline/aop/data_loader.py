import csv
import json
import sqlite3
from pathlib import Path


def load_records(context_dir: str | Path) -> dict[str, list[dict]]:
    """
    Load CSV and JSON files in context_dir into a dict of {key: [records]}.

    CSV  -> {filename: [{"col": val, ...}, ...]}
    JSON -> {filename: [record, ...]}
    SQLite (.db) files are intentionally skipped — use load_sqlite_sources() instead.
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

    return result


def load_sqlite_sources(context_dir: str | Path) -> dict[str, tuple[Path, str]]:
    """
    Return metadata for all SQLite tables in context_dir without loading data.

    Returns {"filename.db::tablename": (Path, "tablename"), ...}
    Use SqliteFilterOp to query these tables at execution time.
    """
    context_dir = Path(context_dir)
    result: dict[str, tuple[Path, str]] = {}

    for file_path in sorted(context_dir.iterdir()):
        if file_path.is_file() and file_path.suffix.lower() == ".db":
            result.update(_collect_sqlite_sources(file_path))

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


def _collect_sqlite_sources(file_path: Path) -> dict[str, tuple[Path, str]]:
    result: dict[str, tuple[Path, str]] = {}
    with sqlite3.connect(file_path) as conn:
        cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
        for (table,) in cursor.fetchall():
            key = f"{file_path.name}::{table}"
            result[key] = (file_path, table)
    return result
