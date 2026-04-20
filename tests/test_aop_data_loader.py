"""
data_loader.load_records() unit tests.

Each file format (CSV, JSON, SQLite) is covered by a dedicated test class.
"""

import csv
import json
import sqlite3
from pathlib import Path

import pytest

from data_agent_baseline.aop.data_loader import load_records


class TestCsvLoading:
    def test_basic_csv_returns_list_of_dicts(self, tmp_path):
        f = tmp_path / "patients.csv"
        f.write_text("ID,SEX\n1,F\n2,M\n")

        result = load_records(tmp_path)

        assert "patients.csv" in result
        assert result["patients.csv"] == [{"ID": "1", "SEX": "F"}, {"ID": "2", "SEX": "M"}]

    def test_multiple_csv_files_all_loaded(self, tmp_path):
        (tmp_path / "a.csv").write_text("X\n1\n")
        (tmp_path / "b.csv").write_text("Y\n2\n")

        result = load_records(tmp_path)

        assert "a.csv" in result
        assert "b.csv" in result

    def test_empty_csv_returns_empty_list(self, tmp_path):
        (tmp_path / "empty.csv").write_text("ID,SEX\n")

        result = load_records(tmp_path)

        assert result["empty.csv"] == []


class TestJsonLoading:
    def test_json_array_loaded_directly(self, tmp_path):
        data = [{"id": 1}, {"id": 2}]
        (tmp_path / "data.json").write_text(json.dumps(data))

        result = load_records(tmp_path)

        assert result["data.json"] == data

    def test_json_dict_with_records_key(self, tmp_path):
        rows = [{"id": 1}, {"id": 2}]
        (tmp_path / "data.json").write_text(json.dumps({"records": rows, "meta": "ignored"}))

        result = load_records(tmp_path)

        assert result["data.json"] == rows

    def test_json_plain_dict_wrapped_in_list(self, tmp_path):
        obj = {"key": "value"}
        (tmp_path / "data.json").write_text(json.dumps(obj))

        result = load_records(tmp_path)

        assert result["data.json"] == [obj]


class TestSqliteLoading:
    def _make_db(self, path: Path, tables: dict[str, list[dict]]) -> None:
        conn = sqlite3.connect(path)
        for table_name, rows in tables.items():
            if not rows:
                continue
            cols = list(rows[0].keys())
            col_defs = ", ".join(f"{c} TEXT" for c in cols)
            conn.execute(f"CREATE TABLE {table_name} ({col_defs})")
            placeholders = ", ".join("?" for _ in cols)
            conn.executemany(
                f"INSERT INTO {table_name} VALUES ({placeholders})",
                [[r[c] for c in cols] for r in rows],
            )
        conn.commit()
        conn.close()

    def test_single_table_loaded(self, tmp_path):
        db_path = tmp_path / "test.db"
        self._make_db(db_path, {"patient": [{"ID": "1", "SEX": "F"}]})

        result = load_records(tmp_path)

        assert "test.db::patient" in result
        assert result["test.db::patient"] == [{"ID": "1", "SEX": "F"}]

    def test_multiple_tables_keyed_by_filename_table(self, tmp_path):
        db_path = tmp_path / "multi.db"
        self._make_db(
            db_path,
            {
                "patient": [{"ID": "1"}],
                "examination": [{"ID": "1", "Thrombosis": "2"}],
            },
        )

        result = load_records(tmp_path)

        assert "multi.db::patient" in result
        assert "multi.db::examination" in result


class TestIgnoredFiles:
    def test_txt_and_md_files_ignored(self, tmp_path):
        (tmp_path / "notes.txt").write_text("ignore me")
        (tmp_path / "readme.md").write_text("# ignore")

        result = load_records(tmp_path)

        assert result == {}

    def test_empty_directory_returns_empty_dict(self, tmp_path):
        assert load_records(tmp_path) == {}
