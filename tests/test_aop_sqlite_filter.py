"""
SqliteFilterOp unit tests.

Uses tmp SQLite fixtures and a mock llm_fn — no real LLM API calls.
"""

import json
import sqlite3
from pathlib import Path

import pytest

from data_agent_baseline.aop.operators.sqlite_filter import SqliteFilterOp


def _make_db(path: Path, table: str, rows: list[dict]) -> None:
    cols = list(rows[0].keys())
    col_defs = ", ".join(f"{c} TEXT" for c in cols)
    placeholders = ", ".join("?" for _ in cols)
    with sqlite3.connect(path) as conn:
        conn.execute(f"CREATE TABLE {table} ({col_defs})")
        conn.executemany(
            f"INSERT INTO {table} VALUES ({placeholders})",
            [[r[c] for c in cols] for r in rows],
        )
        conn.commit()


@pytest.fixture
def sample_db(tmp_path):
    db_path = tmp_path / "results.db"
    _make_db(
        db_path,
        "results",
        [
            {"id": "1", "sex": "F", "score": "90"},
            {"id": "2", "sex": "M", "score": "85"},
            {"id": "3", "sex": "F", "score": "95"},
        ],
    )
    return db_path


class TestSqliteFilterOp:
    def test_basic_filter_returns_matching_rows(self, sample_db):
        def mock_llm(prompt):
            return "sex = 'F'"

        op = SqliteFilterOp(sample_db, "results", "sex is F")
        result = op.execute(mock_llm)

        assert len(result) == 2
        assert all(r["sex"] == "F" for r in result)

    def test_no_condition_returns_all_rows(self, sample_db):
        op = SqliteFilterOp(sample_db, "results", "")
        result = op.execute(lambda p: "")

        assert len(result) == 3

    def test_no_matching_rows_returns_empty(self, sample_db):
        def mock_llm(prompt):
            return "sex = 'X'"

        op = SqliteFilterOp(sample_db, "results", "sex is X")
        result = op.execute(mock_llm)

        assert result == []

    def test_invalid_sql_where_raises_value_error(self, sample_db):
        def mock_llm(prompt):
            return "THIS IS NOT VALID SQL @@@@"

        op = SqliteFilterOp(sample_db, "results", "some condition")
        with pytest.raises(ValueError, match="SQL execution failed"):
            op.execute(mock_llm)

    def test_column_names_appear_in_llm_prompt(self, sample_db):
        captured = {}

        def mock_llm(prompt):
            captured["prompt"] = prompt
            return "score > 90"

        op = SqliteFilterOp(sample_db, "results", "score greater than 90")
        op.execute(mock_llm)

        assert "id" in captured["prompt"] or "score" in captured["prompt"]
        assert "sex" in captured["prompt"]

    def test_returns_list_of_dicts(self, sample_db):
        op = SqliteFilterOp(sample_db, "results", "")
        result = op.execute(lambda p: "")

        assert isinstance(result, list)
        assert all(isinstance(r, dict) for r in result)

    def test_numeric_comparison_filter(self, sample_db):
        def mock_llm(prompt):
            return "CAST(score AS INTEGER) > 90"

        op = SqliteFilterOp(sample_db, "results", "score greater than 90")
        result = op.execute(mock_llm)

        assert len(result) == 1
        assert result[0]["id"] == "3"
