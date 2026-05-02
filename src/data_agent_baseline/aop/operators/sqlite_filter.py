import sqlite3
from pathlib import Path
from typing import Callable

from data_agent_baseline.tools.sqlite import execute_read_only_sql

_RESULT_LIMIT = 50_000


class SqliteFilterOp:
    """
    Filter records from a SQLite table using an LLM-generated WHERE clause.

    Avoids loading the full table into memory — executes SQL directly.
    Same execute(llm_fn) -> list[dict] interface as RecordScanOp.
    """

    def __init__(self, db_path: Path, table: str, condition: str) -> None:
        self.db_path = db_path
        self.table = table
        self.condition = condition

    def execute(self, llm_fn: Callable[[str], str]) -> list[dict]:
        where_clause = ""
        if self.condition:
            columns = self._get_columns()
            prompt = (
                f'Generate a SQL WHERE clause for the condition: "{self.condition}"\n'
                f'Table "{self.table}" columns: {columns}\n'
                f'Return ONLY the WHERE clause expression (no "WHERE" keyword).\n'
                f"Example: sex = 'F' AND score > 90"
            )
            where_clause = llm_fn(prompt).strip().strip("`").strip()

        sql = f'SELECT * FROM "{self.table}"'
        if where_clause:
            sql += f" WHERE {where_clause}"

        try:
            result = execute_read_only_sql(self.db_path, sql, limit=_RESULT_LIMIT)
        except Exception as e:
            raise ValueError(f"SQL execution failed for table '{self.table}': {e}") from e

        cols = result["columns"]
        return [dict(zip(cols, row)) for row in result["rows"]]

    def _get_columns(self) -> list[str]:
        uri = f"file:{self.db_path.resolve().as_posix()}?mode=ro"
        with sqlite3.connect(uri, uri=True) as conn:
            cursor = conn.execute(f'PRAGMA table_info("{self.table}")')
            return [row[1] for row in cursor.fetchall()]
