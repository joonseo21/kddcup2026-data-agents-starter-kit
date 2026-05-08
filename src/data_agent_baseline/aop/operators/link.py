# aop/operators/link.py
from __future__ import annotations
import csv
import json
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class TermContext:
    knowledge: str  # raw content of knowledge.md (empty string if absent)
    columns: dict[str, list[str]] = field(default_factory=dict)  # filename -> column names

    def as_context_string(self) -> str:
        """Format as a plain-text block for LLM prompt injection."""
        parts = []
        if self.knowledge:
            parts.append(f"=== Knowledge ===\n{self.knowledge.strip()}")
        if self.columns:
            col_lines = "\n".join(
                f"  {fname}: {', '.join(cols)}"
                for fname, cols in self.columns.items()
            )
            parts.append(f"=== Available Columns ===\n{col_lines}")
        return "\n\n".join(parts)


class LinkOperator:
    """
    Reads knowledge.md and data-file headers from a task directory.
    Returns a TermContext for use as Planner context before DAG construction.
    """

    def __init__(self, task_dir: str | Path) -> None:
        self.task_dir = Path(task_dir)

    def execute(self, query: str = "") -> TermContext:
        # knowledge = (
        #     self._read_knowledge_filtered(query)
        #     if query
        #     else self._read_knowledge()
        # )
        knowledge = self._read_knowledge()
        return TermContext(
            knowledge=knowledge,
            columns=self._extract_columns(),
        )

    def _read_knowledge_filtered(self, query: str, top_k: int = 5) -> str:
        """Return only the most query-relevant chunks from knowledge.md."""
        from data_agent_baseline.tools.retrieve import keyword_score
        from data_agent_baseline.tools.text_utils import split_text_chunks

        raw = self._read_knowledge()
        if not raw:
            return ""

        chunks = split_text_chunks(raw)
        if not chunks:
            return raw

        scored = sorted(
            ((keyword_score(query, chunk), chunk) for chunk in chunks),
            key=lambda x: x[0],
            reverse=True,
        )
        relevant = [chunk for score, chunk in scored[:top_k] if score > 0]
        return "\n\n".join(relevant) if relevant else raw

    def _read_knowledge(self) -> str:
        for candidate in [
            self.task_dir / "context" / "knowledge.md",
            self.task_dir / "knowledge.md",
        ]:
            if candidate.exists():
                return candidate.read_text(encoding="utf-8")
        return ""

    def _extract_columns(self) -> dict[str, list[str]]:
        columns: dict[str, list[str]] = {}
        self._read_csv_columns(columns)
        self._read_json_columns(columns)
        self._read_sqlite_columns(columns)
        return columns

    def _read_csv_columns(self, out: dict[str, list[str]]) -> None:
        csv_dir = self.task_dir / "context" / "csv"
        if not csv_dir.is_dir():
            return
        for f in sorted(csv_dir.glob("*.csv")):
            with f.open(newline="", encoding="utf-8") as fh:
                reader = csv.reader(fh)
                try:
                    out[f.name] = next(reader)
                except StopIteration:
                    out[f.name] = []

    def _read_json_columns(self, out: dict[str, list[str]]) -> None:
        json_dir = self.task_dir / "context" / "json"
        if not json_dir.is_dir():
            return
        for f in sorted(json_dir.glob("*.json")):
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
                if isinstance(data, list) and data and isinstance(data[0], dict):
                    out[f.name] = list(data[0].keys())
            except (json.JSONDecodeError, OSError):
                pass

    def _read_sqlite_columns(self, out: dict[str, list[str]]) -> None:
        db_dir = self.task_dir / "context" / "db"
        if not db_dir.is_dir():
            return
        for f in sorted(db_dir.glob("*.db")):
            try:
                con = sqlite3.connect(f)
                cur = con.cursor()
                tables = [
                    row[0]
                    for row in cur.execute(
                        "SELECT name FROM sqlite_master WHERE type='table'"
                    ).fetchall()
                ]
                for table in tables:
                    cols = [
                        row[1]
                        for row in cur.execute(
                            f"PRAGMA table_info({table})"
                        ).fetchall()
                    ]
                    key = f"{f.name}::{table}"
                    out[key] = cols
                con.close()
            except sqlite3.Error:
                pass
