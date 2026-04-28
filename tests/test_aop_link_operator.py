"""LinkOperator 단위 테스트."""
import csv
import json
import sqlite3

import pytest

from data_agent_baseline.aop.operators.link import LinkOperator, TermContext


# ──────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────

@pytest.fixture
def task_dir(tmp_path):
    """CSV + JSON + knowledge.md가 있는 최소 태스크 디렉터리."""
    ctx = tmp_path / "context"
    (ctx / "csv").mkdir(parents=True)
    (ctx / "json").mkdir()
    (ctx / "db").mkdir()

    # knowledge.md
    (ctx / "knowledge.md").write_text("Thrombosis: 1=mild, 2=severe", encoding="utf-8")

    # CSV file
    with (ctx / "csv" / "patients.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["PatientID", "Thrombosis", "Age"])
        writer.writerow(["001", "2", "45"])

    # JSON file
    data = [{"DrugID": "D1", "Name": "Aspirin", "Dose": 100}]
    (ctx / "json" / "drugs.json").write_text(json.dumps(data), encoding="utf-8")

    # SQLite
    con = sqlite3.connect(ctx / "db" / "clinic.db")
    con.execute("CREATE TABLE visits (id INTEGER, date TEXT, score REAL)")
    con.commit()
    con.close()

    return tmp_path


@pytest.fixture
def op(task_dir):
    return LinkOperator(task_dir)


# ──────────────────────────────────────────
# TermContext
# ──────────────────────────────────────────

class TestTermContext:
    def test_as_context_string_includes_knowledge(self):
        ctx = TermContext(knowledge="Thrombosis: 1=mild", columns={})
        s = ctx.as_context_string()
        assert "Thrombosis" in s
        assert "Knowledge" in s

    def test_as_context_string_includes_columns(self):
        ctx = TermContext(knowledge="", columns={"patients.csv": ["ID", "Age"]})
        s = ctx.as_context_string()
        assert "patients.csv" in s
        assert "ID" in s

    def test_as_context_string_empty_when_no_data(self):
        ctx = TermContext(knowledge="", columns={})
        assert ctx.as_context_string() == ""

    def test_as_context_string_both_sections(self):
        ctx = TermContext(knowledge="KM content", columns={"f.csv": ["a", "b"]})
        s = ctx.as_context_string()
        assert "Knowledge" in s
        assert "Available Columns" in s


# ──────────────────────────────────────────
# LinkOperator.execute()
# ──────────────────────────────────────────

class TestLinkOperatorExecute:
    def test_returns_term_context(self, op):
        result = op.execute()
        assert isinstance(result, TermContext)

    def test_reads_knowledge_md(self, op):
        result = op.execute()
        assert "Thrombosis" in result.knowledge

    def test_reads_csv_columns(self, op):
        result = op.execute()
        assert "patients.csv" in result.columns
        assert result.columns["patients.csv"] == ["PatientID", "Thrombosis", "Age"]

    def test_reads_json_columns(self, op):
        result = op.execute()
        assert "drugs.json" in result.columns
        assert result.columns["drugs.json"] == ["DrugID", "Name", "Dose"]

    def test_reads_sqlite_columns(self, op):
        result = op.execute()
        key = "clinic.db::visits"
        assert key in result.columns
        assert result.columns[key] == ["id", "date", "score"]


# ──────────────────────────────────────────
# knowledge.md path fallback
# ──────────────────────────────────────────

class TestKnowledgeFallback:
    def test_knowledge_at_root_level(self, tmp_path):
        """context/knowledge.md가 없고 task_dir/knowledge.md가 있는 경우."""
        (tmp_path / "knowledge.md").write_text("root-level KM", encoding="utf-8")
        ctx = TermContext(knowledge="", columns={})
        op = LinkOperator(tmp_path)
        result = op.execute()
        assert "root-level KM" in result.knowledge

    def test_context_level_takes_priority(self, tmp_path):
        """둘 다 있으면 context/knowledge.md 우선."""
        (tmp_path / "knowledge.md").write_text("root KM", encoding="utf-8")
        ctx_dir = tmp_path / "context"
        ctx_dir.mkdir()
        (ctx_dir / "knowledge.md").write_text("context KM", encoding="utf-8")
        op = LinkOperator(tmp_path)
        result = op.execute()
        assert "context KM" in result.knowledge

    def test_missing_knowledge_returns_empty_string(self, tmp_path):
        op = LinkOperator(tmp_path)
        result = op.execute()
        assert result.knowledge == ""


# ──────────────────────────────────────────
# Edge cases
# ──────────────────────────────────────────

class TestLinkOperatorEdgeCases:
    def test_empty_task_dir(self, tmp_path):
        op = LinkOperator(tmp_path)
        result = op.execute()
        assert result.knowledge == ""
        assert result.columns == {}

    def test_empty_csv_returns_empty_columns(self, tmp_path):
        ctx = tmp_path / "context" / "csv"
        ctx.mkdir(parents=True)
        (ctx / "empty.csv").write_text("", encoding="utf-8")
        op = LinkOperator(tmp_path)
        result = op.execute()
        assert result.columns.get("empty.csv") == []

    def test_json_non_list_ignored(self, tmp_path):
        ctx = tmp_path / "context" / "json"
        ctx.mkdir(parents=True)
        (ctx / "meta.json").write_text('{"key": "value"}', encoding="utf-8")
        op = LinkOperator(tmp_path)
        result = op.execute()
        assert "meta.json" not in result.columns

    def test_multiple_csv_files(self, tmp_path):
        ctx = tmp_path / "context" / "csv"
        ctx.mkdir(parents=True)
        for name, cols in [("a.csv", ["x", "y"]), ("b.csv", ["p", "q", "r"])]:
            with (ctx / name).open("w", newline="") as f:
                csv.writer(f).writerow(cols)
        result = LinkOperator(tmp_path).execute()
        assert set(result.columns.keys()) == {"a.csv", "b.csv"}
