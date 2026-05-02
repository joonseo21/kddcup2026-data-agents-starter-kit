"""AOPAgent 단위 테스트. ScriptedModelAdapter로 LLM 없이 검증."""
import json
import sqlite3
from pathlib import Path

import pytest

from data_agent_baseline.agents.aop_agent import AOPAgent, AOPAgentConfig, _result_to_answer_table
from data_agent_baseline.agents.model import ScriptedModelAdapter
from data_agent_baseline.agents.runtime import AgentRunResult
from data_agent_baseline.benchmark.schema import AnswerTable
from data_agent_baseline.aop.logical_representations import (
    RECORD_FILTER_LR, JOIN_SIMPLE_LR, EXTRACT_LR, COUNT_LR, GROUPBY_SUM_LR, SQLITE_FILTER_LR,
)

ALL_LRS = [RECORD_FILTER_LR, JOIN_SIMPLE_LR, EXTRACT_LR, COUNT_LR, GROUPBY_SUM_LR, SQLITE_FILTER_LR]


# ──────────────────────────────────────────
# Helpers / Fixtures
# ──────────────────────────────────────────

def make_task(tmp_path, question: str, csv_data: list[dict] | None = None):
    """Minimal PublicTask-like object pointing at tmp_path."""
    import csv as csv_mod
    from pathlib import Path
    from unittest.mock import MagicMock

    ctx = tmp_path / "context"
    csv_dir = ctx / "csv"
    csv_dir.mkdir(parents=True)

    if csv_data:
        with (csv_dir / "data.csv").open("w", newline="") as f:
            writer = csv_mod.DictWriter(f, fieldnames=list(csv_data[0].keys()))
            writer.writeheader()
            writer.writerows(csv_data)

    task = MagicMock()
    task.task_id = "task_test"
    task.question = question
    task.task_dir = tmp_path
    return task


SAMPLE_RECORDS = [
    {"ID": 1, "Name": "Alice", "Score": 90},
    {"ID": 2, "Name": "Bob",   "Score": 70},
    {"ID": 3, "Name": "Carol", "Score": 85},
]

PLAN_EXTRACT_ALL = json.dumps([
    {"op": "RecordScan", "params": {"table": "data.csv", "condition": ""}},
    {"op": "Extract",    "params": {"columns": ["ID", "Name", "Score"]}},
])

PLAN_COUNT = json.dumps([
    {"op": "RecordScan", "params": {"table": "data.csv", "condition": ""}},
    {"op": "Count",      "params": {"field": None}},
])


# ──────────────────────────────────────────
# _result_to_answer_table
# ──────────────────────────────────────────

class TestResultToAnswerTable:
    def test_list_of_dicts(self):
        records = [{"a": 1, "b": 2}, {"a": 3, "b": 4}]
        ans = _result_to_answer_table(records)
        assert isinstance(ans, AnswerTable)
        assert ans.columns == ["a", "b"]
        assert ans.rows == [[1, 2], [3, 4]]

    def test_int_count(self):
        ans = _result_to_answer_table(42)
        assert isinstance(ans, AnswerTable)
        assert ans.columns == ["count"]
        assert ans.rows == [[42]]

    def test_float_count(self):
        ans = _result_to_answer_table(3.14)
        assert isinstance(ans, AnswerTable)
        assert ans.rows[0][0] == 3.14

    def test_empty_list_returns_none(self):
        assert _result_to_answer_table([]) is None

    def test_none_returns_none(self):
        assert _result_to_answer_table(None) is None

    def test_string_returns_none(self):
        assert _result_to_answer_table("unexpected") is None


# ──────────────────────────────────────────
# AOPAgent.run — successful extract pipeline
# ──────────────────────────────────────────

class TestAOPAgentExtract:
    @pytest.fixture
    def agent_and_task(self, tmp_path):
        model = ScriptedModelAdapter(responses=[
            PLAN_EXTRACT_ALL,   # planner call
        ])
        agent = AOPAgent(model=model, lrs=ALL_LRS, config=AOPAgentConfig(max_steps=16))
        task = make_task(tmp_path, "List all records", SAMPLE_RECORDS)
        return agent, task

    def test_run_returns_agent_run_result(self, agent_and_task):
        agent, task = agent_and_task
        result = agent.run(task)
        assert isinstance(result, AgentRunResult)

    def test_run_succeeds(self, agent_and_task):
        agent, task = agent_and_task
        result = agent.run(task)
        assert result.succeeded

    def test_run_answer_columns(self, agent_and_task):
        agent, task = agent_and_task
        result = agent.run(task)
        assert result.answer is not None
        assert set(result.answer.columns) == {"ID", "Name", "Score"}

    def test_run_answer_row_count(self, agent_and_task):
        agent, task = agent_and_task
        result = agent.run(task)
        assert len(result.answer.rows) == 3

    def test_run_steps_recorded(self, agent_and_task):
        agent, task = agent_and_task
        result = agent.run(task)
        assert len(result.steps) >= 2


# ──────────────────────────────────────────
# AOPAgent.run — count pipeline
# ──────────────────────────────────────────

class TestAOPAgentCount:
    def test_count_returns_answer_table(self, tmp_path):
        model = ScriptedModelAdapter(responses=[PLAN_COUNT])
        agent = AOPAgent(model=model, lrs=ALL_LRS)
        task = make_task(tmp_path, "How many records?", SAMPLE_RECORDS)
        result = agent.run(task)
        assert result.succeeded
        assert result.answer.columns == ["count"]
        assert result.answer.rows == [[3]]


# ──────────────────────────────────────────
# AOPAgent.run — failure handling
# ──────────────────────────────────────────

class TestAOPAgentFailure:
    def test_bad_plan_returns_failed_result(self, tmp_path):
        model = ScriptedModelAdapter(responses=["not valid json at all"])
        agent = AOPAgent(model=model, lrs=ALL_LRS)
        task = make_task(tmp_path, "Any question", SAMPLE_RECORDS)
        result = agent.run(task)
        assert not result.succeeded
        assert result.failure_reason is not None
        assert result.answer is None

    def test_task_id_preserved_on_failure(self, tmp_path):
        model = ScriptedModelAdapter(responses=["bad json"])
        agent = AOPAgent(model=model, lrs=ALL_LRS)
        task = make_task(tmp_path, "Any question")
        result = agent.run(task)
        assert result.task_id == "task_test"


# ──────────────────────────────────────────
# AOPAgent.run — SQLite source pipeline
# ──────────────────────────────────────────

def _make_sqlite_task(tmp_path, question: str, rows: list[dict]):
    """Task with a SQLite file in context/db/."""
    from unittest.mock import MagicMock
    db_dir = tmp_path / "context" / "db"
    db_dir.mkdir(parents=True)
    db_path = db_dir / "results.db"
    cols = list(rows[0].keys())
    col_defs = ", ".join(f"{c} TEXT" for c in cols)
    placeholders = ", ".join("?" for _ in cols)
    with sqlite3.connect(db_path) as conn:
        conn.execute(f"CREATE TABLE results ({col_defs})")
        conn.executemany(
            f"INSERT INTO results VALUES ({placeholders})",
            [[r[c] for c in cols] for r in rows],
        )
        conn.commit()

    task = MagicMock()
    task.task_id = "task_sqlite_test"
    task.question = question
    task.task_dir = tmp_path
    return task


class TestAOPAgentSqlite:
    def test_sqlite_task_succeeds(self, tmp_path):
        rows = [{"id": str(i), "score": str(i * 10)} for i in range(1, 6)]
        task = _make_sqlite_task(tmp_path, "Find records with high score", rows)

        # Scripted responses: 1 = planner (SqliteFilter plan), 2 = WHERE clause from SqliteFilterOp
        plan = json.dumps([
            {"op": "SqliteFilter", "params": {"table": "results.db::results", "condition": "score > 30"}},
        ])
        model = ScriptedModelAdapter(responses=[plan, "CAST(score AS INTEGER) > 30"])
        agent = AOPAgent(model=model, lrs=ALL_LRS)

        result = agent.run(task)

        assert result.succeeded
        assert result.answer is not None
        assert len(result.answer.rows) == 2  # score 40, 50

    def test_sqlite_sources_logged(self, tmp_path, caplog):
        import logging
        rows = [{"id": "1", "val": "A"}]
        task = _make_sqlite_task(tmp_path, "list all", rows)

        plan = json.dumps([
            {"op": "SqliteFilter", "params": {"table": "results.db::results", "condition": ""}},
        ])
        model = ScriptedModelAdapter(responses=[plan, ""])
        agent = AOPAgent(model=model, lrs=ALL_LRS)

        with caplog.at_level(logging.INFO):
            agent.run(task)

        assert "results.db::results" in caplog.text
