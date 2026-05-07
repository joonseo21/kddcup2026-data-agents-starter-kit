"""
SemanticPlanner 단위 테스트.

mock_llm_fn을 주입해 실제 LLM API 호출 없이 플래닝 로직 검증.
MockEmbedder로 BQMatcher 의존성도 주입.
"""

import json
import sqlite3
from pathlib import Path

import pytest

from data_agent_baseline.aop.planner import SemanticPlanner
from data_agent_baseline.aop.dag_executor import DagNode, DagExecutor
from data_agent_baseline.aop.embedding import MockEmbedder
from data_agent_baseline.aop.operators.link import TermContext
from data_agent_baseline.aop.logical_representations import (
    RECORD_FILTER_LR, JOIN_SIMPLE_LR, EXTRACT_LR, COUNT_LR, GROUPBY_SUM_LR,
)

ALL_LRS = [RECORD_FILTER_LR, JOIN_SIMPLE_LR, EXTRACT_LR, COUNT_LR, GROUPBY_SUM_LR]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def planner():
    return SemanticPlanner(lrs=ALL_LRS, embedder=MockEmbedder())


@pytest.fixture
def sample_records():
    examination = [
        {"ID": 1, "Thrombosis": 2, "Diagnosis": "SLE"},
        {"ID": 2, "Thrombosis": 0, "Diagnosis": "APS"},
        {"ID": 3, "Thrombosis": 2, "Diagnosis": "PSS"},
    ]
    patient = [
        {"ID": 1, "SEX": "F"},
        {"ID": 2, "SEX": "M"},
        {"ID": 3, "SEX": "F"},
    ]
    return {
        "Examination.json": examination,
        "Patient.json": patient,
    }


def make_llm(plan: list[dict]):
    """지정된 plan JSON을 반환하는 mock LLM 함수 생성기."""
    def llm_fn(prompt: str) -> str:
        return json.dumps(plan)
    return llm_fn


# ---------------------------------------------------------------------------
# 프롬프트 생성
# ---------------------------------------------------------------------------

class TestBuildPrompt:
    def test_prompt_contains_query(self, planner, sample_records):
        prompt = planner._build_prompt("list patients", list(sample_records.keys()))
        assert "list patients" in prompt

    def test_prompt_contains_table_names(self, planner, sample_records):
        prompt = planner._build_prompt("list patients", list(sample_records.keys()))
        assert "Examination.json" in prompt
        assert "Patient.json" in prompt

    def test_prompt_includes_knowledge_when_term_context_given(self, planner, sample_records):
        ctx = TermContext(knowledge="Thrombosis: 1=mild, 2=severe", columns={})
        prompt = planner._build_prompt("list patients", list(sample_records.keys()), ctx)
        assert "Thrombosis" in prompt
        assert "Knowledge" in prompt

    def test_prompt_includes_columns_when_term_context_given(self, planner, sample_records):
        ctx = TermContext(knowledge="", columns={"patients.csv": ["ID", "SEX"]})
        prompt = planner._build_prompt("list patients", list(sample_records.keys()), ctx)
        assert "patients.csv" in prompt
        assert "ID" in prompt

    def test_prompt_no_context_section_when_term_context_none(self, planner, sample_records):
        prompt = planner._build_prompt("list patients", list(sample_records.keys()), None)
        assert "Knowledge" not in prompt
        assert "Available Columns" not in prompt

    def test_plan_passes_term_context_to_prompt(self, planner, sample_records):
        ctx = TermContext(knowledge="domain knowledge here", columns={})
        captured = []

        def capturing_llm(prompt):
            captured.append(prompt)
            return json.dumps([
                {"op": "RecordScan", "params": {"table": "Examination.json", "condition": ""}},
            ])

        planner.plan("list all", sample_records, capturing_llm, term_context=ctx)
        assert "domain knowledge here" in captured[0]


# ---------------------------------------------------------------------------
# LLM 응답 파싱
# ---------------------------------------------------------------------------

class TestParsePlan:
    def test_parses_plain_json_array(self, planner):
        raw = json.dumps([{"op": "Count", "params": {"field": None}}])
        steps = planner._parse_plan(raw)
        assert len(steps) == 1
        assert steps[0]["op"] == "Count"

    def test_strips_markdown_code_block(self, planner):
        raw = "```json\n[{\"op\": \"Extract\", \"params\": {\"columns\": [\"ID\"]}}]\n```"
        steps = planner._parse_plan(raw)
        assert steps[0]["op"] == "Extract"

    def test_invalid_json_raises_value_error(self, planner):
        with pytest.raises(ValueError, match="parse"):
            planner._parse_plan("not valid json at all")

    def test_non_list_response_raises_value_error(self, planner):
        with pytest.raises(ValueError):
            planner._parse_plan(json.dumps({"op": "Count"}))


# ---------------------------------------------------------------------------
# DAG 빌드 (단순 케이스)
# ---------------------------------------------------------------------------

class TestStepsToDag:
    def test_single_count_step(self, planner, sample_records):
        scan_steps = [
            {"op": "RecordScan", "params": {"table": "Examination.json", "condition": "Thrombosis equals 2"}},
            {"op": "Count", "params": {"field": None}},
        ]
        node = planner._steps_to_dag(scan_steps, sample_records, {})
        assert node.op_type == "Count"
        assert node.children[0].op_type == "RecordScan"

    def test_extract_chained_after_scan(self, planner, sample_records):
        steps = [
            {"op": "RecordScan", "params": {"table": "Examination.json", "condition": "Thrombosis equals 2"}},
            {"op": "Extract", "params": {"columns": ["ID", "Diagnosis"]}},
        ]
        node = planner._steps_to_dag(steps, sample_records, {})
        assert node.op_type == "Extract"
        assert node.params["columns"] == ["ID", "Diagnosis"]
        assert node.children[0].op_type == "RecordScan"

    def test_empty_steps_raises_value_error(self, planner, sample_records):
        with pytest.raises(ValueError, match="비어있음"):
            planner._steps_to_dag([], sample_records, {})


# ---------------------------------------------------------------------------
# plan() — Task 11 전체 파이프라인
# ---------------------------------------------------------------------------

class TestPlanTask11:
    """
    Task 11: 혈전증 심한 환자의 ID, SEX, Diagnosis 조회
    RecordScan → Join → Extract 파이프라인
    """

    @pytest.fixture
    def task11_llm(self):
        plan = [
            {"op": "RecordScan", "params": {"table": "Examination.json", "condition": "Thrombosis equals 2"}},
            {"op": "Join",       "params": {"left": "prev", "right": "Patient.json", "left_key": "ID", "right_key": "ID"}},
            {"op": "Extract",    "params": {"columns": ["ID", "SEX", "Diagnosis"]}},
        ]
        return make_llm(plan)

    def test_plan_returns_dag_node(self, planner, sample_records, task11_llm):
        node = planner.plan("For patients with severe thrombosis, list ID sex and disease",
                            sample_records, task11_llm)
        assert isinstance(node, DagNode)

    def test_plan_root_is_extract(self, planner, sample_records, task11_llm):
        node = planner.plan("...", sample_records, task11_llm)
        assert node.op_type == "Extract"
        assert node.params["columns"] == ["ID", "SEX", "Diagnosis"]

    def test_plan_has_join_child(self, planner, sample_records, task11_llm):
        node = planner.plan("...", sample_records, task11_llm)
        join_node = node.children[0]
        assert join_node.op_type == "Join"
        assert join_node.params["left_key"] == "ID"

    def test_plan_join_has_two_children(self, planner, sample_records, task11_llm):
        node = planner.plan("...", sample_records, task11_llm)
        join_node = node.children[0]
        assert len(join_node.children) == 2

    def test_full_pipeline_executes_correctly(self, planner, sample_records, task11_llm):
        """DagExecutor로 실제 실행해서 결과 검증."""
        def mock_record_scan_llm(prompt):
            return json.dumps({"field": "Thrombosis", "op": "=", "value": "2"})

        node = planner.plan("...", sample_records, task11_llm)
        executor = DagExecutor(llm_fn=mock_record_scan_llm)
        result = executor.execute(node)

        # Thrombosis=2인 ID 1, 3만 남아야 함
        assert isinstance(result, list)
        ids = {r["ID"] for r in result}
        assert ids == {1, 3}
        # Extract로 컬럼 제한됨
        for row in result:
            assert set(row.keys()) == {"ID", "SEX", "Diagnosis"}


# ---------------------------------------------------------------------------
# plan() — 단순 Count 쿼리
# ---------------------------------------------------------------------------

class TestPlanCount:
    def test_count_pipeline(self, planner, sample_records):
        plan = [
            {"op": "RecordScan", "params": {"table": "Examination.json", "condition": "Thrombosis equals 2"}},
            {"op": "Count", "params": {"field": None}},
        ]
        def mock_record_scan_llm(prompt):
            return json.dumps({"field": "Thrombosis", "op": "=", "value": "2"})

        node = planner.plan("...", sample_records, make_llm(plan))
        executor = DagExecutor(llm_fn=mock_record_scan_llm)
        result = executor.execute(node)
        assert result == 2  # Thrombosis=2인 레코드 2개


# ---------------------------------------------------------------------------
# SqliteFilter 계획 생성
# ---------------------------------------------------------------------------

def _make_sqlite_db(path: Path, table: str, rows: list[dict]) -> None:
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


class TestSqliteFilterPlan:
    @pytest.fixture
    def sqlite_sources(self, tmp_path):
        db_path = tmp_path / "results.db"
        _make_sqlite_db(db_path, "results", [
            {"id": "1", "score": "90"},
            {"id": "2", "score": "85"},
        ])
        return {"results.db::results": (db_path, "results")}

    def test_prompt_contains_sqlite_table_key(self, planner, sqlite_sources):
        prompt = planner._build_prompt(
            "find high scores", [], None, sqlite_sources
        )
        assert "results.db::results" in prompt

    def test_steps_to_dag_creates_sqlite_filter_node(self, planner, sqlite_sources):
        steps = [
            {"op": "SqliteFilter", "params": {"table": "results.db::results", "condition": "score > 90"}},
        ]
        node = planner._steps_to_dag(steps, {}, sqlite_sources)

        assert node.op_type == "SqliteFilter"
        assert node.params["table"] == "results"
        assert node.params["condition"] == "score > 90"

    def test_plan_with_sqlite_sources_returns_sqlite_filter_node(self, planner, sqlite_sources):
        plan = [
            {"op": "SqliteFilter", "params": {"table": "results.db::results", "condition": "score greater than 90"}},
        ]
        node = planner.plan("find high scores", {}, make_llm(plan), sqlite_sources=sqlite_sources)
        assert node.op_type == "SqliteFilter"

    def test_unknown_sqlite_key_raises(self, planner):
        steps = [
            {"op": "SqliteFilter", "params": {"table": "nonexistent.db::table", "condition": "x"}},
        ]
        with pytest.raises(ValueError, match="Unknown SQLite source"):
            planner._steps_to_dag(steps, {}, {})
