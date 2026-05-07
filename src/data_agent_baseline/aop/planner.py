"""SemanticPlanner: 자연어 쿼리를 DagNode 실행 계획으로 변환."""

import json
import logging
import re
from pathlib import Path
from typing import Callable

logger = logging.getLogger(__name__)

from .bm25_matcher import BM25Matcher
from .dag_executor import DagNode
from .operators.link import TermContext


PLAN_PROMPT_TEMPLATE = """\
You are a data pipeline planner. Given a question and available tables, \
output a JSON execution plan.
{context}
Available in-memory tables: {tables}
Available SQLite tables: {sqlite_tables}

Question: {query}

Output ONLY a JSON array of steps. Each step has:
- "op": one of "RecordScan", "SqliteFilter", "Join", "Extract", "Count", "GroupBy", "Sum"
- "params": op-specific parameters (see below)
- "input": "prev" (use previous step output) or a table name (load directly)

Op params:
- RecordScan:   {{"table": "<in-memory table name>", "condition": "<natural language condition>"}}
  → Use ONLY for in-memory tables (CSV/JSON)
- SqliteFilter: {{"table": "<sqlite_table_key>", "condition": "<natural language condition>"}}
  → Use ONLY for SQLite table keys (e.g. "results.db::results")
- Join:         {{"left": "prev"|"<table>", "right": "prev"|"<table>", "left_key": "<field>", "right_key": "<field>"}}
- Extract:      {{"columns": ["<col1>", ...]}}
- Count:        {{"field": "<field_name>" or null}}
- GroupBy:      {{"field": "<group_field>", "sum_field": "<sum_field>"}}
- Sum:          {{"field": "<field>"}}

Example for "For patients with severe thrombosis, list their ID, sex and disease":
[
  {{"op": "RecordScan", "params": {{"table": "Examination.json", "condition": "Thrombosis equals 2"}}}},
  {{"op": "Join", "params": {{"left": "prev", "right": "Patient.json", "left_key": "ID", "right_key": "ID"}}}},
  {{"op": "Extract", "params": {{"columns": ["ID", "SEX", "Diagnosis"]}}}}
]

Example for "Find results where score > 90":
[
  {{"op": "SqliteFilter", "params": {{"table": "results.db::results", "condition": "score greater than 90"}}}}
]"""


class SemanticPlanner:
    """
    LLM으로 쿼리를 단계별 계획으로 분해하고, DagNode 트리를 구성.

    Args:
        lrs: LR 목록 (BQMatcher에 전달)
        embedder: 임베딩 모델 (BQMatcher에 전달)
    """

    def __init__(self, lrs: list[dict], embedder=None) -> None:
        self._matcher = BM25Matcher(lrs)

    def plan(
        self,
        query: str,
        all_records_data: dict[str, list[dict]],
        llm_fn: Callable[[str], str],
        term_context: TermContext | None = None,
        sqlite_sources: dict[str, tuple[Path, str]] | None = None,
    ) -> DagNode:
        """
        쿼리를 분석해 실행 가능한 DagNode 트리 반환.

        Raises:
            ValueError: LLM 응답이 파싱 불가하거나 steps가 비어있을 때
        """
        sqlite_sources = sqlite_sources or {}
        tables = list(all_records_data.keys())
        prompt = self._build_prompt(query, tables, term_context, sqlite_sources)
        logger.info("[PLANNER] prompt (%d chars):\n%s", len(prompt), prompt)
        response = llm_fn(prompt)
        logger.info("[PLANNER] llm_response:\n%s", response)
        steps = self._parse_plan(response)
        logger.info("[PLANNER] parsed steps: %s", [s["op"] for s in steps])
        return self._steps_to_dag(steps, all_records_data, sqlite_sources)

    def _build_prompt(
        self,
        query: str,
        tables: list[str],
        term_context: TermContext | None = None,
        sqlite_sources: dict[str, tuple[Path, str]] | None = None,
    ) -> str:
        """LLM에게 전달할 계획 요청 프롬프트 생성."""
        if term_context:
            ctx_str = term_context.as_context_string()
            context_block = f"\n{ctx_str}\n" if ctx_str else ""
        else:
            context_block = ""
        sqlite_sources = sqlite_sources or {}
        sqlite_tables_str = ", ".join(sqlite_sources.keys()) if sqlite_sources else "(none)"
        sqlite_col_lines = _sqlite_column_lines(sqlite_sources)
        if sqlite_col_lines:
            context_block += sqlite_col_lines
        return PLAN_PROMPT_TEMPLATE.format(
            context=context_block,
            tables=", ".join(tables) if tables else "(none)",
            sqlite_tables=sqlite_tables_str,
            query=query,
        )

    def _parse_plan(self, response: str) -> list[dict]:
        """LLM 응답(JSON 배열)을 파싱. 마크다운 코드블록 있으면 제거."""
        cleaned = re.sub(r"```(?:json)?\s*", "", response).strip()
        cleaned = cleaned.rstrip("`").strip()

        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError as e:
            raise ValueError(f"LLM 응답을 parse할 수 없습니다: {e}") from e

        if not isinstance(parsed, list):
            raise ValueError(
                f"LLM 응답이 JSON 배열이어야 합니다. 받은 타입: {type(parsed).__name__}"
            )

        return parsed

    def _steps_to_dag(
        self,
        steps: list[dict],
        all_records_data: dict[str, list[dict]],
        sqlite_sources: dict[str, tuple[Path, str]],
    ) -> DagNode:
        """
        순차 steps 리스트를 DagNode 트리로 변환.
        각 step은 이전 step의 DagNode를 children으로 받음.
        """
        prev_node: DagNode | None = None

        for step in steps:
            op = step["op"]
            params = step["params"]

            if op == "RecordScan":
                table = params["table"]
                records = self._resolve_records(table, all_records_data)
                node = DagNode(
                    op_type="RecordScan",
                    params={"records": records, "condition": params["condition"]},
                )

            elif op == "SqliteFilter":
                table_key = params["table"]
                if table_key not in sqlite_sources:
                    raise ValueError(
                        f"Unknown SQLite source: {table_key!r}. Available: {list(sqlite_sources.keys())}"
                    )
                db_path, table_name = sqlite_sources[table_key]
                node = DagNode(
                    op_type="SqliteFilter",
                    params={"db_path": db_path, "table": table_name, "condition": params["condition"]},
                )

            elif op == "Join":
                left_src = params.get("left", "prev")
                right_src = params.get("right", "")

                if left_src == "prev" and prev_node is not None:
                    left_node = prev_node
                else:
                    left_node = self._make_source_node(left_src, all_records_data, sqlite_sources)

                right_node = self._make_source_node(right_src, all_records_data, sqlite_sources)

                node = DagNode(
                    op_type="Join",
                    params={"left_key": params["left_key"], "right_key": params["right_key"]},
                    children=[left_node, right_node],
                )

            elif op == "Extract":
                node = DagNode(
                    op_type="Extract",
                    params={"columns": params["columns"]},
                    children=[prev_node] if prev_node else [],
                )

            elif op == "Count":
                node = DagNode(
                    op_type="Count",
                    params={"field": params.get("field")},
                    children=[prev_node] if prev_node else [],
                )

            elif op == "GroupBy":
                groupby_node = DagNode(
                    op_type="GroupBy",
                    params={"field": params["field"]},
                    children=[prev_node] if prev_node else [],
                )
                node = DagNode(
                    op_type="Sum",
                    params={"field": params["sum_field"]},
                    children=[groupby_node],
                )

            elif op == "Sum":
                node = DagNode(
                    op_type="Sum",
                    params={"field": params["field"]},
                    children=[prev_node] if prev_node else [],
                )

            else:
                raise ValueError(f"Unknown op: {op!r}")

            prev_node = node

        if prev_node is None:
            raise ValueError("steps 리스트가 비어있음")

        return prev_node

    def _make_source_node(
        self,
        src: str,
        all_records_data: dict[str, list[dict]],
        sqlite_sources: dict[str, tuple[Path, str]],
    ) -> DagNode:
        """테이블 소스에 따라 RecordScan 또는 SqliteFilter 노드 생성."""
        if src in sqlite_sources:
            db_path, table_name = sqlite_sources[src]
            return DagNode(
                op_type="SqliteFilter",
                params={"db_path": db_path, "table": table_name, "condition": ""},
            )
        # LLM이 "Patient.json" 처럼 파일명만 쓴 경우, "Patient.json::*" 키로 fallback
        src_prefix = src.split("::")[0]  # already-qualified keys pass through
        for key, (db_path, table_name) in sqlite_sources.items():
            if key.startswith(src_prefix + "::") or key.startswith(src + "::"):
                return DagNode(
                    op_type="SqliteFilter",
                    params={"db_path": db_path, "table": table_name, "condition": ""},
                )
        records = self._resolve_records(src, all_records_data)
        return DagNode("RecordScan", {"records": records, "condition": ""})

    def _resolve_records(
        self, source: str, all_records_data: dict[str, list[dict]]
    ) -> list[dict]:
        """
        테이블명으로 레코드 조회. 없으면 빈 리스트.

        Exact match를 먼저 시도하고, 실패하면 확장자 추가/제거 후 재시도.
        예: "Examination" → "Examination.json" 매칭
        """
        if source in all_records_data:
            return all_records_data[source]

        with_json = source + ".json"
        if with_json in all_records_data:
            return all_records_data[with_json]

        stem = source.rsplit(".", 1)[0] if "." in source else source
        if stem in all_records_data:
            return all_records_data[stem]

        return []


def _sqlite_column_lines(sqlite_sources: dict[str, tuple[Path, str]]) -> str:
    """SQLite 소스의 컬럼명을 쿼리해 Available Columns 형식 문자열로 반환."""
    import sqlite3
    lines: list[str] = []
    for key, (db_path, table_name) in sqlite_sources.items():
        try:
            with sqlite3.connect(db_path) as conn:
                cursor = conn.execute(f'PRAGMA table_info("{table_name}")')
                cols = [row[1] for row in cursor.fetchall()]
            if cols:
                lines.append(f"  {key}: {', '.join(cols)}")
        except Exception:
            pass
    if not lines:
        return ""
    return "\n=== SQLite Table Columns ===\n" + "\n".join(lines) + "\n"
