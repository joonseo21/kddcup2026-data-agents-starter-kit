"""SemanticPlanner: 자연어 쿼리를 DagNode 실행 계획으로 변환."""

import json
import logging
import re
from typing import Callable

logger = logging.getLogger(__name__)

from .bm25_matcher import BM25Matcher
from .dag_executor import DagNode
from .operators.link import TermContext


PLAN_PROMPT_TEMPLATE = """\
You are a data pipeline planner. Given a question and available tables, \
output a JSON execution plan.
{context}
Available tables: {tables}

Question: {query}

Output ONLY a JSON array of steps. Each step has:
- "op": one of "RecordScan", "Join", "Extract", "Count", "GroupBy", "Sum"
- "params": op-specific parameters (see below)
- "input": "prev" (use previous step output) or a table name (load directly)

Op params:
- RecordScan: {{"table": "<table_name>", "condition": "<natural language condition>"}}
- Join:       {{"left": "prev"|"<table>", "right": "prev"|"<table>", "left_key": "<field>", "right_key": "<field>"}}
- Extract:    {{"columns": ["<col1>", ...]}}
- Count:      {{"field": "<field_name>" or null}}
- GroupBy:    {{"field": "<group_field>", "sum_field": "<sum_field>"}}
- Sum:        {{"field": "<field>"}}

Example for "For patients with severe thrombosis, list their ID, sex and disease":
[
  {{"op": "RecordScan", "params": {{"table": "Examination.json", "condition": "Thrombosis equals 2"}}}},
  {{"op": "Join", "params": {{"left": "prev", "right": "Patient.json", "left_key": "ID", "right_key": "ID"}}}},
  {{"op": "Extract", "params": {{"columns": ["ID", "SEX", "Diagnosis"]}}}}
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
    ) -> DagNode:
        """
        쿼리를 분석해 실행 가능한 DagNode 트리 반환.

        Raises:
            ValueError: LLM 응답이 파싱 불가하거나 steps가 비어있을 때
        """
        tables = list(all_records_data.keys())
        prompt = self._build_prompt(query, tables, term_context)
        logger.debug("[PLANNER] prompt (%d chars):\n%s", len(prompt), prompt)
        response = llm_fn(prompt)
        logger.debug("[PLANNER] llm_response:\n%s", response)
        steps = self._parse_plan(response)
        logger.info("[PLANNER] parsed steps: %s", [s["op"] for s in steps])
        return self._steps_to_dag(steps, all_records_data)

    def _build_prompt(self, query: str, tables: list[str], term_context: TermContext | None = None) -> str:
        """LLM에게 전달할 계획 요청 프롬프트 생성."""
        if term_context:
            ctx_str = term_context.as_context_string()
            context_block = f"\n{ctx_str}\n" if ctx_str else ""
        else:
            context_block = ""
        return PLAN_PROMPT_TEMPLATE.format(
            context=context_block,
            tables=", ".join(tables),
            query=query,
        )

    def _parse_plan(self, response: str) -> list[dict]:
        """LLM 응답(JSON 배열)을 파싱. 마크다운 코드블록 있으면 제거."""
        # 마크다운 코드블록 제거 (```json ... ``` 또는 ``` ... ```)
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

            elif op == "Join":
                left_src = params.get("left", "prev")
                right_src = params.get("right", "")

                # left: "prev"이면 이전 노드, 아니면 테이블 직접 로딩
                if left_src == "prev" and prev_node is not None:
                    left_node = prev_node
                else:
                    left_recs = self._resolve_records(left_src, all_records_data)
                    left_node = DagNode("RecordScan", {"records": left_recs, "condition": ""})

                right_recs = self._resolve_records(right_src, all_records_data)
                right_node = DagNode("RecordScan", {"records": right_recs, "condition": ""})

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

        # 확장자 없는 경우: ".json" 붙여서 재시도
        with_json = source + ".json"
        if with_json in all_records_data:
            return all_records_data[with_json]

        # 확장자 있는 경우: 제거 후 재시도
        stem = source.rsplit(".", 1)[0] if "." in source else source
        if stem in all_records_data:
            return all_records_data[stem]

        return []
