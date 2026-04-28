from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from data_agent_baseline.agents.model import ModelAdapter, ModelMessage
from data_agent_baseline.agents.runtime import AgentRunResult, StepRecord
from data_agent_baseline.benchmark.schema import AnswerTable, PublicTask
from data_agent_baseline.aop.data_loader import load_records
from data_agent_baseline.aop.dag_executor import DagExecutor
from data_agent_baseline.aop.operators.link import LinkOperator
from data_agent_baseline.aop.planner import SemanticPlanner


@dataclass(frozen=True, slots=True)
class AOPAgentConfig:
    max_steps: int = 16


class AOPAgent:
    """
    AOP-based agent: LinkOperator → SemanticPlanner → DagExecutor.
    Same run() interface as ReActAgent.
    """

    def __init__(
        self,
        model: ModelAdapter,
        lrs: list[dict],
        config: AOPAgentConfig | None = None,
    ) -> None:
        self._model = model
        self._planner = SemanticPlanner(lrs=lrs)
        self._config = config or AOPAgentConfig()

    def run(self, task: PublicTask) -> AgentRunResult:
        steps: list[StepRecord] = []

        def llm_fn(prompt: str) -> str:
            return self._model.complete([ModelMessage(role="user", content=prompt)])

        try:
            # Step 1: Link — read knowledge.md + column names
            term_context = LinkOperator(task.task_dir).execute()
            steps.append(_make_step(0, "link", {"task_dir": str(task.task_dir)}, term_context.as_context_string()))

            # Step 2: Load data records (handles both flat and subdirectory layouts)
            context_dir = task.task_dir / "context"
            all_records = _load_all_records(context_dir)

            # Step 3: Plan — build DAG from query
            dag = self._planner.plan(
                query=task.question,
                all_records_data=all_records,
                llm_fn=llm_fn,
                term_context=term_context,
            )
            steps.append(_make_step(1, "plan", {"query": task.question}, repr(dag)))

            # Step 4: Execute DAG
            executor = DagExecutor(llm_fn=llm_fn)
            result = executor.execute(dag)
            steps.append(_make_step(2, "execute", {}, repr(result)))

        except Exception as exc:  # noqa: BLE001
            return AgentRunResult(
                task_id=task.task_id,
                answer=None,
                steps=steps,
                failure_reason=str(exc),
            )

        answer = _result_to_answer_table(result)
        return AgentRunResult(
            task_id=task.task_id,
            answer=answer,
            steps=steps,
            failure_reason=None if answer is not None else "DAG produced unrecognised result type",
        )


def _load_all_records(context_dir) -> dict:
    """Load records from context_dir, supporting both flat and csv/json/db subdirectory layouts."""
    from pathlib import Path
    context_dir = Path(context_dir)
    if not context_dir.exists():
        return {}
    merged: dict = {}
    for subdir in ("csv", "json", "db"):
        sub = context_dir / subdir
        if sub.is_dir():
            merged.update(load_records(sub))
    if not merged:
        merged = load_records(context_dir)
    return merged


def _make_step(index: int, action: str, action_input: dict, observation: str) -> StepRecord:
    return StepRecord(
        step_index=index,
        thought="",
        action=action,
        action_input=action_input,
        raw_response="",
        observation={"output": observation},
        ok=True,
    )


def _result_to_answer_table(result: Any) -> AnswerTable | None:
    if isinstance(result, list) and result and isinstance(result[0], dict):
        columns = list(result[0].keys())
        rows = [[row.get(c) for c in columns] for row in result]
        return AnswerTable(columns=columns, rows=rows)
    if isinstance(result, (int, float)):
        return AnswerTable(columns=["count"], rows=[[result]])
    if isinstance(result, dict):
        columns = list(next(iter(result.values()), {}).keys()) if result and isinstance(next(iter(result.values())), dict) else ["group", "value"]
        if columns == ["group", "value"]:
            rows = [[k, v] for k, v in result.items()]
        else:
            rows = [[k] + [v.get(c) for c in columns] for k, v in result.items()]
        return AnswerTable(columns=columns, rows=rows)
    return None
