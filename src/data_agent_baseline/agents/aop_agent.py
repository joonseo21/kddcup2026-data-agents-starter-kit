from __future__ import annotations

import logging
from dataclasses import dataclass
from time import perf_counter
from typing import Any

logger = logging.getLogger(__name__)

from data_agent_baseline.agents.model import ModelAdapter, ModelMessage
from data_agent_baseline.agents.runtime import AgentRunResult, StepRecord
from data_agent_baseline.benchmark.schema import AnswerTable, PublicTask
from data_agent_baseline.aop.data_loader import load_records, load_sqlite_sources
from data_agent_baseline.tools.scan import build_structured_sqlite_database

_LARGE_TABLE_THRESHOLD = 1000
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
        t0 = perf_counter()
        tid = task.task_id
        logger.info("[%s] START question=%r", tid, task.question[:120])

        def llm_fn(prompt: str) -> str:
            return self._model.complete([ModelMessage(role="user", content=prompt)])

        try:
            # Step 1: Link — read knowledge.md + column names
            term_context = LinkOperator(task.task_dir).execute()
            logger.info("[%s] LINK knowledge=%d chars files=%s",
                        tid, len(term_context.knowledge), list(term_context.columns.keys()))
            steps.append(_make_step(0, "link", {"task_dir": str(task.task_dir)}, term_context.as_context_string()))

            # Step 2: Load data records + SQLite source paths
            context_dir = task.task_dir / "context"
            all_records, sqlite_sources = _collect_sources(context_dir, task=task)
            logger.info("[%s] DATA tables=%s sqlite=%s",
                        tid, {k: len(v) for k, v in all_records.items()}, list(sqlite_sources.keys()))

            # Step 3: Plan — build DAG from query
            dag = self._planner.plan(
                query=task.question,
                all_records_data=all_records,
                llm_fn=llm_fn,
                term_context=term_context,
                sqlite_sources=sqlite_sources,
            )
            logger.info("[%s] PLAN dag_root=%s", tid, dag.op_type)
            steps.append(_make_step(1, "plan", {"query": task.question}, repr(dag)))

            # Step 4: Execute DAG
            executor = DagExecutor(llm_fn=llm_fn)
            result = executor.execute(dag)
            result_summary = len(result) if isinstance(result, list) else result
            logger.info("[%s] EXEC result=%r", tid, result_summary)
            steps.append(_make_step(2, "execute", {}, repr(result)))

        except Exception as exc:  # noqa: BLE001
            logger.error("[%s] FAILED after %.1fs: %s", tid, perf_counter() - t0, exc)
            return AgentRunResult(
                task_id=task.task_id,
                answer=None,
                steps=steps,
                failure_reason=str(exc),
            )

        answer = _result_to_answer_table(result)
        elapsed = perf_counter() - t0
        if answer is not None:
            logger.info("[%s] DONE rows=%d elapsed=%.1fs", tid, len(answer.rows), elapsed)
        else:
            logger.warning("[%s] NO_ANSWER result_type=%s elapsed=%.1fs", tid, type(result).__name__, elapsed)
        return AgentRunResult(
            task_id=task.task_id,
            answer=answer,
            steps=steps,
            failure_reason=None if answer is not None else "DAG produced unrecognised result type",
        )


def _collect_sources(context_dir, task=None) -> tuple[dict, dict]:
    """Return (all_records, sqlite_sources), supporting flat and subdirectory layouts."""
    from pathlib import Path
    context_dir = Path(context_dir)
    if not context_dir.exists():
        return {}, {}
    all_records: dict = {}
    sqlite_srcs: dict = {}
    for subdir in ("csv", "json", "db"):
        sub = context_dir / subdir
        if sub.is_dir():
            all_records.update(load_records(sub))
            sqlite_srcs.update(load_sqlite_sources(sub))
    if not all_records and not sqlite_srcs:
        all_records = load_records(context_dir)
        sqlite_srcs = load_sqlite_sources(context_dir)
    if task is not None:
        all_records, sqlite_srcs = _migrate_large_to_sqlite(task, all_records, sqlite_srcs)
    return all_records, sqlite_srcs


def _migrate_large_to_sqlite(task, all_records: dict, sqlite_srcs: dict) -> tuple[dict, dict]:
    """Move in-memory tables exceeding _LARGE_TABLE_THRESHOLD rows to a temp SQLite DB."""
    from pathlib import Path
    large_keys = [k for k, v in all_records.items() if len(v) > _LARGE_TABLE_THRESHOLD]
    if not large_keys:
        return all_records, sqlite_srcs

    large_sources: list[str] = []
    for key in large_keys:
        matches = list(task.context_dir.rglob(key))
        if matches:
            large_sources.append(matches[0].relative_to(task.context_dir).as_posix())

    if not large_sources:
        return all_records, sqlite_srcs

    try:
        scan_result = build_structured_sqlite_database(task, sources=large_sources)
        db_path = Path(scan_result["path"])
        new_records = {k: v for k, v in all_records.items() if k not in large_keys}
        new_sqlite = dict(sqlite_srcs)
        for tbl in scan_result["tables"]:
            key = f"{Path(tbl['source_path']).name}::{tbl['table_name']}"
            new_sqlite[key] = (db_path, tbl["table_name"])
        logger.info("SCAN migrated large tables to temp SQLite: %s", large_sources)
        return new_records, new_sqlite
    except Exception as exc:
        logger.warning("SCAN migration failed (%s); keeping in-memory records.", exc)
        return all_records, sqlite_srcs


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
