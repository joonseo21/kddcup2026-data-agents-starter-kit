from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from data_agent_baseline.benchmark.schema import AnswerTable, PublicTask
from data_agent_baseline.tools.filesystem import (
    list_context_tree,
    read_csv_preview,
    read_doc_preview,
    read_json_preview,
    resolve_context_path,
)
from data_agent_baseline.tools.link import link_sources
from data_agent_baseline.tools.python_exec import execute_python_code
from data_agent_baseline.tools.retrieve import build_markdown_database, retrieve_by_keyword, search_keyword_database
from data_agent_baseline.tools.scan import scan_sources
from data_agent_baseline.tools.sqlite import execute_read_only_sql, inspect_sqlite_schema

EXECUTE_PYTHON_TIMEOUT_SECONDS = 30


@dataclass(frozen=True, slots=True)
class ToolSpec:
    name: str
    description: str
    input_schema: dict[str, Any]


@dataclass(frozen=True, slots=True)
class ToolExecutionResult:
    ok: bool
    content: dict[str, Any]
    is_terminal: bool = False
    answer: AnswerTable | None = None


ToolHandler = Callable[[PublicTask, dict[str, Any]], ToolExecutionResult]


def _resolve_sqlite_tool_path(task: PublicTask, raw_path: str):
    # Allow SQL tools to open either a context-relative sqlite file or an
    # absolute path returned by scan temp database creation.
    candidate = Path(raw_path)
    if candidate.is_absolute():
        resolved = candidate.resolve()
        if not resolved.exists():
            raise FileNotFoundError(f"Missing sqlite database: {raw_path}")
        return resolved
    return resolve_context_path(task, raw_path)


def _list_context(task: PublicTask, action_input: dict[str, Any]) -> ToolExecutionResult:
    max_depth = int(action_input.get("max_depth", 4))
    return ToolExecutionResult(ok=True, content=list_context_tree(task, max_depth=max_depth))


def _read_csv(task: PublicTask, action_input: dict[str, Any]) -> ToolExecutionResult:
    path = str(action_input["path"])
    max_rows = int(action_input.get("max_rows", 20))
    return ToolExecutionResult(ok=True, content=read_csv_preview(task, path, max_rows=max_rows))


def _read_json(task: PublicTask, action_input: dict[str, Any]) -> ToolExecutionResult:
    path = str(action_input["path"])
    max_chars = int(action_input.get("max_chars", 4000))
    return ToolExecutionResult(ok=True, content=read_json_preview(task, path, max_chars=max_chars))


def _read_doc(task: PublicTask, action_input: dict[str, Any]) -> ToolExecutionResult:
    path = str(action_input["path"])
    max_chars = int(action_input.get("max_chars", 4000))
    return ToolExecutionResult(ok=True, content=read_doc_preview(task, path, max_chars=max_chars))


def _retrieve(task: PublicTask, action_input: dict[str, Any]) -> ToolExecutionResult:
    # Route retrieval requests to the markdown-only keyword retriever. The
    # caller can either provide markdown sources or a prebuilt retrieval DB.
    top_k = int(action_input.get("top_k", 5))
    if top_k < 1:
        raise ValueError("retrieve.top_k must be at least 1.")
    query = str(action_input["query"])
    raw_sources = action_input.get("sources")
    database = action_input.get("database")

    if raw_sources is None:
        sources = None
    elif isinstance(raw_sources, list) and all(isinstance(item, str) for item in raw_sources):
        sources = list(raw_sources)
    else:
        raise ValueError("retrieve.sources must be a list of relative paths.")

    if database is None:
        content = retrieve_by_keyword(task, query=query, sources=sources, top_k=top_k)
    elif isinstance(database, dict):
        content = search_keyword_database(query=query, database=database, top_k=top_k)
    else:
        raise ValueError("retrieve.database must be an object when provided.")

    return ToolExecutionResult(ok=True, content=content)


def _build_retrieval_database(task: PublicTask, action_input: dict[str, Any]) -> ToolExecutionResult:
    # Build a reusable markdown retrieval database so the agent can search the
    # same set of documents multiple times without redefining sources each turn.
    raw_sources = action_input.get("sources")
    if raw_sources is None:
        sources = None
    elif isinstance(raw_sources, list) and all(isinstance(item, str) for item in raw_sources):
        sources = list(raw_sources)
    else:
        raise ValueError("build_retrieval_database.sources must be a list of relative paths.")

    return ToolExecutionResult(ok=True, content=build_markdown_database(task, sources=sources))


def _scan(task: PublicTask, action_input: dict[str, Any]) -> ToolExecutionResult:
    # Run the Scan operator over structured sources and return the temporary
    # SQLite database metadata that downstream SQL tools will consume.
    raw_sources = action_input.get("sources")
    if raw_sources is None:
        sources = None
    elif isinstance(raw_sources, list) and all(isinstance(item, str) for item in raw_sources):
        sources = list(raw_sources)
    else:
        raise ValueError("scan.sources must be a list of relative paths.")

    return ToolExecutionResult(ok=True, content=scan_sources(task, sources=sources))

def _link(task: PublicTask, action_input: dict[str, Any]) -> ToolExecutionResult:
    # Run the Link operator on allowed source pairs only:
    # db-text, db-md, text-text, text-md, md-md.
    left_db_path = action_input.get("left_db_path")
    left_table = action_input.get("left_table")
    left_source = action_input.get("left_source")
    right_source = action_input.get("right_source")
    right_db_path = action_input.get("right_db_path")
    right_table = action_input.get("right_table")
    right_field = action_input.get("right_field")
    left_field = action_input.get("left_field")
    contains = action_input.get("contains")
    top_k = int(action_input.get("top_k", 20))
    resolved_left_db_path = None
    resolved_right_db_path = None
    if left_db_path is not None:
        resolved_left_db_path = _resolve_sqlite_tool_path(task, str(left_db_path))
    if right_db_path is not None:
        resolved_right_db_path = _resolve_sqlite_tool_path(task, str(right_db_path))
    if left_source is not None and left_db_path is not None:
        raise ValueError("link.left_source and link.left_db_path are mutually exclusive.")
    if right_source is not None and right_db_path is not None:
        raise ValueError("link.right_source and link.right_db_path are mutually exclusive.")
    if resolved_left_db_path is not None and (not isinstance(left_table, str) or not left_table):
        raise ValueError("link.left_table must be a non-empty string when left_db_path is provided.")
    if resolved_left_db_path is not None and (not isinstance(left_field, str) or not left_field):
        raise ValueError("link.left_field must be a non-empty string when left_db_path is provided.")
    if right_db_path is not None:
        if not isinstance(right_table, str) or not right_table:
            raise ValueError("link.right_table must be a non-empty string when right_db_path is provided.")
        if not isinstance(right_field, str) or not right_field:
            raise ValueError("link.right_field must be a non-empty string when right_db_path is provided.")
    if top_k < 1:
        raise ValueError("link.top_k must be at least 1.")

    return ToolExecutionResult(
        ok=True,
        content=link_sources(
            task,
            left_db_path=resolved_left_db_path,
            left_table=str(left_table) if left_table is not None else None,
            left_field=str(left_field) if left_field is not None else None,
            left_source=str(left_source) if left_source is not None else None,
            right_source=str(right_source) if right_source is not None else None,
            right_db_path=resolved_right_db_path,
            right_table=str(right_table) if right_table is not None else None,
            right_field=str(right_field) if right_field is not None else None,
            contains=str(contains) if contains is not None else None,
            top_k=top_k,
        ),
    )


def _inspect_sqlite_schema(task: PublicTask, action_input: dict[str, Any]) -> ToolExecutionResult:
    raw_path = str(action_input["path"])
    path = _resolve_sqlite_tool_path(task, raw_path)
    return ToolExecutionResult(ok=True, content=inspect_sqlite_schema(path))


def _execute_context_sql(task: PublicTask, action_input: dict[str, Any]) -> ToolExecutionResult:
    raw_path = str(action_input["path"])
    path = _resolve_sqlite_tool_path(task, raw_path)
    sql = str(action_input["sql"])
    limit = int(action_input.get("limit", 200))
    return ToolExecutionResult(ok=True, content=execute_read_only_sql(path, sql, limit=limit))


def _execute_python(task: PublicTask, action_input: dict[str, Any]) -> ToolExecutionResult:
    code = str(action_input["code"])
    content = execute_python_code(
        context_root=task.context_dir,
        code=code,
        timeout_seconds=EXECUTE_PYTHON_TIMEOUT_SECONDS,
    )
    return ToolExecutionResult(ok=bool(content.get("success")), content=content)


def _answer(_: PublicTask, action_input: dict[str, Any]) -> ToolExecutionResult:
    columns = action_input.get("columns")
    rows = action_input.get("rows")
    if not isinstance(columns, list) or not columns or not all(isinstance(item, str) for item in columns):
        raise ValueError("answer.columns must be a non-empty list of strings.")
    if not isinstance(rows, list):
        raise ValueError("answer.rows must be a list.")

    normalized_rows: list[list[Any]] = []
    for row in rows:
        if not isinstance(row, list):
            raise ValueError("Each answer row must be a list.")
        if len(row) != len(columns):
            raise ValueError("Each answer row must match the number of columns.")
        normalized_rows.append(list(row))

    answer = AnswerTable(columns=list(columns), rows=normalized_rows)
    return ToolExecutionResult(
        ok=True,
        content={
            "status": "submitted",
            "column_count": len(columns),
            "row_count": len(normalized_rows),
        },
        is_terminal=True,
        answer=answer,
    )


@dataclass(slots=True)
class ToolRegistry:
    specs: dict[str, ToolSpec]
    handlers: dict[str, ToolHandler]

    def describe_for_prompt(self) -> str:
        lines = []
        for name in sorted(self.specs):
            spec = self.specs[name]
            lines.append(f"- {spec.name}: {spec.description}")
            lines.append(f"  input_schema: {spec.input_schema}")
        return "\n".join(lines)

    def execute(self, task: PublicTask, action: str, action_input: dict[str, Any]) -> ToolExecutionResult:
        if action not in self.handlers:
            raise KeyError(f"Unknown tool: {action}")
        return self.handlers[action](task, action_input)


def create_default_tool_registry() -> ToolRegistry:
    # Register the full tool surface exposed to the ReAct agent, including the
    # new markdown retrieval, structured scan, link, and SQLite conversion flow.
    specs = {
        "answer": ToolSpec(
            name="answer",
            description="Submit the final answer table. This is the only valid terminating action.",
            input_schema={
                "columns": ["column_name"],
                "rows": [["value_1"]],
            },
        ),
        "execute_context_sql": ToolSpec(
            name="execute_context_sql",
            description="Run a read-only SQL query against a sqlite/db file inside context.",
            input_schema={"path": "relative/path/to/file.sqlite", "sql": "SELECT ...", "limit": 200},
        ),
        "execute_python": ToolSpec(
            name="execute_python",
            description=(
                "Execute arbitrary Python code with the task context directory as the "
                "working directory. The tool returns the code's captured stdout as `output`. "
                f"The execution timeout is fixed at {EXECUTE_PYTHON_TIMEOUT_SECONDS} seconds."
            ),
            input_schema={
                "code": "import os\nprint(sorted(os.listdir('.')))",
            },
        ),
        "inspect_sqlite_schema": ToolSpec(
            name="inspect_sqlite_schema",
            description="Inspect tables and columns in a sqlite/db file inside context.",
            input_schema={"path": "relative/path/to/file.sqlite"},
        ),
        "list_context": ToolSpec(
            name="list_context",
            description="List files and directories available under context.",
            input_schema={"max_depth": 4},
        ),
        "build_retrieval_database": ToolSpec(
            name="build_retrieval_database",
            description=(
                "Build a markdown-only retrieval database for keyword search over .md documents."
            ),
            input_schema={
                "sources": ["doc/knowledge.md", "doc/major.md"],
            },
        ),
        "scan": ToolSpec(
            name="scan",
            description=(
                "When a task depends on structured CSV/JSON data, run this first. "
                "Convert structured CSV/JSON sources into a temporary SQLite database and return its path and tables."
            ),
            input_schema={
                "sources": ["csv/member.csv", "json/Patient.json"],
            },
        ),
        "link": ToolSpec(
            name="link",
            description=(
                "Link only these source pairs: db-text, db-md, text-text, text-md, md-md. "
                "Use *_db_path + *_table + *_field for db sides, and *_source for text or markdown sides."
            ),
            input_schema={
                "left_db_path": "/tmp/data_agent_structured_x.sqlite",
                "left_table": "member",
                "left_field": "link_to_major",
                "left_source": "doc/notes.txt",
                "right_source": "doc/major.md",
                "right_db_path": "/tmp/data_agent_structured_y.sqlite",
                "right_table": "major",
                "right_field": "registry_id",
                "contains": "Angela Sanders",
                "top_k": 10,
            },
        ),
        "retrieve": ToolSpec(
            name="retrieve",
            description=(
                "Retrieve query-relevant items from markdown documents only using keyword scoring."
            ),
            input_schema={
                "query": "champion team at 2024 Olympic Games",
                "sources": ["doc/news.md", "doc/knowledge.md"],
                "top_k": 5,
            },
        ),
        "read_csv": ToolSpec(
            name="read_csv",
            description="Read a preview of a CSV file inside context.",
            input_schema={"path": "relative/path/to/file.csv", "max_rows": 20},
        ),
        "read_doc": ToolSpec(
            name="read_doc",
            description="Read a text-like document inside context.",
            input_schema={"path": "relative/path/to/file.md", "max_chars": 4000},
        ),
        "read_json": ToolSpec(
            name="read_json",
            description="Read a preview of a JSON file inside context.",
            input_schema={"path": "relative/path/to/file.json", "max_chars": 4000},
        ),
    }
    handlers = {
        "answer": _answer,
        "build_retrieval_database": _build_retrieval_database,
        "execute_context_sql": _execute_context_sql,
        "execute_python": _execute_python,
        "inspect_sqlite_schema": _inspect_sqlite_schema,
        "link": _link,
        "list_context": _list_context,
        "retrieve": _retrieve,
        "scan": _scan,
        "read_csv": _read_csv,
        "read_doc": _read_doc,
        "read_json": _read_json,
    }
    return ToolRegistry(specs=specs, handlers=handlers)
