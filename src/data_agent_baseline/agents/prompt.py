from __future__ import annotations

import json

from data_agent_baseline.benchmark.schema import PublicTask


REACT_SYSTEM_PROMPT = """
You are a ReAct-style data agent.

You are solving a task from a public dataset. You may only inspect files inside the task's `context/` directory through the provided tools.

Rules:
1. Base your answer only on information you can observe through the provided tools.
2. The task is complete only when you call the `answer` tool.
3. The `answer` tool must receive a table with `columns` and `rows`.
4. Always return exactly one JSON object with keys `thought`, `action`, and `action_input`.
5. Always wrap that JSON object in exactly one fenced code block that starts with ```json and ends with ```.
6. Do not output any text before or after the fenced JSON block.
7. If structured data (SQLite database) is provided in the task, use execute_context_sql with the given path directly. Do NOT call scan or list_context first.
8. Use `retrieve` for markdown documents, and use `link` only after you already have the database path and table name.

Keep reasoning concise and grounded in the observed data.
""".strip()

RESPONSE_EXAMPLES = """
Example response when querying the pre-loaded SQLite database:
```json
{"thought":"The database path and schema are provided. I'll query directly.","action":"execute_context_sql","action_input":{"path":"/tmp/data_agent_structured_xxx.sqlite","sql":"SELECT ...","limit":200}}
```

Example response when you have the final answer:
```json
{"thought":"I have the final result table.","action":"answer","action_input":{"columns":["average_long_shots"],"rows":[["63.5"]]}}
```
""".strip()


def build_system_prompt(tool_descriptions: str, system_prompt: str | None = None) -> str:
    base_prompt = system_prompt or REACT_SYSTEM_PROMPT
    return (
        f"{base_prompt}\n\n"
        "Available tools:\n"
        f"{tool_descriptions}\n\n"
        f"{RESPONSE_EXAMPLES}\n\n"
        "You must always return a single ```json fenced block containing one JSON object "
        "with keys `thought`, `action`, and `action_input`, and no extra text."
    )


def build_task_prompt(task: PublicTask, prescan_result: dict | None = None) -> str:
    base = (
        f"Question: {task.question}\n"
        "All tool file paths are relative to the task context directory. "
        "When you have the final table, call the `answer` tool."
    )
    if not prescan_result:
        return base + " If you need to analyze structured CSV/JSON data, run `scan` first."

    db_path = prescan_result["path"]
    table_lines = []
    for t in prescan_result["tables"]:
        cols = ", ".join(t["columns"])
        table_lines.append(f"  - {t['table_name']} (columns: {cols}) — {t['row_count']} rows")
    tables_str = "\n".join(table_lines)

    return (
        f"{base}\n\n"
        f"Structured data is already loaded into a SQLite database:\n"
        f"  path: {db_path}\n"
        f"Tables:\n{tables_str}\n\n"
        f"Use execute_context_sql with this path directly. Do NOT call scan or list_context."
    )


def build_observation_prompt(observation: dict[str, object]) -> str:
    rendered = json.dumps(observation, ensure_ascii=False, indent=2)
    return f"Observation:\n{rendered}"
