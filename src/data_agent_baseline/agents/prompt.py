from __future__ import annotations

import json

from data_agent_baseline.benchmark.schema import PublicTask


REACT_SYSTEM_PROMPT = """
You are a ReAct-style data agent.

You are solving a task from a public dataset. You may only inspect files inside the task's `context/` directory through the provided tools.

Rules:
1. Use tools to inspect the available context before answering.
2. Base your answer only on information you can observe through the provided tools.
3. The task is complete only when you call the `answer` tool.
4. The `answer` tool must receive a table with `columns` and `rows`.
5. Always return exactly one JSON object with keys `thought`, `action`, and `action_input` - We will reask the same question if this format isn't kept.
6. Always wrap that JSON object in exactly one fenced code block that starts with ```json and ends with ```.
7. Do not output any text before or after the fenced JSON block.

Keep reasoning concise and grounded in the observed data.
""".strip()

##M1#########################
OPERATOR_PLAN_PROMPT = """
You MUST follow this operator plan in order:
Phase 1 (Step 1): Call list_context to understand available files.
Phase 2 (Step 2): Call read_doc on knowledge.md if it exists, to understand the data semantics.
Phase 3 (Steps 3+): Explore data files (read_csv, read_json, execute_python, execute_context_sql, etc.) to find the answer.
Phase 4 (Final): Call answer with the result table.

Do NOT skip phases. Do NOT call answer before completing Phase 3.
""".strip()
##############################

RESPONSE_EXAMPLES = """
Example response when you need to inspect the context - reminder that action_input should always be a singular json:
```json
{"thought":"I should inspect the available files first.","action":"list_context","action_input":{"max_depth":4}}
```

Example response when running Python code:
```json
{"thought":"I will run Python to process the data.","action":"execute_python","action_input":{"code":"import json\nprint('hello')"}}
```

Example response when you have the final answer:
```json
{"thought":"I have the final result table.","action":"answer","action_input":{"columns":["average_long_shots"],"rows":[["63.5"]]}}
```
""".strip()

##M4#########################
M4_PLAN_SYSTEM_PROMPT = """
You are a data agent planner. Given a question and available data schema, produce a step-by-step execution plan as a JSON array.

Each step must be a JSON object with:
- "action": one of [list_context, read_doc, read_csv, read_json, inspect_sqlite_schema, execute_context_sql, execute_python]
- "action_input": a JSON object with required parameters
- "purpose": a short string explaining what this step is for

Rules:
- Do NOT include the answer step — it is handled automatically after execution.
- If knowledge.md is already provided in schema context, do NOT re-read it.
- Order steps logically: inspect schema first, then query data.
- Be concise — only include steps truly necessary to answer the question.
- Return ONLY a raw JSON array. No markdown, no explanation, no extra text.

Example:
[
  {"action": "inspect_sqlite_schema", "action_input": {"path": "db/student_club.sqlite"}, "purpose": "understand table structure"},
  {"action": "execute_context_sql", "action_input": {"path": "db/student_club.sqlite", "sql": "SELECT COUNT(*) FROM members WHERE major = 'CS'"}, "purpose": "count CS members"}
]
""".strip()

M4_VALIDATE_SYSTEM_PROMPT = """
You are a data agent validator. A plan step was just executed.
Check if the result is valid and sufficient for its stated purpose.

Respond with a JSON object:
{"valid": true/false, "reason": "short explanation"}

Return ONLY this JSON object, no other text.
""".strip()

M4_ANSWER_SYSTEM_PROMPT = """
You are a data agent. All planned steps have been executed.
Based on the execution results provided, call the answer tool with the final result table.

Return exactly one JSON object in a fenced code block:
```json
{"thought": "...", "action": "answer", "action_input": {"columns": [...], "rows": [[...]]}}
```
""".strip()
##############################

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


def build_task_prompt(task: PublicTask, schema_context: str | None = None) -> str:
    base = (
        f"Question: {task.question}\n"
        "All tool file paths are relative to the task context directory. "
        "When you have the final table, call the `answer` tool."
    )
    ##M2############################
    if schema_context:
        base += f"\n\n[Schema Context]\n{schema_context}"
    #################################
    return base

#M0+를 끌경우 remainingsteps는 항상 none으로 삽입되어 기존대로 글이 안붙음.#
##M0Plus########################
def build_observation_prompt(observation: dict[str, object], remaining_steps: int | None = None) -> str:
    rendered = json.dumps(observation, ensure_ascii=False, indent=2)
    suffix = ""
    if remaining_steps is not None:
        if remaining_steps <= 3:
            suffix = f"\n[WARNING: Only {remaining_steps} steps remaining. You should call answer soon.]"
        else:
            suffix = f"\n[Remaining steps: {remaining_steps}]"
    return f"Observation:\n{rendered}{suffix}"
#################################

##M1############################
def get_operator_hint(step_index: int, completed_actions: list[str]) -> str | None:
    if step_index == 1 and "list_context" not in completed_actions:
        return "OPERATOR HINT: You must call list_context first."
    if step_index == 2 and "read_doc" not in completed_actions:
        return "OPERATOR HINT: You must call read_doc on knowledge.md now if it exists."
    return None
#################################

##M2############################
def build_schema_context(task: PublicTask, max_knowledge_chars: int = 2000) -> str | None:
    context_dir = task.context_dir
    parts: list[str] = []

    # 1. 파일 목록
    file_list: list[str] = []
    for f in sorted(context_dir.rglob("*")):
        if f.is_file():
            rel = f.relative_to(context_dir).as_posix()
            file_list.append(rel)
    if file_list:
        parts.append("Available files:\n" + "\n".join(f"  - {f}" for f in file_list))

    # 2. knowledge.md 읽기
    knowledge_path = context_dir / "knowledge.md"
    if knowledge_path.exists():
        try:
            knowledge_text = knowledge_path.read_text(encoding="utf-8", errors="replace")
            if len(knowledge_text) > max_knowledge_chars:
                knowledge_text = knowledge_text[:max_knowledge_chars] + "\n...[truncated]"
            parts.append(f"knowledge.md:\n{knowledge_text}")
        except Exception:
            pass

    # 3. CSV 컬럼명 미리 읽기
    csv_files = list(context_dir.rglob("*.csv"))
    for csv_path in csv_files[:5]:  # 최대 5개
        try:
            with csv_path.open(encoding="utf-8", errors="replace") as f:
                header = f.readline().strip()
            rel = csv_path.relative_to(context_dir).as_posix()
            parts.append(f"{rel} columns: {header}")
        except Exception:
            pass

    # 4. JSON 최상위 키 / 첫 레코드 키 미리 읽기
    json_files = list(context_dir.rglob("*.json"))
    for json_path in json_files[:5]:  # 최대 5개
        try:
            text = json_path.read_text(encoding="utf-8", errors="replace")
            payload = json.loads(text)
            rel = json_path.relative_to(context_dir).as_posix()
            if isinstance(payload, dict):
                if "records" in payload and isinstance(payload["records"], list) and payload["records"]:
                    keys = list(payload["records"][0].keys())
                    table_name = payload.get("table", rel)
                    parts.append(f"{rel} (table={table_name}) record keys: {keys}")
                else:
                    parts.append(f"{rel} top-level keys: {list(payload.keys())}")
            elif isinstance(payload, list) and payload and isinstance(payload[0], dict):
                keys = list(payload[0].keys())
                parts.append(f"{rel} record keys: {keys}")
        except Exception:
            pass

    if not parts:
        return None
    return "\n\n".join(parts)
#################################

##M3############################
def validate_answer(columns: list, rows: list) -> tuple[bool, str]:
    """
    answer 결과를 검증. (ok, reason) 반환.
    ok=False면 repair 루프 진입.
    """
    if not columns:
        return False, "Answer has no columns."
    if not isinstance(columns, list) or not all(isinstance(c, str) for c in columns):
        return False, "Columns must be a list of strings."
    if not isinstance(rows, list):
        return False, "Rows must be a list."
    if len(rows) == 0:
        return False, "Answer has no rows. If the result is truly empty, double-check your query."
    for i, row in enumerate(rows):
        if not isinstance(row, list):
            return False, f"Row {i} is not a list."
        if len(row) != len(columns):
            return False, f"Row {i} has {len(row)} values but expected {len(columns)} (columns: {columns})."
    return True, "ok"
#################################