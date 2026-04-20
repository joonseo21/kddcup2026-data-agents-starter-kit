# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Purpose

KDD Cup 2026 DataAgent-Bench starter kit. A ReAct-style baseline agent that reads data analysis tasks (question + context files), reasons through them using tools (SQL, Python, file I/O), and outputs CSV predictions.

## Commands

```bash
# Install dependencies (creates .venv)
uv sync

# Run all tests
uv run pytest

# Run a single test file
uv run pytest tests/test_agent.py -v

# Lint / format
uv run ruff check src tests
uv run ruff format src tests

# Verify setup
uv run dabench status --config configs/react_baseline.example.yaml

# Run a single task
uv run dabench run-task task_1 --config configs/react_baseline.example.yaml

# Run full benchmark (optionally limit tasks)
uv run dabench run-benchmark --config configs/react_baseline.example.yaml --limit 5
```

## Architecture

**Data flow**: Config YAML → dataset loads `task.json` metadata → `ReActAgent` loops until `answer` tool called → outputs `trace.json` + `prediction.csv` per task.

### Key Components

| Component | File | Role |
|-----------|------|------|
| CLI | `src/data_agent_baseline/cli.py` | Typer entry point; `status`, `inspect-task`, `run-task`, `run-benchmark` |
| Config | `src/data_agent_baseline/config.py` | Dataclass-based YAML config (`DatasetConfig`, `AgentConfig`, `RunConfig`) |
| ReActAgent | `src/data_agent_baseline/agents/react.py` | Main loop: parses JSON `{thought, action, action_input}`, calls tools, records steps |
| ModelAdapter | `src/data_agent_baseline/agents/model.py` | Protocol for LLM calls; `OpenAIModelAdapter` for real inference, `ScriptedModelAdapter` for tests (no API calls) |
| ToolRegistry | `src/data_agent_baseline/tools/registry.py` | 8 tools: `list_context`, `read_csv`, `read_json`, `read_doc`, `inspect_sqlite_schema`, `execute_context_sql`, `execute_python`, `answer` |
| Dataset | `src/data_agent_baseline/benchmark/dataset.py` | Loads tasks from `data/public/input/task_<id>/`; discovers context files |
| Runner | `src/data_agent_baseline/run/runner.py` | Single-task execution + `ThreadPoolExecutor`-based parallel benchmark runs |

### Agent Loop

1. Model receives system prompt (ReAct rules + tool descriptions) + task prompt (question + context paths)
2. Model returns JSON: `{"thought": "...", "action": "<tool_name>", "action_input": {...}}`
3. Tool executes; result becomes next observation prompt
4. Repeat until `answer` tool called or `max_steps` reached

### Testing Strategy

- Tests use `ScriptedModelAdapter` to replay scripted model responses without LLM API calls
- Tool tests use pytest `tmp_path` fixtures to create fake context directories
- `aop/` (semantic parsing) is experimental/in-progress code on this branch

### Output Artifacts

```
artifacts/runs/<run_id>/
  summary.json              # Benchmark-level metrics
  <task_id>/
    trace.json              # Full step-by-step execution log
    prediction.csv          # Final answer table
```

### Config Schema

```yaml
dataset:
  root_path: data/public/input
agent:
  model: gpt-4-turbo
  api_base: https://api.openai.com/v1
  api_key: sk-...
  max_steps: 16
  temperature: 0.0
run:
  output_dir: artifacts/runs
  max_workers: 4
  task_timeout_seconds: 600   # 0 to disable
```
