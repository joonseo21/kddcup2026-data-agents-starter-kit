from __future__ import annotations

import json
import re
import sqlite3
import tempfile
from pathlib import Path
from typing import Any

from data_agent_baseline.benchmark.schema import PublicTask
from data_agent_baseline.tools.filesystem import load_csv_rows, load_json_value, resolve_context_path

TABLE_NAME_SANITIZER = re.compile(r"[^A-Za-z0-9_]+")


def iter_source_paths(task: PublicTask, sources: list[str] | None) -> list[Path]:
    # 사용자 입력 source -> context 하위 실제 파일 경로
    # source 미지정 시 context 전체 파일 포함
    if not sources:
        return sorted(path for path in task.context_dir.rglob("*") if path.is_file())

    resolved_paths: list[Path] = []
    for source in sources:
        resolved = resolve_context_path(task, source)
        if resolved.is_file():
            resolved_paths.append(resolved)
            continue
        resolved_paths.extend(sorted(path for path in resolved.rglob("*") if path.is_file()))
    return resolved_paths


def to_table_name(name: str) -> str:
    # 파일명/테이블명 -> SQLite 테이블명
    sanitized = TABLE_NAME_SANITIZER.sub("_", name).strip("_").lower()
    return sanitized or "table"


def normalize_sql_value(value: Any) -> Any:
    # 중첩 JSON 값 -> SQLite 저장용 문자열
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return value


def normalize_row_dict(row: dict[str, Any], columns: list[str]) -> list[Any]:
    # row 값 순서 -> 컬럼 순서 맞춤
    return [normalize_sql_value(row.get(column)) for column in columns]


def derive_json_tables(relative_path: str, payload: Any) -> list[dict[str, Any]]:
    # JSON payload -> SQLite 적재용 테이블 spec
    # table 형태/배열 -> row 중심 테이블
    # 일반 dict -> single-row 테이블
    base_table_name = to_table_name(Path(relative_path).stem)

    if isinstance(payload, dict) and isinstance(payload.get("records"), list):
        raw_rows = payload["records"]
        table_name = to_table_name(str(payload.get("table") or base_table_name))
        rows = [row if isinstance(row, dict) else {"value": row} for row in raw_rows]
    elif isinstance(payload, list):
        if payload and all(isinstance(item, dict) for item in payload):
            table_name = base_table_name
            rows = [dict(item) for item in payload]
        else:
            table_name = base_table_name
            rows = [{"value": item} for item in payload]
    elif isinstance(payload, dict):
        table_name = base_table_name
        rows = [dict(payload)]
    else:
        table_name = base_table_name
        rows = [{"value": payload}]

    columns: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in columns:
                columns.append(str(key))
    if not columns:
        columns = ["value"]

    return [
        {
            "source_path": relative_path,
            "source_type": "json",
            "table_name": table_name,
            "columns": columns,
            "rows": [normalize_row_dict(row, columns) for row in rows],
        }
    ]


def derive_csv_table(task: PublicTask, relative_path: str) -> dict[str, Any]:
    # CSV 파일 하나 -> SQLite 적재용 테이블 spec
    columns, rows = load_csv_rows(task, relative_path)
    if not columns:
        columns = ["value"]
    normalized_rows = [
        list(row[: len(columns)]) + [""] * max(0, len(columns) - len(row))
        for row in rows
    ]
    return {
        "source_path": relative_path,
        "source_type": "csv",
        "table_name": to_table_name(Path(relative_path).stem),
        "columns": columns,
        "rows": normalized_rows,
    }


def derive_structured_tables(task: PublicTask, sources: list[str] | None = None) -> list[dict[str, Any]]:
    # .csv, .json만 변환
    tables: list[dict[str, Any]] = []
    for path in iter_source_paths(task, sources):
        relative_path = path.relative_to(task.context_dir).as_posix()
        suffix = path.suffix.lower()
        if suffix == ".csv":
            tables.append(derive_csv_table(task, relative_path))
        elif suffix == ".json":
            tables.extend(derive_json_tables(relative_path, load_json_value(task, relative_path)))
    return tables


def build_structured_sqlite_database(
    task: PublicTask,
    *,
    sources: list[str] | None = None,
) -> dict[str, Any]:
    # CSV/JSON source -> 임시 SQLite DB 적재
    # SQL 질의용 준비 단계
    table_specs = derive_structured_tables(task, sources=sources)
    if not table_specs:
        raise ValueError("No structured CSV/JSON files found for sqlite conversion.")

    handle = tempfile.NamedTemporaryFile(prefix="data_agent_structured_", suffix=".sqlite", delete=False)
    handle.close()
    db_path = Path(handle.name)

    table_counts: dict[str, int] = {}
    materialized_tables: list[dict[str, Any]] = []
    with sqlite3.connect(db_path) as conn:
        for spec in table_specs:
            table_name = spec["table_name"]
            if table_name in table_counts:
                table_counts[table_name] += 1
                table_name = f"{table_name}_{table_counts[table_name]}"
            else:
                table_counts[table_name] = 1

            columns = [str(column) for column in spec["columns"]]
            quoted_columns = [f'"{column}" TEXT' for column in columns]
            conn.execute(f'DROP TABLE IF EXISTS "{table_name}"')
            conn.execute(f'CREATE TABLE "{table_name}" ({", ".join(quoted_columns)})')

            rows = [list(row) for row in spec["rows"]]
            if rows:
                placeholders = ", ".join("?" for _ in columns)
                conn.executemany(
                    f'INSERT INTO "{table_name}" VALUES ({placeholders})',
                    rows,
                )

            materialized_tables.append(
                {
                    "source_path": spec["source_path"],
                    "source_type": spec["source_type"],
                    "table_name": table_name,
                    "columns": columns,
                    "row_count": len(rows),
                }
            )
        conn.commit()

    return {
        "database_type": "sqlite",
        "path": str(db_path),
        "table_count": len(materialized_tables),
        "tables": materialized_tables,
    }


def scan_sources(
    task: PublicTask,
    *,
    sources: list[str] | None = None,
) -> dict[str, Any]:
    # Scan operator 진입점
    # structured data -> 임시 SQLite DB
    return build_structured_sqlite_database(task, sources=sources)
