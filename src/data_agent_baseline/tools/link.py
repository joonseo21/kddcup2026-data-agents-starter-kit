from __future__ import annotations

import math
import sqlite3
from pathlib import Path
from typing import Any

from data_agent_baseline.benchmark.schema import PublicTask
from data_agent_baseline.tools.filesystem import load_document_text
from data_agent_baseline.tools.text_utils import split_text_chunks, tokenize


def contains_all_terms(text: str, query: str) -> bool:
    # row/chunk 후보 얻기 == filtering
    query_tokens = tokenize(query)
    if not query_tokens:
        return True
    text_tokens = set(tokenize(text))
    return all(token in text_tokens for token in query_tokens)


def row_text(row: dict[str, Any]) -> str:
    # SQL row -> 비교용 문자열
    return " | ".join(f"{key}: {value}" for key, value in row.items())

def source_kind_from_path(source: str) -> str:
    # 경로 -> md / text 구분
    suffix = Path(source).suffix.lower()
    if suffix == ".md":
        return "markdown"
    return "text"


def load_structured_rows(
    *,
    db_path: Path,
    table_name: str,
    field: str | None = None,
    contains: str | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    # scan 결과 DB에서 row 로드
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.execute(f'SELECT * FROM "{table_name}"')
        rows = [dict(row) for row in cursor.fetchall()]
        columns = [column[0] for column in cursor.description or []]

    if field is not None and field not in columns:
        raise ValueError(f"Field `{field}` was not found in table `{table_name}`.")

    if contains:
        rows = [row for row in rows if contains_all_terms(row_text(row), contains)]

    table_info = {
        "db_path": str(db_path),
        "table_name": table_name,
        "columns": columns,
    }
    return table_info, rows


def values_from_rows(rows: list[dict[str, Any]], field: str) -> list[dict[str, Any]]:
    # 선택 field 값 추출 + 원본 row 보존
    values: list[dict[str, Any]] = []
    for row in rows:
        value = row.get(field)
        if value in (None, ""):
            continue
        values.append({"value": str(value), "row": dict(row)})
    return values


def load_text_records(task: PublicTask, source: str, contains: str | None = None) -> list[dict[str, Any]]:
    # text/md 파일 -> chunk record 목록
    record_type = "markdown_chunk" if source_kind_from_path(source) == "markdown" else "text_chunk"
    text = load_document_text(task, source)
    records = [
        {
            "path": source,
            "record_type": record_type,
            "record_id": f"{source}#chunk:{index}",
            "text": chunk,
            "metadata": {"chunk_index": index},
        }
        for index, chunk in enumerate(split_text_chunks(text), start=1)
    ]
    if contains:
        records = [record for record in records if contains_all_terms(str(record["text"]), contains)]
    return records


def link_structured_rows_to_text(
    *,
    table_info: dict[str, Any],
    candidates: list[dict[str, Any]],
    text_records: list[dict[str, Any]],
    top_k: int = 20,
) -> dict[str, Any]:
    # structured 값 -> text/md chunk 매칭
    links: list[dict[str, Any]] = []

    for candidate in candidates:
        normalized_value = candidate["value"].casefold()
        for text_record in text_records:
            text_content = str(text_record.get("text", "")).casefold()
            if normalized_value and normalized_value in text_content:
                links.append(
                    {
                        "left": {
                            "db_path": table_info["db_path"],
                            "table_name": table_info["table_name"],
                            "row": candidate["row"],
                        },
                        "right": dict(text_record),
                        "matched_value": candidate["value"],
                        "match_type": "structured_to_text",
                    }
                )

    return {
        "link_count": len(links),
        "links": links[:top_k],
        "truncated": len(links) > top_k,
    }


def text_link_score(left_text: str, right_text: str) -> float:
    # text chunk 간 토큰 overlap 점수
    left_tokens = set(tokenize(left_text))
    right_tokens = set(tokenize(right_text))
    if not left_tokens or not right_tokens:
        return 0.0

    left_folded = left_text.casefold()
    right_folded = right_text.casefold()
    if left_folded in right_folded or right_folded in left_folded:
        return 1.0

    overlap = left_tokens & right_tokens
    if len(overlap) < 2:
        return 0.0

    return len(overlap) / math.sqrt(len(left_tokens) * len(right_tokens))


def link_text_records(
    *,
    left_records: list[dict[str, Any]],
    right_records: list[dict[str, Any]],
    top_k: int = 20,
) -> dict[str, Any]:
    # text/md chunk 간 매칭
    links: list[dict[str, Any]] = []

    for left_record in left_records:
        left_text = str(left_record.get("text", ""))
        for right_record in right_records:
            right_text = str(right_record.get("text", ""))
            score = text_link_score(left_text, right_text)
            if score <= 0:
                continue

            links.append(
                {
                    "left": dict(left_record),
                    "right": dict(right_record),
                    "match_type": "text_to_text",
                    "score": round(score, 6),
                }
            )

    links.sort(key=lambda item: item.get("score", 0.0), reverse=True)
    return {
        "link_count": len(links),
        "links": links[:top_k],
        "truncated": len(links) > top_k,
    }


def swap_link_orientation(result: dict[str, Any]) -> dict[str, Any]:
    # left/right 방향 뒤집기
    swapped_links: list[dict[str, Any]] = []
    for link in result.get("links", []):
        swapped = dict(link)
        swapped["left"], swapped["right"] = swapped["right"], swapped["left"]
        swapped_links.append(swapped)

    return {
        "link_count": int(result.get("link_count", len(swapped_links))),
        "links": swapped_links,
        "truncated": bool(result.get("truncated", False)),
    }


def validate_source_pair(
    *,
    left_db_path: Path | None,
    left_source: str | None,
    right_db_path: Path | None,
    right_source: str | None,
) -> None:
    # 허용 조합 검증
    left_is_db = left_db_path is not None
    right_is_db = right_db_path is not None
    left_is_text = left_source is not None
    right_is_text = right_source is not None

    if left_is_db and right_is_db:
        raise ValueError("link does not support db-to-db comparisons.")
    if left_is_text and right_is_text:
        return
    if left_is_db and right_is_text:
        return
    if left_is_text and right_is_db:
        return
    raise ValueError("link requires one db side and one text side, or two text sources.")


def link_sources(
    task: PublicTask,
    *,
    left_db_path: Path | None = None,
    left_table: str | None = None,
    left_field: str | None = None,
    left_source: str | None = None,
    right_db_path: Path | None = None,
    right_table: str | None = None,
    right_field: str | None = None,
    right_source: str | None = None,
    contains: str | None = None,
    top_k: int = 20,
) -> dict[str, Any]:
    # 허용 pair 기준 link 수행
    validate_source_pair(
        left_db_path=left_db_path,
        left_source=left_source,
        right_db_path=right_db_path,
        right_source=right_source,
    )

    if left_db_path is not None and right_source is not None:
        if left_table is None or left_field is None:
            raise ValueError("db source requires left_table and left_field.")
        left_table_info, left_rows = load_structured_rows(
            db_path=left_db_path,
            table_name=left_table,
            field=left_field,
            contains=contains,
        )
        left_candidates = values_from_rows(left_rows, left_field)
        right_records = load_text_records(task, right_source)
        return link_structured_rows_to_text(
            table_info=left_table_info,
            candidates=left_candidates,
            text_records=right_records,
            top_k=top_k,
        )

    if left_source is not None and right_db_path is not None:
        if right_table is None or right_field is None:
            raise ValueError("db source requires right_table and right_field.")
        right_table_info, right_rows = load_structured_rows(
            db_path=right_db_path,
            table_name=right_table,
            field=right_field,
        )
        right_candidates = values_from_rows(right_rows, right_field)
        left_records = load_text_records(task, left_source, contains=contains)
        result = link_structured_rows_to_text(
            table_info=right_table_info,
            candidates=right_candidates,
            text_records=left_records,
            top_k=top_k,
        )
        return swap_link_orientation(result)

    if left_source is None or right_source is None:
        raise ValueError("text-based linking requires both left_source and right_source.")

    left_records = load_text_records(task, left_source, contains=contains)
    right_records = load_text_records(task, right_source)
    return link_text_records(
        left_records=left_records,
        right_records=right_records,
        top_k=top_k,
    )
