from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from data_agent_baseline.benchmark.schema import PublicTask
from data_agent_baseline.tools.filesystem import load_document_text
from data_agent_baseline.tools.scan import iter_source_paths
from data_agent_baseline.tools.text_utils import split_text_chunks, tokenize


def keyword_score(query: str, content: str) -> float:
    # Score one markdown chunk against the query using token overlap and a
    # small coverage bonus so chunks matching more query terms rank higher.
    query_tokens = tokenize(query)
    if not query_tokens:
        return 0.0

    content_tokens = tokenize(content)
    if not content_tokens:
        return 0.0

    content_counts: dict[str, int] = {}
    for token in content_tokens:
        content_counts[token] = content_counts.get(token, 0) + 1

    overlap = 0.0
    unique_matches = 0
    for token in query_tokens:
        if token in content_counts:
            unique_matches += 1
            overlap += 1.0 + math.log1p(content_counts[token])

    coverage_bonus = unique_matches / len(query_tokens)
    return (overlap / math.sqrt(len(content_tokens))) + coverage_bonus

def build_markdown_database(
    task: PublicTask,
    *,
    sources: list[str] | None = None,
) -> dict[str, Any]:
    # Build a markdown-only retrieval database. Each .md file becomes one or
    # more searchable chunk records that preserve the original source path.
    records: list[dict[str, Any]] = []
    indexed_paths: list[str] = []

    for path in iter_source_paths(task, sources):
        if path.suffix.lower() != ".md":
            continue
        relative_path = path.relative_to(task.context_dir).as_posix()
        indexed_paths.append(relative_path)
        document_name = Path(relative_path).name
        text = load_document_text(task, relative_path)
        for index, chunk in enumerate(split_text_chunks(text), start=1):
            records.append(
                {
                    "path": relative_path,
                    "record_type": "markdown_chunk",
                    "record_id": f"{relative_path}#chunk:{index}",
                    "text": f"document: {document_name} | chunk: {index} | {chunk}",
                    "metadata": {"chunk_index": index, "document": document_name},
                }
            )

    return {
        "database_type": "markdown",
        "record_count": len(records),
        "indexed_paths": indexed_paths,
        "records": records,
    }


def search_keyword_database(
    *,
    query: str,
    database: dict[str, Any],
    top_k: int = 5,
) -> dict[str, Any]:
    # Search a prebuilt markdown database and return the top-scoring matches
    # together with the chunk text and source metadata.
    raw_records = database.get("records")
    if not isinstance(raw_records, list):
        raise ValueError("database.records must be a list.")

    scored_matches: list[dict[str, Any]] = []
    for raw_record in raw_records:
        if not isinstance(raw_record, dict):
            raise ValueError("Each database record must be an object.")

        text = str(raw_record.get("text", ""))
        score = keyword_score(query, text)
        if score <= 0:
            continue

        match = dict(raw_record)
        match["score"] = round(score, 6)
        match["content"] = text
        scored_matches.append(match)

    scored_matches.sort(key=lambda item: item["score"], reverse=True)
    return {
        "mode": "keyword",
        "query": query,
        "database_type": str(database.get("database_type", "markdown")),
        "record_count": int(database.get("record_count", len(raw_records))),
        "match_count": len(scored_matches),
        "matches": scored_matches[:top_k],
    }


def retrieve_by_keyword(
    task: PublicTask,
    *,
    query: str,
    sources: list[str] | None = None,
    top_k: int = 5,
) -> dict[str, Any]:
    # One-shot helper that builds the markdown database from task files and
    # immediately runs keyword retrieval over it.
    database = build_markdown_database(task, sources=sources)
    return search_keyword_database(query=query, database=database, top_k=top_k)
