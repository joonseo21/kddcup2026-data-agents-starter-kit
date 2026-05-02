"""Integration tests for AOPAgent with a live LLM.

Skipped automatically when env vars are absent or dataset is not found.
Run with:
  export MODEL_API_URL=http://localhost:11434/v1
  export MODEL_API_KEY=ollama
  export MODEL_NAME=qwen3.5:35b-a3b
  uv run pytest tests/test_aop_integration.py -v -m integration
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from data_agent_baseline.agents.aop_agent import AOPAgent, AOPAgentConfig
from data_agent_baseline.agents.model import OpenAIModelAdapter
from data_agent_baseline.aop.logical_representations import (
    COUNT_DISTINCT_LR,
    COUNT_LR,
    EXTRACT_LR,
    GROUPBY_SUM_LR,
    JOIN_SIMPLE_LR,
    RECORD_FILTER_LR,
)
from data_agent_baseline.benchmark.dataset import DABenchPublicDataset
from data_agent_baseline.benchmark.schema import AnswerTable

DATASET_ROOT = Path(__file__).resolve().parents[1] / "data" / "public" / "input"
ALL_LRS = [RECORD_FILTER_LR, JOIN_SIMPLE_LR, EXTRACT_LR, COUNT_LR, COUNT_DISTINCT_LR, GROUPBY_SUM_LR]

pytestmark = pytest.mark.integration


def _require_env(*names: str) -> None:
    missing = [n for n in names if not os.environ.get(n)]
    if missing:
        pytest.skip(f"env vars not set: {missing}")


@pytest.fixture(scope="module")
def model() -> OpenAIModelAdapter:
    _require_env("MODEL_API_URL", "MODEL_API_KEY", "MODEL_NAME")
    return OpenAIModelAdapter(
        model=os.environ["MODEL_NAME"],
        api_base=os.environ["MODEL_API_URL"],
        api_key=os.environ["MODEL_API_KEY"],
        temperature=0.0,
    )


@pytest.fixture(scope="module")
def dataset() -> DABenchPublicDataset:
    if not DATASET_ROOT.exists():
        pytest.skip(f"dataset not found: {DATASET_ROOT}")
    return DABenchPublicDataset(DATASET_ROOT)


class TestAOPAgentIntegration:
    def test_task_11_returns_valid_table(self, model: OpenAIModelAdapter, dataset: DABenchPublicDataset) -> None:
        """task_11: LINK→PLAN→EXEC 전 과정이 succeeded=True이고 유효한 AnswerTable을 반환하는지 검증."""
        task = dataset.get_task("task_11")
        agent = AOPAgent(model=model, lrs=ALL_LRS, config=AOPAgentConfig(max_steps=16))
        result = agent.run(task)

        assert result.succeeded, f"task failed: {result.failure_reason}"
        assert isinstance(result.answer, AnswerTable)
        assert len(result.answer.columns) > 0
        assert isinstance(result.answer.rows, list)
        assert all(isinstance(row, list) for row in result.answer.rows)
        assert all(len(row) == len(result.answer.columns) for row in result.answer.rows)
