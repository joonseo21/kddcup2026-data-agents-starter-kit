"""
BQMatcher 단위 테스트.

MockEmbedder를 주입해 실제 임베딩 모델 없이 BQMatcher 라우팅 로직을 검증.
"""

import numpy as np
import pytest

from data_agent_baseline.aop.embedding import MockEmbedder
from data_agent_baseline.aop.bq_matcher import BQMatcher
from data_agent_baseline.aop.logical_representations import (
    RECORD_FILTER_LR,
    JOIN_SIMPLE_LR,
    EXTRACT_LR,
    COUNT_LR,
    GROUPBY_SUM_LR,
)

ALL_LRS = [RECORD_FILTER_LR, JOIN_SIMPLE_LR, EXTRACT_LR, COUNT_LR, GROUPBY_SUM_LR]


@pytest.fixture
def embedder():
    return MockEmbedder()


@pytest.fixture
def matcher(embedder):
    return BQMatcher(lrs=ALL_LRS, embedder=embedder)


class TestBQMatcherInit:
    def test_matcher_builds_without_error(self, embedder):
        matcher = BQMatcher(lrs=ALL_LRS, embedder=embedder)
        assert matcher is not None

    def test_lr_vectors_shape(self, matcher):
        """LR 수만큼 벡터가 생성되어야 함"""
        assert matcher._lr_vectors.shape[0] == len(ALL_LRS)

    def test_lr_vectors_are_normalized(self, matcher):
        """각 LR 벡터의 L2 norm은 1이어야 함"""
        norms = np.linalg.norm(matcher._lr_vectors, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-6)


class TestBQMatcherRouting:
    """MockEmbedder 키워드 기반 라우팅이 올바르게 동작하는지 검증."""

    def test_filter_query_matches_filter_lr(self, matcher):
        result = matcher.match("filter records where condition equals value")
        assert result["name"] == "RECORD_FILTER_LR"

    def test_join_query_matches_join_lr(self, matcher):
        result = matcher.match("join two tables and merge records")
        assert result["name"] == "JOIN_SIMPLE_LR"

    def test_count_query_matches_count_lr(self, matcher):
        result = matcher.match("count how many rows are in the table")
        assert result["name"] in ("COUNT_LR", "COUNT_DISTINCT_LR")

    def test_extract_query_matches_extract_lr(self, matcher):
        result = matcher.match("list columns and select extract fields")
        assert result["name"] == "EXTRACT_LR"


class TestBQMatcherTopK:
    def test_top_k_returns_k_results(self, matcher):
        results = matcher.match_top_k("filter records where field equals value", k=3)
        assert len(results) == 3

    def test_top_k_sorted_by_score_descending(self, matcher):
        results = matcher.match_top_k("filter records where condition", k=len(ALL_LRS))
        scores = [r["score"] for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_top_k_larger_than_lrs_clamped(self, matcher):
        results = matcher.match_top_k("any query", k=100)
        assert len(results) == len(ALL_LRS)

    def test_each_result_has_lr_and_score_keys(self, matcher):
        results = matcher.match_top_k("join tables", k=2)
        for r in results:
            assert "lr" in r
            assert "score" in r
            assert isinstance(r["score"], float)


class TestBQMatcherWithScore:
    def test_returns_lr_and_float_score(self, matcher):
        lr, score = matcher.match_with_score("filter records where field equals value")
        assert isinstance(lr, dict)
        assert isinstance(score, float)

    def test_score_between_minus_one_and_one(self, matcher):
        _, score = matcher.match_with_score("count rows in table")
        assert -1.0 <= score <= 1.0


class TestBQMatcherEdgeCases:
    def test_empty_lr_list_match_top_k_returns_empty(self, embedder):
        matcher = BQMatcher(lrs=[], embedder=embedder)
        results = matcher.match_top_k("any query", k=3)
        assert results == []

    def test_empty_lr_list_match_raises_value_error(self, embedder):
        matcher = BQMatcher(lrs=[], embedder=embedder)
        with pytest.raises(ValueError):
            matcher.match("any query")

    def test_single_lr_always_matched(self, embedder):
        matcher = BQMatcher(lrs=[RECORD_FILTER_LR], embedder=embedder)
        result = matcher.match("completely unrelated query about weather")
        assert result["name"] == "RECORD_FILTER_LR"

    def test_same_query_always_returns_same_lr(self, matcher):
        """결정론적이어야 함 — 같은 쿼리는 항상 같은 LR 반환"""
        query = "filter records where Thrombosis equals 2"
        result1 = matcher.match(query)
        result2 = matcher.match(query)
        assert result1["name"] == result2["name"]
