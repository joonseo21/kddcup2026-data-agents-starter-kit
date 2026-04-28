"""BM25Matcher 단위 테스트."""
import pytest

from data_agent_baseline.aop.bm25_matcher import BM25Matcher
from data_agent_baseline.aop.logical_representations import (
    RECORD_FILTER_LR,
    JOIN_SIMPLE_LR,
    EXTRACT_LR,
    COUNT_LR,
    GROUPBY_SUM_LR,
)

ALL_LRS = [RECORD_FILTER_LR, JOIN_SIMPLE_LR, EXTRACT_LR, COUNT_LR, GROUPBY_SUM_LR]


@pytest.fixture
def matcher():
    return BM25Matcher(lrs=ALL_LRS)


# ──────────────────────────────────────────
# Initialization
# ──────────────────────────────────────────

class TestBM25MatcherInit:
    def test_builds_without_error(self):
        BM25Matcher(lrs=ALL_LRS)

    def test_corpus_length_equals_lr_count(self, matcher):
        assert len(matcher._corpus) == len(ALL_LRS)

    def test_idf_is_non_empty(self, matcher):
        assert len(matcher._idf) > 0

    def test_avgdl_positive(self, matcher):
        assert matcher._avgdl > 0


# ──────────────────────────────────────────
# Routing (keyword overlap based)
# ──────────────────────────────────────────

class TestBM25MatcherRouting:
    def test_filter_keywords_match_filter_lr(self, matcher):
        result = matcher.match("filter records where condition equals value")
        assert result["name"] == "RECORD_FILTER_LR"

    def test_join_keywords_match_join_lr(self, matcher):
        result = matcher.match("join two tables merge rows")
        assert result["name"] == "JOIN_SIMPLE_LR"

    def test_count_keywords_match_count_lr(self, matcher):
        result = matcher.match("count how many rows")
        assert result["name"] in ("COUNT_LR", "COUNT_DISTINCT_LR")

    def test_extract_keywords_match_extract_lr(self, matcher):
        result = matcher.match("extract select list columns fields")
        assert result["name"] == "EXTRACT_LR"

    def test_deterministic_same_query(self, matcher):
        q = "filter records where Thrombosis equals 2"
        assert matcher.match(q)["name"] == matcher.match(q)["name"]


# ──────────────────────────────────────────
# match_top_k
# ──────────────────────────────────────────

class TestBM25MatcherTopK:
    def test_returns_k_results(self, matcher):
        results = matcher.match_top_k("filter records where field equals value", k=3)
        assert len(results) == 3

    def test_sorted_descending_by_score(self, matcher):
        results = matcher.match_top_k("filter records", k=len(ALL_LRS))
        scores = [r["score"] for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_k_larger_than_lrs_clamped(self, matcher):
        results = matcher.match_top_k("any query", k=100)
        assert len(results) == len(ALL_LRS)

    def test_each_result_has_lr_and_score(self, matcher):
        for r in matcher.match_top_k("join tables", k=2):
            assert "lr" in r and "score" in r
            assert isinstance(r["score"], float)


# ──────────────────────────────────────────
# match_with_score
# ──────────────────────────────────────────

class TestBM25MatcherWithScore:
    def test_returns_lr_dict_and_float(self, matcher):
        lr, score = matcher.match_with_score("filter records")
        assert isinstance(lr, dict)
        assert isinstance(score, float)

    def test_score_is_non_negative(self, matcher):
        _, score = matcher.match_with_score("count rows in table")
        assert score >= 0.0


# ──────────────────────────────────────────
# Edge cases
# ──────────────────────────────────────────

class TestBM25MatcherEdgeCases:
    def test_empty_lr_list_top_k_returns_empty(self):
        m = BM25Matcher(lrs=[])
        assert m.match_top_k("any query", k=3) == []

    def test_empty_lr_list_match_raises(self):
        m = BM25Matcher(lrs=[])
        with pytest.raises(ValueError):
            m.match("any query")

    def test_single_lr_always_returned(self):
        m = BM25Matcher(lrs=[RECORD_FILTER_LR])
        assert m.match("completely unrelated weather query")["name"] == "RECORD_FILTER_LR"

    def test_unknown_terms_score_zero(self):
        m = BM25Matcher(lrs=[RECORD_FILTER_LR])
        _, score = m.match_with_score("xyzzy frobnicator quux")
        assert score == 0.0

    def test_custom_k1_b_accepted(self):
        BM25Matcher(lrs=ALL_LRS, k1=1.2, b=0.5)
