"""
EmbeddingModel 구현체 단위 테스트.

MockEmbedder의 결정론적 동작을 검증.
SentenceTransformerEmbedder는 패키지 미설치 환경에서 skip.
"""

import numpy as np
import pytest

from data_agent_baseline.aop.embedding import MockEmbedder


class TestMockEmbedder:
    @pytest.fixture
    def embedder(self):
        return MockEmbedder()

    def test_embed_returns_correct_shape(self, embedder):
        result = embedder.embed(["hello world", "foo bar"])
        assert result.shape == (2, MockEmbedder.DIM)

    def test_embed_single_text_returns_2d_array(self, embedder):
        result = embedder.embed(["one text"])
        assert result.ndim == 2
        assert result.shape[0] == 1

    def test_output_is_l2_normalized(self, embedder):
        result = embedder.embed(["filter where condition equals value"])
        norms = np.linalg.norm(result, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-6)

    def test_filter_keywords_activate_same_dimension(self, embedder):
        """filter 관련 텍스트끼리는 같은 차원이 활성화되어야 함"""
        v1 = embedder.embed(["filter records where field equals value"])[0]
        v2 = embedder.embed(["condition equals something"])[0]
        # 코사인 유사도가 양수여야 함
        similarity = float(v1 @ v2)
        assert similarity > 0

    def test_different_keyword_groups_have_low_similarity(self, embedder):
        """서로 다른 연산 유형은 낮은 유사도를 가져야 함"""
        filter_vec = embedder.embed(["filter records where condition equals value"])[0]
        count_vec = embedder.embed(["count how many distinct rows"])[0]
        similarity = float(filter_vec @ count_vec)
        # 완전히 다른 키워드 그룹이면 유사도가 낮아야 함
        assert similarity < 0.9

    def test_identical_texts_produce_identical_vectors(self, embedder):
        text = "join tables on shared key field"
        v1 = embedder.embed([text])[0]
        v2 = embedder.embed([text])[0]
        np.testing.assert_array_equal(v1, v2)

    def test_empty_text_does_not_crash(self, embedder):
        result = embedder.embed([""])
        assert result.shape == (1, MockEmbedder.DIM)


class TestSentenceTransformerEmbedder:
    def test_import_error_raised_when_not_installed(self):
        """sentence-transformers 미설치 시 ImportError with 안내 메시지"""
        import sys
        # sentence_transformers가 실제로 설치되어 있으면 skip
        if "sentence_transformers" in sys.modules or _can_import("sentence_transformers"):
            pytest.skip("sentence-transformers is installed")

        from data_agent_baseline.aop.embedding import SentenceTransformerEmbedder
        with pytest.raises(ImportError, match="sentence-transformers"):
            SentenceTransformerEmbedder()


def _can_import(module: str) -> bool:
    try:
        __import__(module)
        return True
    except ImportError:
        return False
