"""임베딩 모델 인터페이스 및 구현체."""

from typing import Protocol
import numpy as np


class EmbeddingModel(Protocol):
    """임베딩 모델 프로토콜. 어떤 구현체든 이 인터페이스를 따라야 함."""

    def embed(self, texts: list[str]) -> np.ndarray:
        """texts를 임베딩. 반환: shape (len(texts), dim) float32 배열"""
        ...


class MockEmbedder:
    """
    테스트용 결정론적 임베딩 구현체.
    실제 모델 없이 BQMatcher 로직을 검증하기 위해 사용.

    동작 방식: 각 텍스트에서 키워드를 찾아 미리 정해진 차원에 1.0을 세팅.
    같은 키워드를 포함한 텍스트끼리는 높은 코사인 유사도를 가짐.
    """

    # 키워드 → 차원 인덱스 매핑 (총 32차원)
    KEYWORD_DIMS: dict[str, int] = {
        "filter": 0, "where": 0, "equals": 0, "condition": 0,
        "join": 1, "merge": 1, "combine": 1,
        "extract": 2, "list": 2, "columns": 2, "select": 2,
        "count": 3, "how many": 3, "number of": 3,
        "group": 4, "sum": 4, "total": 4, "aggregate": 4,
        "numeric": 5, "greater": 5, "less": 5, "comparison": 5,
        "distinct": 6, "unique": 6,
        "records": 7, "rows": 7, "table": 7,
    }
    DIM = 32

    def embed(self, texts: list[str]) -> np.ndarray:
        vectors = np.zeros((len(texts), self.DIM), dtype=np.float32)
        for i, text in enumerate(texts):
            lower = text.lower()
            for keyword, dim in self.KEYWORD_DIMS.items():
                if keyword in lower:
                    vectors[i, dim] += 1.0
            # 전혀 매칭 없으면 마지막 차원에 작은 값
            if vectors[i].sum() == 0:
                vectors[i, -1] = 0.1
        # L2 정규화
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        return vectors / norms


class SentenceTransformerEmbedder:
    """
    sentence-transformers 기반 실제 임베딩 구현체.
    평가 환경에서 인터넷 없이 로컬로 실행됨.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(model_name)
        except ImportError as e:
            raise ImportError(
                "sentence-transformers 패키지가 필요합니다: "
                "uv pip install sentence-transformers"
            ) from e

    def embed(self, texts: list[str]) -> np.ndarray:
        return self._model.encode(texts, normalize_embeddings=True)
