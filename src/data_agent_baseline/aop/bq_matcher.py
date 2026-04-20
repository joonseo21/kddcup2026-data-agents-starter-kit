"""BQMatcher: 자연어 쿼리를 가장 유사한 LR(Logical Representation)에 매핑."""

import numpy as np
from .embedding import EmbeddingModel


class BQMatcher:
    """
    LR 목록과 임베딩 모델을 받아, 쿼리와 가장 유사한 LR을 반환.

    각 LR은 다음 키를 가진 dict:
        name: str
        description: str
        example_questions: list[str]
        operators: list[str]
        output_type: str

    유사도 계산: 쿼리 벡터와 각 LR의 "앵커 벡터"(description + example_questions 평균) 사이의 코사인 유사도.
    """

    def __init__(self, lrs: list[dict], embedder: EmbeddingModel):
        self._lrs = lrs
        self._embedder = embedder
        self._lr_vectors: np.ndarray = self._build_lr_vectors()

    def _build_lr_vectors(self) -> np.ndarray:
        """각 LR의 앵커 벡터 계산. init 시 한 번만 수행."""
        if not self._lrs:
            return np.empty((0,), dtype=np.float32)

        vectors = []
        for lr in self._lrs:
            texts = [lr["description"]] + lr.get("example_questions", [])
            embedded = self._embedder.embed(texts)  # (n_texts, dim)
            vectors.append(embedded.mean(axis=0))   # (dim,) 평균
        arr = np.array(vectors, dtype=np.float32)   # (n_lrs, dim)
        # L2 정규화
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        return arr / norms

    def match(self, query: str) -> dict:
        """가장 유사한 LR 반환. LR이 없으면 ValueError."""
        results = self.match_top_k(query, k=1)
        if not results:
            raise ValueError("LR 목록이 비어있습니다.")
        return results[0]["lr"]

    def match_top_k(self, query: str, k: int) -> list[dict]:
        """
        상위 k개 LR을 유사도 내림차순으로 반환.
        반환 형식: [{"lr": <LR dict>, "score": <float>}, ...]
        """
        if not self._lrs:
            return []

        query_vec = self._embedder.embed([query])[0]  # (dim,)
        # 코사인 유사도 = 정규화된 벡터의 내적
        scores = self._lr_vectors @ query_vec          # (n_lrs,)

        top_k = min(k, len(self._lrs))
        top_indices = np.argsort(scores)[::-1][:top_k]

        return [
            {"lr": self._lrs[i], "score": float(scores[i])}
            for i in top_indices
        ]

    def match_with_score(self, query: str) -> tuple[dict, float]:
        """가장 유사한 LR과 유사도 점수를 함께 반환."""
        results = self.match_top_k(query, k=1)
        if not results:
            raise ValueError("LR 목록이 비어있습니다.")
        return results[0]["lr"], results[0]["score"]
