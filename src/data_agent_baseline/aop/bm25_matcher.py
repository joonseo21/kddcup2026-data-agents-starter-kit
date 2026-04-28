# aop/bm25_matcher.py
"""BM25Matcher: 자연어 쿼리를 LR에 매핑 (BM25 기반, 임베딩 불필요)."""
from __future__ import annotations
import math
from collections import Counter


def _tokenize(text: str) -> list[str]:
    return text.lower().split()


class BM25Matcher:
    """
    BM25 기반 LR 매처. BQMatcher와 동일한 인터페이스.

    LR dict 스키마:
        name: str
        description: str
        example_questions: list[str]
        operators: list[str]
        output_type: str
    """

    def __init__(
        self,
        lrs: list[dict],
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        self._lrs = lrs
        self._k1 = k1
        self._b = b
        self._corpus = self._build_corpus()
        self._idf = self._compute_idf()
        self._avgdl = (
            sum(len(doc) for doc in self._corpus) / len(self._corpus)
            if self._corpus
            else 1.0
        )

    # ------------------------------------------------------------------
    # Public API (same as BQMatcher)
    # ------------------------------------------------------------------

    def match(self, query: str) -> dict:
        results = self.match_top_k(query, k=1)
        if not results:
            raise ValueError("LR 목록이 비어있습니다.")
        return results[0]["lr"]

    def match_top_k(self, query: str, k: int) -> list[dict]:
        if not self._lrs:
            return []
        scores = [self._score(query, i) for i in range(len(self._corpus))]
        top_k = min(k, len(self._lrs))
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [{"lr": self._lrs[i], "score": scores[i]} for i in ranked]

    def match_with_score(self, query: str) -> tuple[dict, float]:
        results = self.match_top_k(query, k=1)
        if not results:
            raise ValueError("LR 목록이 비어있습니다.")
        return results[0]["lr"], results[0]["score"]

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _build_corpus(self) -> list[list[str]]:
        docs = []
        for lr in self._lrs:
            text = lr.get("description", "") + " " + " ".join(lr.get("example_questions", []))
            docs.append(_tokenize(text))
        return docs

    def _compute_idf(self) -> dict[str, float]:
        n = len(self._corpus)
        df: Counter[str] = Counter()
        for doc in self._corpus:
            for term in set(doc):
                df[term] += 1
        return {
            term: math.log((n - freq + 0.5) / (freq + 0.5) + 1.0)
            for term, freq in df.items()
        }

    def _score(self, query: str, doc_idx: int) -> float:
        doc = self._corpus[doc_idx]
        dl = len(doc)
        tf_map = Counter(doc)
        score = 0.0
        for term in _tokenize(query):
            idf = self._idf.get(term, 0.0)
            tf = tf_map.get(term, 0)
            numerator = tf * (self._k1 + 1)
            denominator = tf + self._k1 * (1 - self._b + self._b * dl / self._avgdl)
            score += idf * numerator / denominator if denominator else 0.0
        return score
