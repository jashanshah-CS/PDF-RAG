"""Hybrid vector and BM25 retrieval using reciprocal rank fusion."""

from collections.abc import Iterable

from src.rag_project.bm25 import BM25Index
from src.rag_project.embeddings import EmbeddedChunk, TextEncoder
from src.rag_project.search import SearchResult, vector_search


DEFAULT_CANDIDATE_COUNT = 20
DEFAULT_RRF_K = 60
DEFAULT_HYBRID_RRF_K = 5
DEFAULT_VECTOR_WEIGHT = 1.0
DEFAULT_KEYWORD_WEIGHT = 3.0


def _chunk_key(
    result: SearchResult,
) -> tuple[str, int, int | None, int | None, str | None]:
    chunk = result.embedded_chunk.chunk
    return (
        chunk.document_id,
        chunk.chunk_number,
        chunk.location.page_number,
        chunk.location.row_number,
        chunk.location.url,
    )


def reciprocal_rank_fusion(
    rankings: Iterable[list[SearchResult]],
    *,
    top_k: int = 5,
    rrf_k: int = DEFAULT_RRF_K,
    weights: Iterable[float] | None = None,
) -> list[SearchResult]:
    """Fuse ranked lists without assuming their score scales are comparable."""
    if top_k <= 0:
        raise ValueError("The number of results must be greater than zero.")
    if rrf_k <= 0:
        raise ValueError("RRF k must be greater than zero.")
    ranking_lists = list(rankings)
    ranking_weights = (
        list(weights) if weights is not None else [1.0] * len(ranking_lists)
    )
    if len(ranking_weights) != len(ranking_lists):
        raise ValueError("RRF needs one weight for every ranking.")
    if any(weight <= 0 for weight in ranking_weights):
        raise ValueError("RRF weights must be greater than zero.")

    scores: dict[
        tuple[str, int, int | None, int | None, str | None], float
    ] = {}
    best_ranks: dict[
        tuple[str, int, int | None, int | None, str | None], int
    ] = {}
    first_seen: dict[
        tuple[str, int, int | None, int | None, str | None], int
    ] = {}
    results_by_key: dict[
        tuple[str, int, int | None, int | None, str | None], SearchResult
    ] = {}

    encounter = 0
    for ranking, weight in zip(ranking_lists, ranking_weights, strict=True):
        seen_in_ranking = set()
        for rank, result in enumerate(ranking, start=1):
            key = _chunk_key(result)
            if key in seen_in_ranking:
                continue
            seen_in_ranking.add(key)
            scores[key] = scores.get(key, 0.0) + weight / (rrf_k + rank)
            best_ranks[key] = min(best_ranks.get(key, rank), rank)
            if key not in first_seen:
                first_seen[key] = encounter
                results_by_key[key] = result
                encounter += 1

    ranked_keys = sorted(
        scores,
        key=lambda key: (-scores[key], best_ranks[key], first_seen[key]),
    )
    return [
        SearchResult(results_by_key[key].embedded_chunk, scores[key])
        for key in ranked_keys[:top_k]
    ]


class HybridRetriever:
    """Reusable vector/BM25 retriever with a prebuilt keyword index."""

    def __init__(
        self,
        chunks: Iterable[EmbeddedChunk],
        model: TextEncoder,
        *,
        candidate_k: int = DEFAULT_CANDIDATE_COUNT,
        rrf_k: int = DEFAULT_HYBRID_RRF_K,
        vector_weight: float = DEFAULT_VECTOR_WEIGHT,
        keyword_weight: float = DEFAULT_KEYWORD_WEIGHT,
    ) -> None:
        if candidate_k <= 0:
            raise ValueError("The candidate count must be greater than zero.")
        if rrf_k <= 0:
            raise ValueError("RRF k must be greater than zero.")
        if vector_weight <= 0 or keyword_weight <= 0:
            raise ValueError("Retriever weights must be greater than zero.")
        self._chunks = list(chunks)
        self._model = model
        self._candidate_k = candidate_k
        self._rrf_k = rrf_k
        self._weights = (vector_weight, keyword_weight)
        self._bm25 = BM25Index(self._chunks)

    @property
    def configuration(self) -> dict[str, int | float]:
        """Return the frozen parameters needed to reproduce this ranking."""
        return {
            "candidate_k": self._candidate_k,
            "rrf_k": self._rrf_k,
            "vector_weight": self._weights[0],
            "keyword_weight": self._weights[1],
        }

    def search(self, query: str, *, top_k: int = 5) -> list[SearchResult]:
        """Search vector and keyword indexes, then fuse their candidate ranks."""
        if not query.strip():
            raise ValueError("Enter a question before searching.")
        if top_k <= 0:
            raise ValueError("The number of results must be greater than zero.")
        if top_k > self._candidate_k:
            raise ValueError("The result count cannot exceed the candidate count.")
        if not self._chunks:
            return []

        vector_results = vector_search(
            query,
            self._chunks,
            self._model,
            top_k=self._candidate_k,
        )
        keyword_results = self._bm25.search(query, top_k=self._candidate_k)
        return reciprocal_rank_fusion(
            [vector_results, keyword_results],
            top_k=top_k,
            rrf_k=self._rrf_k,
            weights=self._weights,
        )
