"""Offline second-stage reranking for focused hybrid retrieval candidates."""

from src.rag_project.bm25 import BM25Index
from src.rag_project.embeddings import EmbeddedChunk, TextEncoder
from src.rag_project.hybrid import reciprocal_rank_fusion
from src.rag_project.search import SearchResult, vector_search
from src.rag_project.sentence_window import SentenceWindowRetriever


DEFAULT_RERANK_POOL_SIZE = 5
DEFAULT_RERANK_RRF_K = 5
DEFAULT_ORIGINAL_WEIGHT = 3.0
DEFAULT_VECTOR_WEIGHT = 1.0
DEFAULT_KEYWORD_WEIGHT = 2.0


def rerank_results(
    query: str,
    results: list[SearchResult],
    model: TextEncoder,
    *,
    top_k: int = 5,
    rrf_k: int = DEFAULT_RERANK_RRF_K,
    weights: tuple[float, float, float] = (
        DEFAULT_ORIGINAL_WEIGHT,
        DEFAULT_VECTOR_WEIGHT,
        DEFAULT_KEYWORD_WEIGHT,
    ),
) -> list[SearchResult]:
    """Rerank candidates from original, vector, and focused-text BM25 ranks."""
    if not query.strip():
        raise ValueError("Enter a question before searching.")
    if top_k <= 0:
        raise ValueError("The number of results must be greater than zero.")
    if rrf_k <= 0:
        raise ValueError("Reranker RRF k must be greater than zero.")
    if len(weights) != 3 or any(weight <= 0 for weight in weights):
        raise ValueError("Reranker weights must contain three positive values.")
    if not results:
        return []

    candidates = [result.embedded_chunk for result in results]
    vector_results = vector_search(
        query,
        candidates,
        model,
        top_k=len(candidates),
    )
    keyword_results = BM25Index(candidates).search(
        query,
        top_k=len(candidates),
    )
    return reciprocal_rank_fusion(
        [results, vector_results, keyword_results],
        top_k=min(top_k, len(results)),
        rrf_k=rrf_k,
        weights=weights,
    )


class RerankedRetriever:
    """Sentence-window retrieval followed by a deterministic local reranker."""

    def __init__(
        self,
        chunks: list[EmbeddedChunk],
        model: TextEncoder,
        *,
        rerank_pool_size: int = DEFAULT_RERANK_POOL_SIZE,
        rrf_k: int = DEFAULT_RERANK_RRF_K,
        original_weight: float = DEFAULT_ORIGINAL_WEIGHT,
        vector_weight: float = DEFAULT_VECTOR_WEIGHT,
        keyword_weight: float = DEFAULT_KEYWORD_WEIGHT,
    ) -> None:
        if rerank_pool_size <= 0:
            raise ValueError("The rerank pool size must be greater than zero.")
        if rrf_k <= 0:
            raise ValueError("Reranker RRF k must be greater than zero.")
        weights = (original_weight, vector_weight, keyword_weight)
        if any(weight <= 0 for weight in weights):
            raise ValueError("Reranker weights must be greater than zero.")
        self._model = model
        self._pool_size = rerank_pool_size
        self._rrf_k = rrf_k
        self._weights = weights
        self._window_retriever = SentenceWindowRetriever(chunks, model)

    @property
    def configuration(self) -> dict[str, int | float | str]:
        """Return all retrieval and reranking parameters for reports."""
        return {
            **self._window_retriever.configuration,
            "rerank_pool_size": self._pool_size,
            "rerank_rrf_k": self._rrf_k,
            "rerank_original_weight": self._weights[0],
            "rerank_vector_weight": self._weights[1],
            "rerank_keyword_weight": self._weights[2],
        }

    def search(self, query: str, *, top_k: int = 5) -> list[SearchResult]:
        """Retrieve a candidate pool, focus its text, and rerank the final set."""
        if top_k <= 0:
            raise ValueError("The number of results must be greater than zero.")
        if top_k > self._pool_size:
            raise ValueError("The result count cannot exceed the rerank pool size.")
        candidates = self._window_retriever.search(
            query,
            top_k=self._pool_size,
        )
        return rerank_results(
            query,
            candidates,
            self._model,
            top_k=top_k,
            rrf_k=self._rrf_k,
            weights=self._weights,
        )
