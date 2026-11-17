"""Optional learned reranking with a small local sentence-transformers model."""

from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from src.rag_project.config import SETTINGS
from src.rag_project.embeddings import EmbeddedChunk, TextEncoder
from src.rag_project.reranker import RerankedRetriever
from src.rag_project.search import SearchResult
from src.rag_project.sentence_window import SentenceWindowRetriever


DEFAULT_CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2"
DEFAULT_CANDIDATE_POOL_SIZE = 10


class PairScorer(Protocol):
    """The small part of sentence-transformers CrossEncoder used here."""

    def predict(
        self,
        sentences: Sequence[tuple[str, str]],
        *,
        batch_size: int,
        show_progress_bar: bool,
        convert_to_numpy: bool,
    ) -> NDArray[np.float32]: ...


def load_cross_encoder(
    model_name: str = DEFAULT_CROSS_ENCODER_MODEL,
    *,
    local_only: bool = True,
) -> PairScorer:
    """Load the optional CPU reranker, forbidding downloads by default."""
    from huggingface_hub import snapshot_download
    from sentence_transformers import CrossEncoder

    SETTINGS.model_cache.mkdir(parents=True, exist_ok=True)
    model_path = Path(model_name)
    if not model_path.exists():
        model_path = Path(
            snapshot_download(
                repo_id=model_name,
                cache_dir=SETTINGS.model_cache,
                local_files_only=local_only,
            )
        )
    return CrossEncoder(
        str(model_path),
        device="cpu",
        local_files_only=True,
    )


def cross_encoder_cache_size_bytes(
    model_name: str = DEFAULT_CROSS_ENCODER_MODEL,
    cache_folder: Path = SETTINGS.model_cache,
) -> int:
    """Return the on-disk footprint of this model's Hugging Face cache."""
    model_directory = cache_folder / f"models--{model_name.replace('/', '--')}"
    if not model_directory.exists():
        return 0
    return sum(
        path.stat().st_size
        for path in model_directory.rglob("*")
        if path.is_file()
    )


def cross_encoder_rerank(
    query: str,
    results: list[SearchResult],
    scorer: PairScorer,
    *,
    top_k: int = 5,
    batch_size: int = 16,
) -> list[SearchResult]:
    """Score each question–passage pair and return a stable descending rank."""
    if not query.strip():
        raise ValueError("Enter a question before searching.")
    if top_k <= 0 or batch_size <= 0:
        raise ValueError("Result count and batch size must be greater than zero.")
    if not results:
        return []
    pairs = [
        (query.strip(), result.embedded_chunk.chunk.text) for result in results
    ]
    scores = np.asarray(
        scorer.predict(
            pairs,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
    ).reshape(-1)
    if len(scores) != len(results) or not np.isfinite(scores).all():
        raise ValueError("The cross-encoder returned invalid relevance scores.")
    reranked = [
        SearchResult(result.embedded_chunk, float(scores[index]))
        for index, result in enumerate(results)
    ]
    return [
        item
        for _, item in sorted(
            enumerate(reranked),
            key=lambda pair: (-pair[1].score, pair[0]),
        )[: min(top_k, len(reranked))]
    ]


class CrossEncoderRetriever:
    """Sentence-window candidates followed by optional learned reranking."""

    def __init__(
        self,
        chunks: list[EmbeddedChunk],
        embedding_model: TextEncoder,
        scorer: PairScorer | None,
        *,
        model_name: str = DEFAULT_CROSS_ENCODER_MODEL,
        candidate_pool_size: int = DEFAULT_CANDIDATE_POOL_SIZE,
        batch_size: int = 16,
        fallback_on_error: bool = True,
    ) -> None:
        if candidate_pool_size <= 0 or batch_size <= 0:
            raise ValueError("Candidate pool and batch size must be greater than zero.")
        self._scorer = scorer
        self._model_name = model_name
        self._pool_size = candidate_pool_size
        self._batch_size = batch_size
        self._fallback_on_error = fallback_on_error
        self._window_retriever = SentenceWindowRetriever(chunks, embedding_model)
        self._fallback = RerankedRetriever(chunks, embedding_model)

    @property
    def configuration(self) -> dict[str, int | float | str | bool]:
        return {
            **self._window_retriever.configuration,
            "cross_encoder_model": self._model_name,
            "cross_encoder_candidate_pool_size": self._pool_size,
            "cross_encoder_batch_size": self._batch_size,
            "cross_encoder_device": "cpu",
            "cross_encoder_available": self._scorer is not None,
            "cross_encoder_fallback_on_error": self._fallback_on_error,
            "cross_encoder_cache_bytes": cross_encoder_cache_size_bytes(
                self._model_name
            ),
        }

    def search(self, query: str, *, top_k: int = 5) -> list[SearchResult]:
        if top_k <= 0:
            raise ValueError("The number of results must be greater than zero.")
        if top_k > self._pool_size:
            raise ValueError("The result count cannot exceed the candidate pool size.")
        if self._scorer is None:
            if self._fallback_on_error:
                return self._fallback.search(query, top_k=top_k)
            raise RuntimeError("The optional cross-encoder model is unavailable.")
        try:
            candidates = self._window_retriever.search(
                query, top_k=self._pool_size
            )
            return cross_encoder_rerank(
                query,
                candidates,
                self._scorer,
                top_k=top_k,
                batch_size=self._batch_size,
            )
        except Exception:
            if not self._fallback_on_error:
                raise
            return self._fallback.search(query, top_k=top_k)
