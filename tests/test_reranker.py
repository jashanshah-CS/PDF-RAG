from datetime import UTC, datetime

import numpy as np
import pytest

from src.rag_project.documents import DocumentChunk, SourceLocation, SourceType
from src.rag_project.embeddings import EmbeddedChunk
from src.rag_project.reranker import RerankedRetriever, rerank_results
from src.rag_project.search import SearchResult


class QueryModel:
    def encode(self, sentences, **kwargs):
        vectors = []
        for sentence in sentences:
            if "semantic" in sentence.casefold() or sentence == "ZX-42 semantic":
                vectors.append([1.0, 0.0])
            else:
                vectors.append([0.0, 1.0])
        return np.array(vectors, dtype=np.float32)


def result(
    source: str,
    text: str,
    vector: tuple[float, ...],
    score: float,
) -> SearchResult:
    chunk = DocumentChunk(
        document_id=f"pdf-{source}",
        source_type=SourceType.PDF,
        source_name=source,
        chunk_number=1,
        text=text,
        location=SourceLocation(page_number=1),
        added_at=datetime(2026, 11, 14, tzinfo=UTC),
    )
    return SearchResult(EmbeddedChunk(chunk, vector), score)


def test_reranker_promotes_candidate_with_exact_focused_text() -> None:
    generic = result(
        "generic.pdf", "General deployment information.", (1.0, 0.0), 0.9
    )
    exact = result(
        "exact.pdf", "The deployment identifier is ZX-42.", (0.0, 1.0), 0.8
    )

    reranked = rerank_results(
        "ZX-42 semantic",
        [generic, exact],
        QueryModel(),
        top_k=2,
    )

    assert reranked[0].embedded_chunk.chunk.source_name == "exact.pdf"


def test_reranker_preserves_windowed_text_and_metadata() -> None:
    focused = result("policy.pdf", "Target sentence.", (1.0, 0.0), 0.7)
    focused_chunk = focused.embedded_chunk.chunk
    focused = SearchResult(
        EmbeddedChunk(
            DocumentChunk(
                **{
                    **focused_chunk.__dict__,
                    "metadata": {"sentence_window": True},
                }
            ),
            focused.embedded_chunk.embedding,
        ),
        focused.score,
    )

    reranked = rerank_results("target semantic", [focused], QueryModel())

    assert reranked[0].embedded_chunk.chunk.text == "Target sentence."
    assert reranked[0].embedded_chunk.chunk.metadata["sentence_window"] is True


def test_reranker_returns_empty_without_candidates() -> None:
    assert rerank_results("question", [], QueryModel()) == []


def test_reranker_validates_query_limits_and_weights() -> None:
    with pytest.raises(ValueError, match="Enter a question"):
        rerank_results(" ", [], QueryModel())
    with pytest.raises(ValueError, match="greater than zero"):
        rerank_results("question", [], QueryModel(), top_k=0)
    with pytest.raises(ValueError, match="three positive"):
        rerank_results("question", [], QueryModel(), weights=(1.0, 0.0, 1.0))

    retriever = RerankedRetriever([], QueryModel(), rerank_pool_size=4)
    with pytest.raises(ValueError, match="cannot exceed"):
        retriever.search("question", top_k=5)


def test_configuration_is_reproducible() -> None:
    retriever = RerankedRetriever([], QueryModel())

    assert retriever.configuration["rerank_pool_size"] == 5
    assert retriever.configuration["rerank_rrf_k"] == 5
    assert retriever.configuration["rerank_original_weight"] == 3.0
    assert retriever.configuration["rerank_vector_weight"] == 1.0
    assert retriever.configuration["rerank_keyword_weight"] == 2.0
