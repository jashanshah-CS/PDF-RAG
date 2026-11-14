from datetime import UTC, datetime

import numpy as np
import pytest

from src.rag_project.documents import DocumentChunk, SourceLocation, SourceType
from src.rag_project.embeddings import EmbeddedChunk
from src.rag_project.hybrid import HybridRetriever, reciprocal_rank_fusion
from src.rag_project.search import SearchResult


class QueryModel:
    def encode(self, sentences, **kwargs):
        return np.array([[1.0, 0.0]], dtype=np.float32)


def embedded(
    text: str,
    vector: tuple[float, ...],
    *,
    source: str,
    chunk_number: int = 1,
) -> EmbeddedChunk:
    return EmbeddedChunk(
        DocumentChunk(
            document_id=f"pdf-{source}",
            source_type=SourceType.PDF,
            source_name=source,
            chunk_number=chunk_number,
            text=text,
            location=SourceLocation(page_number=1),
            added_at=datetime(2026, 11, 14, tzinfo=UTC),
        ),
        vector,
    )


def search_result(item: EmbeddedChunk, score: float) -> SearchResult:
    return SearchResult(item, score)


def test_fusion_rewards_chunks_found_by_both_retrievers() -> None:
    shared = embedded("shared", (1.0, 0.0), source="shared.pdf")
    vector_only = embedded("vector", (1.0, 0.0), source="vector.pdf")
    keyword_only = embedded("keyword", (0.0, 1.0), source="keyword.pdf")

    results = reciprocal_rank_fusion(
        [
            [search_result(vector_only, 0.99), search_result(shared, 0.5)],
            [search_result(keyword_only, 100.0), search_result(shared, 2.0)],
        ]
    )

    assert results[0].embedded_chunk.chunk.source_name == "shared.pdf"


def test_fusion_uses_ranks_not_incompatible_raw_scores() -> None:
    first = embedded("first", (1.0, 0.0), source="first.pdf")
    second = embedded("second", (0.0, 1.0), source="second.pdf")

    results = reciprocal_rank_fusion(
        [
            [search_result(first, 0.01), search_result(second, 0.001)],
            [search_result(first, 1.0), search_result(second, 10_000.0)],
        ]
    )

    assert results[0].embedded_chunk.chunk.source_name == "first.pdf"


def test_fusion_supports_explicit_retriever_weights() -> None:
    vector_first = embedded("vector", (1.0, 0.0), source="vector.pdf")
    keyword_first = embedded("keyword", (0.0, 1.0), source="keyword.pdf")

    results = reciprocal_rank_fusion(
        [
            [search_result(vector_first, 0.9), search_result(keyword_first, 0.8)],
            [search_result(keyword_first, 9.0), search_result(vector_first, 8.0)],
        ],
        weights=(1.0, 2.0),
    )

    assert results[0].embedded_chunk.chunk.source_name == "keyword.pdf"


def test_fusion_deduplicates_a_chunk_within_and_across_rankings() -> None:
    item = embedded("same", (1.0, 0.0), source="same.pdf")

    results = reciprocal_rank_fusion(
        [
            [search_result(item, 1.0), search_result(item, 0.5)],
            [search_result(item, 4.0)],
        ]
    )

    assert len(results) == 1
    assert results[0].score == pytest.approx((1 / 61) + (1 / 61))


def test_fusion_has_stable_first_seen_tie_breaking() -> None:
    first = embedded("first", (1.0, 0.0), source="first.pdf")
    second = embedded("second", (0.0, 1.0), source="second.pdf")

    results = reciprocal_rank_fusion(
        [[search_result(first, 1.0)], [search_result(second, 1.0)]]
    )

    assert [result.embedded_chunk.chunk.source_name for result in results] == [
        "first.pdf",
        "second.pdf",
    ]


def test_hybrid_retriever_includes_vector_and_exact_term_candidates() -> None:
    semantic = embedded(
        "Annual leave allowance.", (1.0, 0.0), source="semantic.pdf"
    )
    exact = embedded(
        "Deployment identifier ZX-42.", (0.0, 1.0), source="exact.pdf"
    )
    noise = embedded("Office furniture.", (-1.0, 0.0), source="noise.pdf")
    retriever = HybridRetriever([semantic, exact, noise], QueryModel(), candidate_k=3)

    results = retriever.search("holiday policy ZX-42", top_k=2)

    assert {result.embedded_chunk.chunk.source_name for result in results} == {
        "semantic.pdf",
        "exact.pdf",
    }
    assert retriever.configuration == {
        "candidate_k": 3,
        "rrf_k": 5,
        "vector_weight": 1.0,
        "keyword_weight": 3.0,
    }


def test_hybrid_retriever_validates_limits_and_blank_queries() -> None:
    retriever = HybridRetriever([], QueryModel(), candidate_k=2)

    with pytest.raises(ValueError, match="Enter a question"):
        retriever.search(" ")
    with pytest.raises(ValueError, match="cannot exceed"):
        retriever.search("policy", top_k=3)
    with pytest.raises(ValueError, match="greater than zero"):
        reciprocal_rank_fusion([], top_k=0)
    with pytest.raises(ValueError, match="RRF k"):
        reciprocal_rank_fusion([], rrf_k=0)
    with pytest.raises(ValueError, match="one weight"):
        reciprocal_rank_fusion([[]], weights=[])
    with pytest.raises(ValueError, match="weights"):
        reciprocal_rank_fusion([[]], weights=[0])
