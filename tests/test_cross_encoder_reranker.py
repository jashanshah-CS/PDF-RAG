from datetime import UTC, datetime

import numpy as np
import pytest

from src.rag_project.cross_encoder_reranker import (
    CrossEncoderRetriever,
    cross_encoder_cache_size_bytes,
    cross_encoder_rerank,
)
from src.rag_project.documents import DocumentChunk, SourceLocation, SourceType
from src.rag_project.embeddings import EmbeddedChunk
from src.rag_project.search import SearchResult


class PairScorer:
    def predict(
        self, sentences, *, batch_size, show_progress_bar, convert_to_numpy
    ):
        assert batch_size == 16
        assert show_progress_bar is False
        assert convert_to_numpy is True
        return np.array(
            [5.0 if "exact target" in passage else -1.0 for _, passage in sentences],
            dtype=np.float32,
        )


class BrokenScorer:
    def predict(self, sentences, **kwargs):
        raise RuntimeError("model failed")


class QueryModel:
    def encode(self, sentences, **kwargs):
        return np.array([[1.0, 0.0] for _ in sentences], dtype=np.float32)


def result(name: str, text: str, score: float = 0.5) -> SearchResult:
    chunk = DocumentChunk(
        document_id=name,
        source_type=SourceType.PDF,
        source_name=name,
        chunk_number=1,
        text=text,
        location=SourceLocation(page_number=1),
        added_at=datetime(2026, 11, 17, tzinfo=UTC),
    )
    return SearchResult(EmbeddedChunk(chunk, (1.0, 0.0)), score)


def test_cross_encoder_promotes_highest_pair_score_and_preserves_metadata() -> None:
    generic = result("generic.pdf", "generic material", 0.9)
    exact = result("exact.pdf", "the exact target passage", 0.2)

    ranked = cross_encoder_rerank(
        "target question", [generic, exact], PairScorer(), top_k=2
    )

    assert [item.embedded_chunk.chunk.source_name for item in ranked] == [
        "exact.pdf", "generic.pdf"
    ]
    assert ranked[0].score == 5.0


def test_cross_encoder_validates_inputs_and_scores() -> None:
    with pytest.raises(ValueError, match="Enter a question"):
        cross_encoder_rerank(" ", [], PairScorer())
    with pytest.raises(ValueError, match="greater than zero"):
        cross_encoder_rerank("question", [], PairScorer(), top_k=0)

    class InvalidScorer:
        def predict(self, sentences, **kwargs):
            return np.array([float("nan")], dtype=np.float32)

    with pytest.raises(ValueError, match="invalid relevance"):
        cross_encoder_rerank(
            "question", [result("one.pdf", "one")], InvalidScorer()
        )


def test_retriever_uses_deterministic_fallback_or_strict_failure() -> None:
    chunks = [result("one.pdf", "one policy").embedded_chunk]
    fallback = CrossEncoderRetriever(chunks, QueryModel(), None)
    assert len(fallback.search("policy", top_k=1)) == 1

    strict = CrossEncoderRetriever(
        chunks, QueryModel(), BrokenScorer(), fallback_on_error=False
    )
    with pytest.raises(RuntimeError, match="model failed"):
        strict.search("policy", top_k=1)


def test_cache_size_counts_only_requested_model(tmp_path) -> None:
    model = tmp_path / "models--cross-encoder--test"
    (model / "blobs").mkdir(parents=True)
    (model / "blobs" / "weights").write_bytes(b"12345")
    (tmp_path / "unrelated").write_bytes(b"ignored")

    assert cross_encoder_cache_size_bytes("cross-encoder/test", tmp_path) == 5
