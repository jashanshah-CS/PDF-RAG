from datetime import UTC, datetime

import numpy as np
import pytest

from src.rag_project.documents import DocumentChunk, SourceLocation, SourceType
from src.rag_project.embeddings import EmbeddedChunk
from src.rag_project.search import SearchResult
from src.rag_project.sentence_window import (
    SentenceWindowRetriever,
    expand_sentence_windows,
    split_sentences,
)


class WindowModel:
    def encode(self, sentences, **kwargs):
        return np.array(
            [
                [1.0, 0.0]
                if "target" in sentence.casefold() or "meaning" in sentence.casefold()
                else [0.0, 1.0]
                for sentence in sentences
            ],
            dtype=np.float32,
        )


def result(text: str, *, score: float = 0.8) -> SearchResult:
    chunk = DocumentChunk(
        document_id="pdf-policy",
        source_type=SourceType.PDF,
        source_name="policy.pdf",
        chunk_number=2,
        text=text,
        location=SourceLocation(page_number=4),
        added_at=datetime(2026, 11, 14, tzinfo=UTC),
        metadata={"source_key": "pdf:policy.pdf"},
    )
    return SearchResult(EmbeddedChunk(chunk, (0.5, 0.5)), score)


def test_splits_normalized_sentences_and_keeps_punctuation() -> None:
    assert split_sentences("First sentence.\nSecond sentence!  Third?") == [
        "First sentence.",
        "Second sentence!",
        "Third?",
    ]


def test_expands_around_the_most_relevant_sentence() -> None:
    original = result(
        "Unrelated opening. Previous context. Target rule is here. "
        "Following condition. Unrelated ending."
    )

    expanded = expand_sentence_windows(
        "target question", [original], WindowModel(), radius=1, anchor_count=1
    )

    assert expanded[0].embedded_chunk.chunk.text == (
        "Previous context. Target rule is here. Following condition."
    )


def test_window_preserves_citation_identity_embedding_score_and_metadata() -> None:
    original = result("Opening. Target evidence. Closing.", score=0.42)

    expanded = expand_sentence_windows(
        "target", [original], WindowModel(), anchor_count=1
    )[0]

    assert expanded.score == 0.42
    assert expanded.embedded_chunk.embedding == (0.5, 0.5)
    assert expanded.embedded_chunk.chunk.document_id == "pdf-policy"
    assert expanded.embedded_chunk.chunk.location.page_number == 4
    assert expanded.embedded_chunk.chunk.metadata["source_key"] == "pdf:policy.pdf"
    assert expanded.embedded_chunk.chunk.metadata["sentence_window_anchors"] == [2]
    assert expanded.embedded_chunk.chunk.metadata["sentence_window_ranges"] == [[1, 3]]
    assert expanded.embedded_chunk.chunk.metadata[
        "sentence_window_original_characters"
    ] == len("Opening. Target evidence. Closing.")
    assert expanded.embedded_chunk.chunk.metadata["sentence_window_characters"] == len(
        "Opening. Target evidence. Closing."
    )


def test_boundary_anchor_does_not_wrap_to_the_other_end() -> None:
    original = result("Target first. Second sentence. Third sentence. Fourth sentence.")

    expanded = expand_sentence_windows(
        "target", [original], WindowModel(), radius=1, anchor_count=1
    )[0]

    assert expanded.embedded_chunk.chunk.text == "Target first. Second sentence."


def test_no_lexical_anchor_uses_semantic_fallback() -> None:
    original = result("First sentence. Relevant meaning. Third sentence.")

    expanded = expand_sentence_windows(
        "target question",
        [original],
        WindowModel(),
        radius=0,
        anchor_count=1,
    )[0]

    assert expanded.embedded_chunk.chunk.text == "Relevant meaning."
    assert expanded.embedded_chunk.chunk.metadata["sentence_anchor_strategy"] == (
        "semantic-fallback"
    )


def test_rejects_blank_queries_and_negative_radius() -> None:
    with pytest.raises(ValueError, match="Enter a question"):
        expand_sentence_windows(" ", [result("Text.")], WindowModel())
    with pytest.raises(ValueError, match="cannot be negative"):
        expand_sentence_windows(
            "target", [result("Text.")], WindowModel(), radius=-1
        )
    with pytest.raises(ValueError, match="cannot be negative"):
        SentenceWindowRetriever([], WindowModel(), window_radius=-1)
    with pytest.raises(ValueError, match="anchor count"):
        SentenceWindowRetriever([], WindowModel(), anchor_count=0)
    with pytest.raises(ValueError, match="between zero and one"):
        SentenceWindowRetriever(
            [], WindowModel(), minimum_anchor_coverage=1.1
        )


def test_default_configuration_records_window_tuning() -> None:
    retriever = SentenceWindowRetriever([], WindowModel())

    assert retriever.configuration["sentence_window_radius"] == 1
    assert retriever.configuration["sentence_anchor_count"] == 2
    assert retriever.configuration["minimum_sentence_anchor_coverage"] == 0.4
    assert retriever.configuration["sentence_anchor_strategy"] == (
        "lexical-with-semantic-fallback"
    )
