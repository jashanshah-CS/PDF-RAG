import numpy as np
import pytest

from src.rag_project.chunker import PdfChunk
from src.rag_project.embeddings import EmbeddedChunk
from src.rag_project.search import semantic_search


class QueryModel:
    def encode(
        self,
        sentences: list[str],
        *,
        convert_to_numpy: bool,
        normalize_embeddings: bool,
        show_progress_bar: bool,
    ) -> np.ndarray:
        assert sentences == ["holiday allowance"]
        assert convert_to_numpy is True
        assert normalize_embeddings is True
        assert show_progress_bar is False
        return np.array([[1.0, 0.0]], dtype=np.float32)


def embedded(text: str, vector: tuple[float, ...]) -> EmbeddedChunk:
    return EmbeddedChunk(
        chunk=PdfChunk("handbook.pdf", 1, 1, text),
        embedding=vector,
    )


def test_semantic_search_ranks_most_similar_chunks_first() -> None:
    chunks = [
        embedded("office location", (0.0, 1.0)),
        embedded("annual leave is 25 days", (1.0, 0.0)),
        embedded("holiday rules", (0.8, 0.6)),
    ]

    results = semantic_search("holiday allowance", chunks, QueryModel(), top_k=2)

    assert [result.embedded_chunk.chunk.text for result in results] == [
        "annual leave is 25 days",
        "holiday rules",
    ]
    assert [result.score for result in results] == pytest.approx([1.0, 0.8])


def test_semantic_search_rejects_blank_questions() -> None:
    with pytest.raises(ValueError, match="Enter a question"):
        semantic_search("   ", [], QueryModel())


def test_semantic_search_rejects_invalid_result_count() -> None:
    with pytest.raises(ValueError, match="greater than zero"):
        semantic_search("holiday allowance", [], QueryModel(), top_k=0)


def test_semantic_search_returns_empty_list_when_there_are_no_chunks() -> None:
    assert semantic_search("holiday allowance", [], QueryModel()) == []
