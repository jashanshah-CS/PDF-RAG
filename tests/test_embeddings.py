import numpy as np
import pytest
from datetime import UTC, datetime

from src.rag_project.documents import DocumentChunk, SourceLocation, SourceType
from src.rag_project.embeddings import embed_chunks


class FakeModel:
    def encode(
        self,
        sentences: list[str],
        *,
        convert_to_numpy: bool,
        normalize_embeddings: bool,
        show_progress_bar: bool,
    ) -> np.ndarray:
        assert sentences == ["annual leave policy", "remote working policy"]
        assert convert_to_numpy is True
        assert normalize_embeddings is True
        assert show_progress_bar is False
        return np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)


def test_embedding_is_attached_to_its_original_chunk() -> None:
    chunks = [
        make_chunk("annual leave policy", 2),
        make_chunk("remote working policy", 3),
    ]

    embedded = embed_chunks(chunks, FakeModel())

    assert [item.chunk for item in embedded] == chunks
    assert embedded[0].embedding == pytest.approx((0.1, 0.2))
    assert embedded[1].embedding == pytest.approx((0.3, 0.4))


def test_empty_chunk_list_needs_no_model_work() -> None:
    assert embed_chunks([], FakeModel()) == []


def make_chunk(text: str, page: int) -> DocumentChunk:
    return DocumentChunk(
        document_id="pdf-handbook",
        source_type=SourceType.PDF,
        source_name="handbook.pdf",
        chunk_number=1,
        text=text,
        location=SourceLocation(page_number=page),
        added_at=datetime(2026, 10, 5, tzinfo=UTC),
    )
