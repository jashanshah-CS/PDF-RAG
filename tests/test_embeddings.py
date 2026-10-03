import numpy as np
import pytest

from src.rag_project.chunker import PdfChunk
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
        PdfChunk("handbook.pdf", 2, 1, "annual leave policy"),
        PdfChunk("handbook.pdf", 3, 1, "remote working policy"),
    ]

    embedded = embed_chunks(chunks, FakeModel())

    assert [item.chunk for item in embedded] == chunks
    assert embedded[0].embedding == pytest.approx((0.1, 0.2))
    assert embedded[1].embedding == pytest.approx((0.3, 0.4))


def test_empty_chunk_list_needs_no_model_work() -> None:
    assert embed_chunks([], FakeModel()) == []
