"""Create local semantic embeddings for source-neutral document chunks."""

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import NDArray
from sentence_transformers import SentenceTransformer

from src.rag_project.documents import DocumentChunk
from src.rag_project.config import SETTINGS


MODEL_NAME = SETTINGS.embedding_model
MODEL_CACHE = SETTINGS.model_cache


class TextEncoder(Protocol):
    """The small part of SentenceTransformer used by this project."""

    def encode(
        self,
        sentences: list[str],
        *,
        convert_to_numpy: bool,
        normalize_embeddings: bool,
        show_progress_bar: bool,
    ) -> NDArray[np.float32]: ...


@dataclass(frozen=True)
class EmbeddedChunk:
    """A document chunk paired with its numerical meaning representation."""

    chunk: DocumentChunk
    embedding: tuple[float, ...]


def load_embedding_model() -> SentenceTransformer:
    """Load the free embedding model on the CPU and cache it locally."""
    MODEL_CACHE.mkdir(parents=True, exist_ok=True)
    return SentenceTransformer(
        MODEL_NAME,
        device="cpu",
        cache_folder=str(MODEL_CACHE),
        local_files_only=SETTINGS.embedding_local_only,
    )


def embed_chunks(
    chunks: list[DocumentChunk],
    model: TextEncoder,
) -> list[EmbeddedChunk]:
    """Create one normalized embedding for every chunk."""
    if not chunks:
        return []

    vectors = model.encode(
        [chunk.text for chunk in chunks],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    return [
        EmbeddedChunk(
            chunk=chunk,
            embedding=tuple(float(value) for value in vector),
        )
        for chunk, vector in zip(chunks, vectors, strict=True)
    ]
