"""Semantic search over embedded PDF chunks."""

from dataclasses import dataclass

import numpy as np

from src.rag_project.embeddings import EmbeddedChunk, TextEncoder


@dataclass(frozen=True)
class SearchResult:
    """One retrieved chunk and its cosine-similarity score."""

    embedded_chunk: EmbeddedChunk
    score: float


def semantic_search(
    query: str,
    embedded_chunks: list[EmbeddedChunk],
    model: TextEncoder,
    top_k: int = 3,
) -> list[SearchResult]:
    """Return the chunks whose meanings are closest to the query."""
    if not query.strip():
        raise ValueError("Enter a question before searching.")
    if top_k <= 0:
        raise ValueError("The number of results must be greater than zero.")
    if not embedded_chunks:
        return []

    query_vector = model.encode(
        [query.strip()],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )[0]

    results = [
        SearchResult(
            embedded_chunk=item,
            score=float(np.dot(query_vector, np.asarray(item.embedding))),
        )
        for item in embedded_chunks
    ]

    results.sort(key=lambda result: result.score, reverse=True)
    return results[:top_k]

