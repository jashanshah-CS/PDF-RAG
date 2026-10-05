"""Semantic search over embedded PDF chunks."""

from dataclasses import dataclass
import re

import numpy as np

from src.rag_project.embeddings import EmbeddedChunk, TextEncoder


@dataclass(frozen=True)
class SearchResult:
    """One retrieved chunk and its cosine-similarity score."""

    embedded_chunk: EmbeddedChunk
    score: float


SEARCH_STOPWORDS = {
    "a", "an", "and", "are", "does", "for", "how", "in", "is", "it",
    "much", "of", "on", "the", "to", "what", "when", "which", "who",
}


def lexical_terms(text: str) -> set[str]:
    """Return lightly normalized terms for a small exact-match search signal."""
    terms = set()
    for token in re.findall(r"[a-z0-9]+", text.lower()):
        if token in SEARCH_STOPWORDS:
            continue
        if len(token) > 4 and token.endswith("s"):
            token = token[:-1]
        terms.add(token)
    return terms


def semantic_search(
    query: str,
    embedded_chunks: list[EmbeddedChunk],
    model: TextEncoder,
    top_k: int = 5,
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

    query_terms = lexical_terms(query)
    ranked_results = []
    for item in embedded_chunks:
        semantic_score = float(np.dot(query_vector, np.asarray(item.embedding)))
        # Later PDF pages often omit the document title. Including the source
        # name preserves that context for queries naming the document.
        chunk_terms = lexical_terms(f"{item.chunk.source_name} {item.chunk.text}")
        lexical_coverage = (
            len(query_terms.intersection(chunk_terms)) / len(query_terms)
            if query_terms
            else 0.0
        )
        # Semantic similarity remains primary. The lexical bonus protects exact
        # names, dates, and policy terms from being displaced by vague matches.
        combined_score = semantic_score + (0.2 * lexical_coverage)
        ranked_results.append(
            (combined_score, SearchResult(item, semantic_score))
        )

    ranked_results.sort(key=lambda item: item[0], reverse=True)
    return [result for _, result in ranked_results[:top_k]]
