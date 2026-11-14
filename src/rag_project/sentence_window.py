"""Focus retrieved chunks on a relevant sentence and its local neighbours."""

from dataclasses import replace
import re

import numpy as np

from src.rag_project.embeddings import EmbeddedChunk, TextEncoder
from src.rag_project.hybrid import HybridRetriever
from src.rag_project.search import SearchResult, lexical_terms


DEFAULT_WINDOW_RADIUS = 1
DEFAULT_ANCHOR_COUNT = 2
DEFAULT_MINIMUM_ANCHOR_COVERAGE = 0.4
SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+(?=[\"'A-Z0-9])")


def split_sentences(text: str) -> list[str]:
    """Split normalized extracted text while retaining sentence punctuation."""
    normalized = " ".join(text.split())
    if not normalized:
        return []
    return [
        sentence.strip()
        for sentence in SENTENCE_BOUNDARY.split(normalized)
        if sentence.strip()
    ]


def expand_sentence_windows(
    query: str,
    results: list[SearchResult],
    model: TextEncoder,
    *,
    radius: int = DEFAULT_WINDOW_RADIUS,
    anchor_count: int = DEFAULT_ANCHOR_COUNT,
    minimum_anchor_coverage: float = DEFAULT_MINIMUM_ANCHOR_COVERAGE,
) -> list[SearchResult]:
    """Replace each result text with its best sentence plus nearby context."""
    if not query.strip():
        raise ValueError("Enter a question before searching.")
    if radius < 0:
        raise ValueError("The sentence-window radius cannot be negative.")
    if anchor_count <= 0:
        raise ValueError("The sentence anchor count must be greater than zero.")
    if not 0 <= minimum_anchor_coverage <= 1:
        raise ValueError("Minimum sentence coverage must be between zero and one.")
    if not results:
        return []

    sentences_by_result = [
        split_sentences(result.embedded_chunk.chunk.text) for result in results
    ]
    lexical_scores_by_result: list[list[float]] = []
    semantic_fallback_indices: list[int] = []
    for result_index, (result, sentences) in enumerate(
        zip(results, sentences_by_result, strict=True)
    ):
        source_terms = lexical_terms(result.embedded_chunk.chunk.source_name)
        query_terms = lexical_terms(query).difference(source_terms)
        lexical_scores = [
            len(query_terms.intersection(lexical_terms(sentence)))
            / len(query_terms)
            if query_terms
            else 0.0
            for sentence in sentences
        ]
        lexical_scores_by_result.append(lexical_scores)
        if sentences and max(lexical_scores, default=0.0) < minimum_anchor_coverage:
            semantic_fallback_indices.append(result_index)

    semantic_scores_by_result: dict[int, list[float]] = {}
    fallback_sentences = [
        sentence
        for result_index in semantic_fallback_indices
        for sentence in sentences_by_result[result_index]
    ]
    if fallback_sentences:
        vectors = model.encode(
            [query.strip(), *fallback_sentences],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        query_vector = np.asarray(vectors[0])
        sentence_vectors = vectors[1:]
        offset = 0
        for result_index in semantic_fallback_indices:
            sentence_count = len(sentences_by_result[result_index])
            current_vectors = sentence_vectors[offset : offset + sentence_count]
            semantic_scores_by_result[result_index] = list(
                np.asarray(current_vectors) @ query_vector
            )
            offset += sentence_count

    expanded: list[SearchResult] = []
    for result_index, (result, sentences) in enumerate(
        zip(results, sentences_by_result, strict=True)
    ):
        if not sentences:
            expanded.append(result)
            continue
        used_semantic_fallback = result_index in semantic_scores_by_result
        selection_scores = semantic_scores_by_result.get(
            result_index, lexical_scores_by_result[result_index]
        )
        anchors = sorted(
            int(index)
            for index in sorted(
                range(len(sentences)),
                key=lambda index: (-selection_scores[index], index),
            )[: min(anchor_count, len(sentences))]
        )
        selected_indices = sorted(
            {
                sentence_index
                for anchor in anchors
                for sentence_index in range(
                    max(0, anchor - radius),
                    min(len(sentences), anchor + radius + 1),
                )
            }
        )
        ranges: list[tuple[int, int]] = []
        range_start = selected_indices[0]
        previous = selected_indices[0]
        for sentence_index in selected_indices[1:]:
            if sentence_index != previous + 1:
                ranges.append((range_start, previous + 1))
                range_start = sentence_index
            previous = sentence_index
        ranges.append((range_start, previous + 1))
        window_text = " […] ".join(
            " ".join(sentences[start:end]) for start, end in ranges
        )
        original_chunk = result.embedded_chunk.chunk
        windowed_chunk = replace(
            original_chunk,
            text=window_text,
            metadata={
                **original_chunk.metadata,
                "sentence_window": True,
                "sentence_window_anchors": [anchor + 1 for anchor in anchors],
                "sentence_window_ranges": [
                    [start + 1, end] for start, end in ranges
                ],
                "sentence_count": len(sentences),
                "sentence_window_original_characters": len(original_chunk.text),
                "sentence_window_characters": len(window_text),
                "sentence_anchor_strategy": (
                    "semantic-fallback" if used_semantic_fallback else "lexical"
                ),
            },
        )
        expanded.append(
            SearchResult(
                EmbeddedChunk(windowed_chunk, result.embedded_chunk.embedding),
                result.score,
            )
        )
    return expanded


class SentenceWindowRetriever:
    """Hybrid retrieval followed by semantic sentence-window selection."""

    def __init__(
        self,
        chunks: list[EmbeddedChunk],
        model: TextEncoder,
        *,
        window_radius: int = DEFAULT_WINDOW_RADIUS,
        anchor_count: int = DEFAULT_ANCHOR_COUNT,
        minimum_anchor_coverage: float = DEFAULT_MINIMUM_ANCHOR_COVERAGE,
    ) -> None:
        if window_radius < 0:
            raise ValueError("The sentence-window radius cannot be negative.")
        if anchor_count <= 0:
            raise ValueError("The sentence anchor count must be greater than zero.")
        if not 0 <= minimum_anchor_coverage <= 1:
            raise ValueError("Minimum sentence coverage must be between zero and one.")
        self._model = model
        self._window_radius = window_radius
        self._anchor_count = anchor_count
        self._minimum_anchor_coverage = minimum_anchor_coverage
        self._hybrid = HybridRetriever(chunks, model)

    @property
    def configuration(self) -> dict[str, int | float | str]:
        """Return hybrid and window parameters for benchmark reports."""
        return {
            **self._hybrid.configuration,
            "sentence_window_radius": self._window_radius,
            "sentence_anchor_count": self._anchor_count,
            "minimum_sentence_anchor_coverage": self._minimum_anchor_coverage,
            "sentence_anchor_strategy": "lexical-with-semantic-fallback",
        }

    def search(self, query: str, *, top_k: int = 5) -> list[SearchResult]:
        """Retrieve chunks and focus each one on its strongest local sentence."""
        results = self._hybrid.search(query, top_k=top_k)
        return expand_sentence_windows(
            query,
            results,
            self._model,
            radius=self._window_radius,
            anchor_count=self._anchor_count,
            minimum_anchor_coverage=self._minimum_anchor_coverage,
        )
