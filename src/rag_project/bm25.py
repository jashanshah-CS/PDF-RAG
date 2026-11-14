"""A small, dependency-free BM25 keyword index for document chunks."""

from collections import Counter, defaultdict
from collections.abc import Iterable
import math
import re

from src.rag_project.embeddings import EmbeddedChunk
from src.rag_project.search import SearchResult


BM25_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "be",
    "does",
    "for",
    "from",
    "how",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "the",
    "their",
    "to",
    "what",
    "when",
    "which",
    "who",
    "with",
}


def _singularize(token: str) -> str:
    """Apply conservative plural normalization without a stemming dependency."""
    if len(token) > 4 and token.endswith("ies"):
        return f"{token[:-3]}y"
    if len(token) > 4 and token.endswith("s") and not token.endswith(
        ("is", "ss", "us")
    ):
        return token[:-1]
    return token


def bm25_tokens(text: str) -> list[str]:
    """Tokenize names, acronyms, dates, and prose consistently for BM25."""
    return [
        _singularize(token)
        for token in re.findall(r"[a-z0-9]+", text.casefold())
        if token not in BM25_STOPWORDS
    ]


class BM25Index:
    """In-memory Okapi BM25 index with an inverted term-to-document map."""

    def __init__(
        self,
        chunks: Iterable[EmbeddedChunk],
        *,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        if k1 <= 0:
            raise ValueError("BM25 k1 must be greater than zero.")
        if not 0 <= b <= 1:
            raise ValueError("BM25 b must be between zero and one.")

        self._chunks = tuple(chunks)
        self._k1 = k1
        self._b = b
        self._document_lengths: list[int] = []
        self._postings: dict[str, dict[int, int]] = defaultdict(dict)

        for document_index, item in enumerate(self._chunks):
            # A page or chunk may not repeat the document title. Indexing the
            # source name once preserves filename/title terms without giving
            # them an arbitrary large weight.
            tokens = bm25_tokens(f"{item.chunk.source_name} {item.chunk.text}")
            self._document_lengths.append(len(tokens))
            for term, frequency in Counter(tokens).items():
                self._postings[term][document_index] = frequency

        self._average_document_length = (
            sum(self._document_lengths) / len(self._document_lengths)
            if self._document_lengths
            else 0.0
        )

    def _inverse_document_frequency(self, document_frequency: int) -> float:
        document_count = len(self._chunks)
        return math.log(
            1 + (document_count - document_frequency + 0.5)
            / (document_frequency + 0.5)
        )

    def search(self, query: str, *, top_k: int = 5) -> list[SearchResult]:
        """Return chunks ranked by Okapi BM25 lexical relevance."""
        if not query.strip():
            raise ValueError("Enter a question before searching.")
        if top_k <= 0:
            raise ValueError("The number of results must be greater than zero.")
        if not self._chunks:
            return []

        scores: dict[int, float] = defaultdict(float)
        for term in set(bm25_tokens(query)):
            postings = self._postings.get(term)
            if not postings:
                continue
            inverse_document_frequency = self._inverse_document_frequency(
                len(postings)
            )
            for document_index, term_frequency in postings.items():
                document_length = self._document_lengths[document_index]
                length_normalization = 1 - self._b + self._b * (
                    document_length / self._average_document_length
                )
                numerator = term_frequency * (self._k1 + 1)
                denominator = term_frequency + self._k1 * length_normalization
                scores[document_index] += (
                    inverse_document_frequency * numerator / denominator
                )

        ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
        return [
            SearchResult(self._chunks[document_index], score)
            for document_index, score in ranked[:top_k]
            if score > 0
        ]
