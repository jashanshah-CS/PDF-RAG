from datetime import UTC, datetime

import pytest

from src.rag_project.bm25 import BM25Index, bm25_tokens
from src.rag_project.documents import DocumentChunk, SourceLocation, SourceType
from src.rag_project.embeddings import EmbeddedChunk


def embedded(
    text: str,
    *,
    source: str = "policy.pdf",
    page: int = 1,
) -> EmbeddedChunk:
    return EmbeddedChunk(
        DocumentChunk(
            document_id=f"pdf-{source}",
            source_type=SourceType.PDF,
            source_name=source,
            chunk_number=page,
            text=text,
            location=SourceLocation(page_number=page),
            added_at=datetime(2026, 11, 14, tzinfo=UTC),
        ),
        (),
    )


def test_tokenizes_acronyms_numbers_and_plural_variants() -> None:
    assert bm25_tokens("Which policies have an RPO of 15 minutes at 17:00?") == [
        "policy",
        "have",
        "rpo",
        "15",
        "minute",
        "at",
        "17",
        "00",
    ]


def test_rare_exact_terms_rank_above_generic_policy_matches() -> None:
    index = BM25Index(
        [
            embedded("General travel policy and approval rules.", source="travel.pdf"),
            embedded("Release identifier ZX-42 requires approval.", source="release.pdf"),
            embedded("General office policy.", source="office.pdf"),
        ]
    )

    results = index.search("Which policy mentions ZX-42?")

    assert results[0].embedded_chunk.chunk.source_name == "release.pdf"
    assert results[0].score > results[1].score


def test_source_filename_supplies_missing_title_context() -> None:
    index = BM25Index(
        [
            embedded("Success is measured at 92 percent.", source="project_orion.pdf"),
            embedded("Success is measured each quarter.", source="general_report.pdf"),
        ]
    )

    results = index.search("Project Orion success measure", top_k=1)

    assert results[0].embedded_chunk.chunk.source_name == "project_orion.pdf"


def test_returns_only_documents_with_lexical_overlap() -> None:
    index = BM25Index([embedded("Remote working rules")])

    assert index.search("cafeteria menu") == []


def test_empty_index_returns_no_results() -> None:
    assert BM25Index([]).search("policy") == []


def test_rejects_invalid_queries_and_configuration() -> None:
    with pytest.raises(ValueError, match="Enter a question"):
        BM25Index([embedded("policy")]).search("   ")
    with pytest.raises(ValueError, match="greater than zero"):
        BM25Index([embedded("policy")]).search("policy", top_k=0)
    with pytest.raises(ValueError, match="k1"):
        BM25Index([], k1=0)
    with pytest.raises(ValueError, match="between zero and one"):
        BM25Index([], b=1.1)
