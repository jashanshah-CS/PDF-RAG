from datetime import UTC, datetime
import json
from pathlib import Path

import pytest

from src.rag_project.documents import DocumentChunk, SourceLocation, SourceType
from src.rag_project.embeddings import EmbeddedChunk
from src.rag_project.retrieval_evaluation import (
    ExpectedLocation,
    RetrievalCase,
    build_report,
    evaluate_retrieval_case,
    load_retrieval_cases,
    summarize_results,
)
from src.rag_project.search import SearchResult


def result(
    source: str,
    text: str,
    score: float,
    *,
    page: int | None = 1,
    url: str | None = None,
) -> SearchResult:
    source_type = SourceType.WEBSITE if url else SourceType.PDF
    chunk = DocumentChunk(
        document_id=f"{source_type.value}-{source}",
        source_type=source_type,
        source_name=source,
        chunk_number=1,
        text=text,
        location=SourceLocation(page_number=page, url=url),
        added_at=datetime(2026, 11, 12, tzinfo=UTC),
    )
    return SearchResult(EmbeddedChunk(chunk, (1.0, 0.0)), score)


def write_cases(path: Path, cases: list[dict]) -> None:
    path.write_text(
        json.dumps({"schema_version": 1, "cases": cases}), encoding="utf-8"
    )


def test_loads_versioned_json_cases(tmp_path: Path) -> None:
    path = tmp_path / "cases.json"
    write_cases(
        path,
        [
            {
                "id": "semantic-1",
                "category": "semantic",
                "question": "What is the allowance?",
                "expected_locations": [
                    {
                        "source": "policy.pdf",
                        "page": 2,
                        "evidence": ["GBP 240"],
                    }
                ],
            },
            {
                "id": "unsupported-1",
                "category": "unsupported",
                "question": "Who is the CEO?",
                "expected_locations": [],
            },
        ],
    )

    cases = load_retrieval_cases(path)

    assert cases[0].expected_locations == (
        ExpectedLocation("policy.pdf", page=2, evidence=("GBP 240",)),
    )
    assert cases[1].is_supported is False


def test_rejects_duplicate_case_ids(tmp_path: Path) -> None:
    path = tmp_path / "cases.json"
    case = {
        "id": "same",
        "category": "unsupported",
        "question": "Unknown?",
        "expected_locations": [],
    }
    write_cases(path, [case, case])

    with pytest.raises(ValueError, match="Duplicate benchmark case id"):
        load_retrieval_cases(path)


def test_measures_rank_and_exact_evidence_coverage() -> None:
    case = RetrievalCase(
        "multi-1",
        "multi_source",
        "Compare both policies",
        (
            ExpectedLocation("alpha.pdf", 1, evidence=("six weeks",)),
            ExpectedLocation("beta.pdf", 2, evidence=("30 days",)),
        ),
    )
    ranked = [
        result("noise.pdf", "Unrelated", 0.95),
        result("alpha.pdf", "An exception lasts six weeks.", 0.9),
        result("beta.pdf", "Claims are due in 30 days.", 0.8, page=2),
    ]

    measured = evaluate_retrieval_case(case, lambda question: ranked)

    assert measured.source_recall == 1.0
    assert measured.location_recall == 1.0
    assert measured.evidence_recall == 1.0
    assert measured.reciprocal_rank == 0.5
    assert measured.matched_count == 2


def test_source_hit_does_not_hide_wrong_page_or_missing_evidence() -> None:
    case = RetrievalCase(
        "context-1",
        "context",
        "What is the rule?",
        (ExpectedLocation("policy.pdf", 2, evidence=("required phrase",)),),
    )
    ranked = [result("policy.pdf", "Different material", 0.9, page=1)]

    measured = evaluate_retrieval_case(case, lambda question: ranked)

    assert measured.source_recall == 1.0
    assert measured.location_recall == 0.0
    assert measured.evidence_recall == 0.0
    assert measured.reciprocal_rank == 0.0


def test_url_matching_ignores_only_trailing_slash() -> None:
    case = RetrievalCase(
        "web-1",
        "semantic",
        "What does the page say?",
        (
            ExpectedLocation(
                "Benefits", url="https://example.com/benefits/", evidence=("hybrid",)
            ),
        ),
    )
    ranked = [
        result(
            "Benefits",
            "The schedule is hybrid.",
            0.8,
            page=None,
            url="https://example.com/benefits",
        )
    ]

    measured = evaluate_retrieval_case(case, lambda question: ranked)

    assert measured.location_recall == 1.0
    assert measured.evidence_recall == 1.0


def test_summary_excludes_unsupported_probes_from_recall() -> None:
    supported = evaluate_retrieval_case(
        RetrievalCase(
            "exact-1",
            "exact",
            "What is the code?",
            (ExpectedLocation("codes.pdf", 1, evidence=("ZX-42",)),),
        ),
        lambda question: [result("codes.pdf", "The code is ZX-42.", 0.9)],
    )
    unsupported = evaluate_retrieval_case(
        RetrievalCase("unsupported-1", "unsupported", "Unknown?", ()),
        lambda question: [result("codes.pdf", "Unrelated", 0.4)],
    )

    summary = summarize_results([supported, unsupported], top_k=5)

    assert summary["recall_at_5"] == 1.0
    assert summary["mean_reciprocal_rank"] == 1.0
    assert summary["unsupported_case_count"] == 1
    assert summary["unsupported_top_score_mean"] == 0.4


def test_report_contains_corpus_and_serializable_details() -> None:
    measured = evaluate_retrieval_case(
        RetrievalCase(
            "semantic-1",
            "semantic",
            "Question?",
            (ExpectedLocation("policy.pdf", 1, evidence=("answer",)),),
        ),
        lambda question: [result("policy.pdf", "The answer.", 0.9)],
    )

    report = build_report(
        [measured], benchmark_name="baseline", top_k=5, corpus=["policy.pdf"]
    )

    assert report["benchmark"] == "baseline"
    assert report["corpus"] == ["policy.pdf"]
    assert report["results"][0]["retrieved"][0]["source"] == "policy.pdf"
    json.dumps(report)
