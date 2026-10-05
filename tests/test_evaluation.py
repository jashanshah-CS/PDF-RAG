from pathlib import Path
from datetime import UTC, datetime

import numpy as np

from src.rag_project.documents import DocumentChunk, SourceLocation, SourceType
from src.rag_project.embeddings import EmbeddedChunk
from src.rag_project.evaluation import (
    EvaluationCase,
    cited_pages,
    evaluate_case,
    load_evaluation_cases,
    normalize_for_keyword_check,
)
from src.rag_project.search import SearchResult


class QueryModel:
    def encode(self, sentences, **kwargs):
        return np.array([[1.0, 0.0]], dtype=np.float32)


def embedded(
    text: str,
    page: int | None,
    vector=(1.0, 0.0),
    *,
    source_name: str = "handbook.pdf",
    source_type: SourceType = SourceType.PDF,
    url: str | None = None,
) -> EmbeddedChunk:
    chunk = DocumentChunk(
        document_id=f"{source_type.value}-{source_name}",
        source_type=source_type,
        source_name=source_name,
        chunk_number=1,
        text=text,
        location=SourceLocation(page_number=page, url=url),
        added_at=datetime(2026, 10, 5, tzinfo=UTC),
    )
    return EmbeddedChunk(chunk, vector)


def test_loads_csv_case_fields(tmp_path: Path) -> None:
    csv_path = tmp_path / "questions.csv"
    csv_path.write_text(
        "question,expected_keywords,expected_page,should_refuse\n"
        '"How much leave?",25 days|annual leave,4,false\n'
        '"What is the CEO birthday?",,,true\n',
        encoding="utf-8",
    )

    cases = load_evaluation_cases(csv_path)

    assert cases[0] == EvaluationCase(
        "How much leave?", ("25 days", "annual leave"), 4, False
    )
    assert cases[1].expected_page is None
    assert cases[1].should_refuse is True


def test_loads_multi_source_expectations(tmp_path: Path) -> None:
    csv_path = tmp_path / "questions.csv"
    csv_path.write_text(
        "question,expected_keywords,expected_sources,expected_pages,expected_urls,should_refuse\n"
        '"Compare the policies",remote|hybrid,handbook.pdf|Benefits,4,'
        'https://example.com/benefits,false\n',
        encoding="utf-8",
    )

    case = load_evaluation_cases(csv_path)[0]

    assert case.expected_sources == ("handbook.pdf", "Benefits")
    assert case.expected_pages == (4,)
    assert case.expected_urls == ("https://example.com/benefits",)


def test_resolves_source_labels_to_retrieved_pages() -> None:
    results = [
        SearchResult(embedded("leave", 4), 0.9),
        SearchResult(embedded("remote work", 7), 0.8),
    ]

    assert cited_pages("See [Source 2] and [Source 1].", results) == [4, 7]


def test_answerable_case_passes_all_checks() -> None:
    case = EvaluationCase("How much leave?", ("25 days",), 4, False)
    chunks = [embedded("Annual leave is 25 days.", 4)]

    result = evaluate_case(
        case,
        chunks,
        QueryModel(),
        answer_question=lambda question, results: "Employees receive 25 days [Source 1].",
    )

    assert result.retrieval_hit is True
    assert result.keyword_hit is True
    assert result.citation_hit is True
    assert result.refusal_correct is True
    assert result.passed is True


def test_cross_source_case_requires_pdf_and_website_citations() -> None:
    case = EvaluationCase(
        "Compare remote work information.",
        ("remote", "hybrid"),
        None,
        False,
        expected_sources=("handbook.pdf", "Benefits"),
        expected_pages=(4,),
        expected_urls=("https://example.com/benefits",),
    )
    chunks = [
        embedded("Remote work is permitted.", 4, source_name="handbook.pdf"),
        embedded(
            "The hybrid schedule has two office days.",
            None,
            source_name="Benefits",
            source_type=SourceType.WEBSITE,
            url="https://example.com/benefits",
        ),
    ]

    result = evaluate_case(
        case,
        chunks,
        QueryModel(),
        answer_question=lambda question, results: (
            "Remote work uses a hybrid schedule [Source 1] [Source 2]."
        ),
    )

    assert result.retrieval_hit is True
    assert result.citation_hit is True
    assert result.retrieved_sources == "Benefits|handbook.pdf"
    assert result.cited_urls == "https://example.com/benefits"
    assert result.passed is True


def test_cross_source_case_fails_when_website_is_not_cited() -> None:
    case = EvaluationCase(
        "Compare remote work information.",
        ("remote",),
        None,
        False,
        expected_sources=("handbook.pdf", "Benefits"),
        expected_pages=(4,),
        expected_urls=("https://example.com/benefits",),
    )
    chunks = [
        embedded("Remote work is permitted.", 4),
        embedded(
            "Hybrid schedule.",
            None,
            source_name="Benefits",
            source_type=SourceType.WEBSITE,
            url="https://example.com/benefits",
        ),
    ]

    result = evaluate_case(
        case,
        chunks,
        QueryModel(),
        answer_question=lambda question, results: "Remote work is permitted [Source 1].",
    )

    assert result.retrieval_hit is True
    assert result.citation_hit is False
    assert result.passed is False


def test_unsupported_case_passes_when_model_refuses() -> None:
    case = EvaluationCase("Unknown fact?", (), None, True)

    result = evaluate_case(
        case,
        [embedded("Unrelated text", 1)],
        QueryModel(),
        answer_question=lambda question, results: (
            "I cannot find this in the supplied document."
        ),
    )

    assert result.refusal_detected is True
    assert result.refusal_correct is True
    assert result.passed is True


def test_generation_error_preserves_retrieval_result() -> None:
    case = EvaluationCase("How much leave?", ("25 days",), 4, False)

    def fail_generation(question, results):
        raise RuntimeError("model response failed")

    result = evaluate_case(
        case,
        [embedded("Annual leave is 25 days.", 4)],
        QueryModel(),
        answer_question=fail_generation,
    )

    assert result.retrieved_pages == "4"
    assert result.retrieval_hit is True
    assert result.error == "model response failed"
    assert result.passed is False


def test_keyword_normalization_treats_hyphenated_forms_as_equivalent() -> None:
    assert normalize_for_keyword_check("multi-tenant and third-party APIs") == (
        "multi tenant and third party apis"
    )


def test_keyword_normalization_treats_pound_symbol_as_gbp() -> None:
    assert normalize_for_keyword_check("The allowance is £1,200.") == (
        "the allowance is gbp 1 200"
    )
