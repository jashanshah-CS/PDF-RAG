import pytest
from datetime import UTC, datetime

from src.rag_project.documents import DocumentChunk, SourceLocation, SourceType
from src.rag_project.embeddings import EmbeddedChunk
from src.rag_project.generator import (
    OLLAMA_MODEL,
    OllamaError,
    build_evidence,
    generate_grounded_answer,
    is_refusal_answer,
    supporting_source,
)
from src.rag_project.search import SearchResult


def result(text: str, page: int, score: float = 0.8) -> SearchResult:
    chunk = DocumentChunk(
        document_id="pdf-handbook",
        source_type=SourceType.PDF,
        source_name="handbook.pdf",
        chunk_number=1,
        text=text,
        location=SourceLocation(page_number=page),
        added_at=datetime(2026, 10, 5, tzinfo=UTC),
    )
    return SearchResult(EmbeddedChunk(chunk, (1.0, 0.0)), score)


def test_evidence_contains_source_labels_and_page_metadata() -> None:
    evidence = build_evidence(
        [result("Annual leave is 25 days.", 4), result("Remote work is allowed.", 7)]
    )

    assert "[Source 1]" in evidence
    assert "Source type: pdf" in evidence
    assert "Source: handbook.pdf" in evidence
    assert "Location: page 4" in evidence
    assert "[Source 2]" in evidence
    assert "Location: page 7" in evidence


def test_evidence_contains_website_url() -> None:
    chunk = DocumentChunk(
        document_id="website-benefits",
        source_type=SourceType.WEBSITE,
        source_name="Employee Benefits",
        chunk_number=1,
        text="Employees receive 25 days.",
        location=SourceLocation(
            url="https://example.com/benefits",
            section="Annual leave",
        ),
        added_at=datetime(2026, 10, 5, tzinfo=UTC),
    )
    evidence = build_evidence(
        [SearchResult(EmbeddedChunk(chunk, (1.0, 0.0)), 0.8)]
    )

    assert "Source type: website" in evidence
    assert "Location: Annual leave" in evidence
    assert "URL: https://example.com/benefits" in evidence


def test_generation_sends_grounded_non_streaming_chat_request() -> None:
    captured_payload = {}

    def fake_request(payload):
        captured_payload.update(payload)
        return {
            "message": {
                "content": '{"answer":"Employees receive 25 days."}'
            }
        }

    answer = generate_grounded_answer(
        "How much leave do employees receive?",
        [result("Annual leave is 25 days.", 4)],
        send_request=fake_request,
    )

    assert answer == "Employees receive 25 days. [Source 1]"
    assert captured_payload["model"] == OLLAMA_MODEL
    assert captured_payload["stream"] is False
    assert captured_payload["think"] is False
    assert captured_payload["format"]["required"] == ["answer", "source_numbers"]
    assert captured_payload["options"]["temperature"] == 0.0
    assert "outside knowledge" in captured_payload["messages"][0]["content"]
    assert "exact attribute requested" in captured_payload["messages"][0]["content"]
    assert "address each one separately" in captured_payload["messages"][0]["content"]
    assert "obvious spelling mistakes" in captured_payload["messages"][0]["content"]
    assert "Location: page 4" in captured_payload["messages"][1]["content"]


def test_generation_rejects_missing_question_or_evidence() -> None:
    with pytest.raises(ValueError, match="Enter a question"):
        generate_grounded_answer(" ", [result("text", 1)])

    with pytest.raises(ValueError, match="No source evidence"):
        generate_grounded_answer("A question", [])


def test_generation_rejects_empty_model_response() -> None:
    with pytest.raises(OllamaError, match="empty or invalid"):
        generate_grounded_answer(
            "A question",
            [result("text", 1)],
            send_request=lambda payload: {"message": {"content": ""}},
        )


def test_python_selects_source_with_best_answer_overlap() -> None:
    results = [
        result("The London office opened in 2020.", 1),
        result("Employees receive 25 days of annual leave.", 4),
    ]

    assert supporting_source("Employees receive 25 days.", results) == 2


def test_generation_can_cite_multiple_model_selected_sources() -> None:
    results = [
        result("Project Orion uses forecasting.", 1),
        result("Jashan likes databases.", 2),
    ]

    def fake_request(payload):
        return {
            "message": {
                "content": (
                    '{"answer":"Project Orion uses forecasting and Jashan likes databases.",'
                    '"source_numbers":[1,2]}'
                )
            }
        }

    assert generate_grounded_answer("Connect these facts", results, fake_request) == (
        "Project Orion uses forecasting and Jashan likes databases. "
        "[Source 1] [Source 2]"
    )


def test_generation_rejects_answer_that_cannot_be_grounded() -> None:
    with pytest.raises(OllamaError, match="matched to supporting evidence"):
        generate_grounded_answer(
            "A question",
            [result("completely unrelated evidence", 1)],
            send_request=lambda payload: {
                "message": {"content": '{"answer":"Zebra quantum pineapple"}'}
            },
        )


@pytest.mark.parametrize(
    "answer",
    [
        "I cannot find this in the supplied document.",
        "I could not find any information about that.",
        "I do not find an explicit salary request.",
        "There is no information about sports.",
        "The cafeteria opening time is not specified in the supplied sources.",
    ],
)
def test_recognizes_equivalent_refusal_phrases(answer: str) -> None:
    assert is_refusal_answer(answer) is True
