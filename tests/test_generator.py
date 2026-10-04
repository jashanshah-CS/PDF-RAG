import pytest

from src.rag_project.chunker import PdfChunk
from src.rag_project.embeddings import EmbeddedChunk
from src.rag_project.generator import (
    OLLAMA_MODEL,
    OllamaError,
    build_evidence,
    generate_grounded_answer,
    supporting_source,
)
from src.rag_project.search import SearchResult


def result(text: str, page: int, score: float = 0.8) -> SearchResult:
    chunk = PdfChunk("handbook.pdf", page, 1, text)
    return SearchResult(EmbeddedChunk(chunk, (1.0, 0.0)), score)


def test_evidence_contains_source_labels_and_page_metadata() -> None:
    evidence = build_evidence(
        [result("Annual leave is 25 days.", 4), result("Remote work is allowed.", 7)]
    )

    assert "[Source 1]" in evidence
    assert "Document: handbook.pdf" in evidence
    assert "Page: 4" in evidence
    assert "[Source 2]" in evidence
    assert "Page: 7" in evidence


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
    assert captured_payload["format"]["required"] == ["answer"]
    assert captured_payload["options"]["temperature"] == 0.1
    assert "outside knowledge" in captured_payload["messages"][0]["content"]
    assert "Page: 4" in captured_payload["messages"][1]["content"]


def test_generation_rejects_missing_question_or_evidence() -> None:
    with pytest.raises(ValueError, match="Enter a question"):
        generate_grounded_answer(" ", [result("text", 1)])

    with pytest.raises(ValueError, match="No PDF evidence"):
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


def test_generation_rejects_answer_that_cannot_be_grounded() -> None:
    with pytest.raises(OllamaError, match="matched to supporting evidence"):
        generate_grounded_answer(
            "A question",
            [result("completely unrelated evidence", 1)],
            send_request=lambda payload: {
                "message": {"content": '{"answer":"Zebra quantum pineapple"}'}
            },
        )
