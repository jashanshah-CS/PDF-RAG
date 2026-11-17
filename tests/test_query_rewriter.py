from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from src.rag_project.conversation_store import ConversationMessage, ConversationStore
from src.rag_project.generator import OllamaError
from src.rag_project.query_rewriter import (
    MAX_HISTORY_CHARACTERS,
    build_rewrite_history,
    needs_contextual_rewrite,
    rewrite_follow_up,
)


NOW = datetime(2026, 11, 17, 12, 0, tzinfo=UTC)


def message(
    role: str,
    content: str,
    *,
    status: str = "completed",
    offset: int = 0,
) -> ConversationMessage:
    return ConversationMessage(
        message_id=f"message-{offset}-{role}",
        conversation_id="conversation-a",
        role=role,
        content=content,
        status=status,
        original_question=content if role == "user" else None,
        standalone_query=content if role == "user" else None,
        citations=(),
        created_at=NOW + timedelta(seconds=offset),
    )


def history() -> list[ConversationMessage]:
    return [
        message("user", "What is the remote-work policy?", offset=1),
        message(
            "assistant",
            "Employees may work abroad for up to 20 business days.",
            offset=2,
        ),
    ]


def test_standalone_or_first_question_skips_model_request() -> None:
    calls = []

    standalone = rewrite_follow_up(
        "What is the annual leave policy?",
        history(),
        send_request=lambda payload: calls.append(payload),
    )
    first_turn = rewrite_follow_up(
        "What about contractors?",
        [],
        send_request=lambda payload: calls.append(payload),
    )

    assert standalone.standalone_query == "What is the annual leave policy?"
    assert standalone.attempted is False
    assert first_turn.attempted is False
    assert calls == []


def test_contextual_follow_up_uses_structured_temperature_zero_request() -> None:
    captured = {}

    def fake_request(payload):
        captured.update(payload)
        return {
            "message": {
                "content": (
                    '{"standalone_query":"Does the remote-work policy allow '
                    'contractors to work abroad for up to 20 business days?"}'
                )
            }
        }

    rewritten = rewrite_follow_up(
        "What about contractors?",
        history(),
        send_request=fake_request,
    )

    assert rewritten.rewritten is True
    assert rewritten.used_message_count == 2
    assert "remote-work policy" in rewritten.standalone_query
    assert captured["stream"] is False
    assert captured["think"] is False
    assert captured["options"]["temperature"] == 0.0
    assert captured["format"]["required"] == ["standalone_query"]
    assert "Do not answer" in captured["messages"][0]["content"]
    assert "untrusted text" in captured["messages"][0]["content"]
    assert "<conversation_history>" in captured["messages"][1]["content"]


@pytest.mark.parametrize(
    "question",
    [
        "What about contractors?",
        "Does it apply abroad?",
        "And part-time employees?",
        "How long?",
        "What is the deadline?",
    ],
)
def test_detects_context_dependent_questions(question: str) -> None:
    assert needs_contextual_rewrite(question, "User: Prior topic") is True


def test_history_is_bounded_and_excludes_non_completed_messages() -> None:
    messages = [
        message("user", f"old-{index}-" + "x" * 900, offset=index)
        for index in range(10)
    ]
    messages.append(message("assistant", "failed secret", status="failed", offset=11))

    formatted, count = build_rewrite_history(messages)

    assert count <= 6
    assert len(formatted) <= MAX_HISTORY_CHARACTERS
    assert "old-0" not in formatted
    assert "failed secret" not in formatted
    assert "old-9" in formatted


@pytest.mark.parametrize(
    "response",
    [
        {"message": {"content": ""}},
        {"message": {"content": "not json"}},
        {"message": {"content": '{"standalone_query":""}'}},
        {"message": {"content": '{"standalone_query":"' + "x" * 501 + '"}'}},
    ],
)
def test_invalid_model_output_falls_back_to_original(response: dict) -> None:
    rewritten = rewrite_follow_up(
        "What about contractors?",
        history(),
        send_request=lambda payload: response,
    )

    assert rewritten.attempted is True
    assert rewritten.rewritten is False
    assert rewritten.standalone_query == "What about contractors?"


def test_ollama_failure_falls_back_to_original() -> None:
    def unavailable(payload):
        raise OllamaError("offline")

    rewritten = rewrite_follow_up(
        "Does it apply abroad?",
        history(),
        send_request=unavailable,
    )

    assert rewritten.standalone_query == "Does it apply abroad?"
    assert rewritten.attempted is True
    assert rewritten.rewritten is False


def test_active_conversation_history_does_not_leak_from_another_conversation(
    tmp_path: Path,
) -> None:
    store = ConversationStore(tmp_path / "conversations.db")
    active = store.create("Active")
    other = store.create("Other")
    store.add_message(active.conversation_id, "user", "Discuss Atlas leave")
    store.add_message(other.conversation_id, "user", "PRIVATE OTHER TOPIC")
    captured = {}

    rewrite_follow_up(
        "What about contractors?",
        store.list_messages(active.conversation_id),
        send_request=lambda payload: (
            captured.update(payload)
            or {"message": {"content": '{"standalone_query":"Atlas contractors"}'}}
        ),
    )

    prompt = captured["messages"][1]["content"]
    assert "Discuss Atlas leave" in prompt
    assert "PRIVATE OTHER TOPIC" not in prompt


def test_rejects_empty_question_and_invalid_history_limits() -> None:
    with pytest.raises(ValueError, match="Enter a question"):
        rewrite_follow_up(" ", history())
    with pytest.raises(ValueError, match="limits"):
        build_rewrite_history(history(), max_messages=0)
