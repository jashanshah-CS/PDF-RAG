"""Deterministic Version 4 conversational quality evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
import json
from pathlib import Path
from statistics import mean
from typing import Any

from src.rag_project.citations import calibrated_confidence, cited_source_numbers
from src.rag_project.conversation_store import (
    ConversationMessage,
    ConversationStore,
)
from src.rag_project.conversation_ui import message_citations
from src.rag_project.documents import DocumentChunk, SourceLocation, SourceType
from src.rag_project.embeddings import EmbeddedChunk
from src.rag_project.generator import generate_grounded_answer
from src.rag_project.query_rewriter import rewrite_follow_up
from src.rag_project.search import SearchResult


SUPPORTED_CATEGORIES = frozenset(
    {
        "follow_up_rewrite",
        "ambiguous_reference",
        "unsupported_answer",
        "citation_correctness",
        "conversation_isolation",
        "incomplete_generation",
    }
)
FIXED_TIME = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)


@dataclass(frozen=True)
class ConversationBenchmarkCase:
    """One versioned conversational behaviour check."""

    case_id: str
    category: str
    data: dict[str, Any]


def _required_text(value: Any, field: str, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string in {context}.")
    return value.strip()


def _required_list(value: Any, field: str, context: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{field} must be a list in {context}.")
    return value


def load_conversation_benchmark(
    path: Path,
) -> tuple[str, list[ConversationBenchmarkCase]]:
    """Load and strictly validate a Version 4 conversational benchmark."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exception:
        raise ValueError(f"Conversation benchmark does not exist: {path}") from exception
    except json.JSONDecodeError as exception:
        raise ValueError(
            f"Invalid conversation benchmark JSON: {exception.msg}."
        ) from exception
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError("Conversation benchmark schema_version must be 1.")
    name = _required_text(payload.get("name"), "name", "benchmark")
    raw_cases = _required_list(payload.get("cases"), "cases", "benchmark")
    if not raw_cases:
        raise ValueError("Conversation benchmark cases cannot be empty.")

    cases: list[ConversationBenchmarkCase] = []
    seen_ids: set[str] = set()
    for index, raw_case in enumerate(raw_cases, start=1):
        context = f"case {index}"
        if not isinstance(raw_case, dict):
            raise ValueError(f"{context} must be an object.")
        case_id = _required_text(raw_case.get("id"), "id", context)
        if case_id in seen_ids:
            raise ValueError(f"Duplicate conversation benchmark case id: {case_id}.")
        seen_ids.add(case_id)
        category = _required_text(raw_case.get("category"), "category", context)
        if category not in SUPPORTED_CATEGORIES:
            raise ValueError(f"Unsupported conversation category {category!r}.")
        data = {key: value for key, value in raw_case.items() if key not in {"id", "category"}}
        _validate_case_data(category, data, context)
        cases.append(ConversationBenchmarkCase(case_id, category, data))
    return name, cases


def _validate_case_data(category: str, data: dict[str, Any], context: str) -> None:
    if category in {"follow_up_rewrite", "ambiguous_reference"}:
        _required_text(data.get("question"), "question", context)
        _required_list(data.get("history"), "history", context)
        _required_text(data.get("model_response"), "model_response", context)
        _required_text(
            data.get("expected_standalone_query"),
            "expected_standalone_query",
            context,
        )
        if not isinstance(data.get("expected_rewritten"), bool):
            raise ValueError(f"expected_rewritten must be boolean in {context}.")
    elif category in {"unsupported_answer", "citation_correctness"}:
        _required_text(data.get("question"), "question", context)
        _required_list(data.get("evidence"), "evidence", context)
        if not isinstance(data.get("model_output"), dict):
            raise ValueError(f"model_output must be an object in {context}.")
        _required_text(data.get("expected_answer"), "expected_answer", context)
        _required_list(
            data.get("expected_source_numbers"),
            "expected_source_numbers",
            context,
        )
        _required_text(data.get("expected_confidence"), "expected_confidence", context)
    elif category == "conversation_isolation":
        _required_text(data.get("question"), "question", context)
        _required_list(data.get("active_history"), "active_history", context)
        _required_list(data.get("other_history"), "other_history", context)
        _required_text(data.get("model_response"), "model_response", context)
        _required_text(
            data.get("expected_standalone_query"),
            "expected_standalone_query",
            context,
        )
    else:
        status = _required_text(data.get("terminal_status"), "terminal_status", context)
        if status not in {"failed", "cancelled"}:
            raise ValueError(
                f"terminal_status must be failed or cancelled in {context}."
            )
        _required_text(data.get("draft_content"), "draft_content", context)
        _required_text(data.get("final_content"), "final_content", context)


def _messages(
    raw_history: list[Any], conversation_id: str
) -> list[ConversationMessage]:
    messages: list[ConversationMessage] = []
    for index, item in enumerate(raw_history, start=1):
        if not isinstance(item, dict):
            raise ValueError("Conversation history entries must be objects.")
        role = _required_text(item.get("role"), "role", "conversation history")
        if role not in {"user", "assistant"}:
            raise ValueError("Conversation history role must be user or assistant.")
        content = _required_text(
            item.get("content"), "content", "conversation history"
        )
        messages.append(
            ConversationMessage(
                message_id=f"{conversation_id}-{index}",
                conversation_id=conversation_id,
                role=role,
                content=content,
                status="completed",
                original_question=content if role == "user" else None,
                standalone_query=content if role == "user" else None,
                citations=(),
                created_at=FIXED_TIME + timedelta(seconds=index),
            )
        )
    return messages


def _results(raw_evidence: list[Any]) -> list[SearchResult]:
    results: list[SearchResult] = []
    for index, item in enumerate(raw_evidence, start=1):
        if not isinstance(item, dict):
            raise ValueError("Evidence entries must be objects.")
        text = _required_text(item.get("text"), "text", "evidence")
        source_name = _required_text(
            item.get("source", f"source-{index}.pdf"), "source", "evidence"
        )
        page = item.get("page", index)
        if not isinstance(page, int) or page <= 0:
            raise ValueError("Evidence page must be a positive integer.")
        chunk = DocumentChunk(
            document_id=f"benchmark-document-{index}",
            source_type=SourceType.PDF,
            source_name=source_name,
            chunk_number=index,
            text=text,
            location=SourceLocation(page_number=page),
            added_at=FIXED_TIME,
        )
        results.append(SearchResult(EmbeddedChunk(chunk, (1.0, 0.0)), 1.0))
    return results


def _result(
    case: ConversationBenchmarkCase,
    checks: dict[str, bool],
    actual: dict[str, Any],
) -> dict[str, Any]:
    return {
        "id": case.case_id,
        "category": case.category,
        "passed": all(checks.values()),
        "checks": checks,
        "actual": actual,
    }


def _evaluate_rewrite(case: ConversationBenchmarkCase) -> dict[str, Any]:
    data = case.data
    rewrite = rewrite_follow_up(
        data["question"],
        _messages(data["history"], case.case_id),
        send_request=lambda payload: {
            "message": {
                "content": json.dumps(
                    {"standalone_query": data["model_response"]}
                )
            }
        },
    )
    checks = {
        "standalone_query": rewrite.standalone_query
        == data["expected_standalone_query"],
        "rewrite_attempted": rewrite.attempted is True,
        "rewrite_decision": rewrite.rewritten is data["expected_rewritten"],
    }
    return _result(
        case,
        checks,
        {
            "standalone_query": rewrite.standalone_query,
            "attempted": rewrite.attempted,
            "rewritten": rewrite.rewritten,
        },
    )


def _evaluate_answer(case: ConversationBenchmarkCase) -> dict[str, Any]:
    data = case.data
    results = _results(data["evidence"])
    answer = generate_grounded_answer(
        data["question"],
        results,
        send_request=lambda payload: {
            "message": {"content": json.dumps(data["model_output"])}
        },
    )
    source_numbers = list(cited_source_numbers(answer, len(results)))
    citations = message_citations(answer, results)
    evidence_texts = {
        index: result.embedded_chunk.chunk.text
        for index, result in enumerate(results, start=1)
    }
    confidence = calibrated_confidence(answer, evidence_texts)
    resolved_numbers = list(
        dict.fromkeys(citation.source_number for citation in citations)
    )
    citations_resolve = (
        resolved_numbers == source_numbers
        and all(
            citation.excerpt
            == results[citation.source_number - 1].embedded_chunk.chunk.text
            and citation.source_name
            == results[citation.source_number - 1].embedded_chunk.chunk.source_name
            for citation in citations
        )
    )
    checks = {
        "answer": answer == data["expected_answer"],
        "source_numbers": source_numbers == data["expected_source_numbers"],
        "confidence": confidence == data["expected_confidence"],
        "citations_resolve": citations_resolve,
    }
    return _result(
        case,
        checks,
        {
            "answer": answer,
            "source_numbers": source_numbers,
            "confidence": confidence,
            "citation_count": len(citations),
        },
    )


def _add_history(
    store: ConversationStore, conversation_id: str, raw_history: list[Any]
) -> None:
    for message in _messages(raw_history, conversation_id):
        store.add_message(
            conversation_id,
            message.role,
            message.content,
            created_at=message.created_at,
        )


def _evaluate_isolation(
    case: ConversationBenchmarkCase, workspace: Path
) -> dict[str, Any]:
    data = case.data
    store = ConversationStore(workspace / f"{case.case_id}.db")
    active = store.create("Active", created_at=FIXED_TIME)
    other = store.create("Other", created_at=FIXED_TIME)
    _add_history(store, active.conversation_id, data["active_history"])
    _add_history(store, other.conversation_id, data["other_history"])
    captured: dict[str, Any] = {}

    def respond(payload: dict[str, Any]) -> dict[str, Any]:
        captured.update(payload)
        return {
            "message": {
                "content": json.dumps(
                    {"standalone_query": data["model_response"]}
                )
            }
        }

    rewrite = rewrite_follow_up(
        data["question"],
        store.list_messages(active.conversation_id),
        send_request=respond,
    )
    prompt = captured["messages"][1]["content"]
    active_texts = [item["content"] for item in data["active_history"]]
    other_texts = [item["content"] for item in data["other_history"]]
    checks = {
        "standalone_query": rewrite.standalone_query
        == data["expected_standalone_query"],
        "active_history_present": all(text in prompt for text in active_texts),
        "other_history_absent": all(text not in prompt for text in other_texts),
        "stored_histories_isolated": all(
            message.conversation_id == active.conversation_id
            for message in store.list_messages(active.conversation_id)
        ),
    }
    return _result(
        case,
        checks,
        {
            "standalone_query": rewrite.standalone_query,
            "active_message_count": len(active_texts),
            "other_message_count": len(other_texts),
        },
    )


def _evaluate_incomplete_generation(
    case: ConversationBenchmarkCase, workspace: Path
) -> dict[str, Any]:
    data = case.data
    store = ConversationStore(workspace / f"{case.case_id}.db")
    conversation = store.create("Generation", created_at=FIXED_TIME)
    pending = store.add_message(
        conversation.conversation_id,
        "assistant",
        data["draft_content"],
        status="pending",
        created_at=FIXED_TIME + timedelta(seconds=1),
    )
    finalized = store.update_assistant_message(
        pending.message_id,
        data["final_content"],
        data["terminal_status"],
        updated_at=FIXED_TIME + timedelta(seconds=2),
    )
    checks = {
        "terminal_status": finalized.status == data["terminal_status"],
        "partial_answer_discarded": data["draft_content"] not in finalized.content,
        "not_completed": finalized.status != "completed",
        "no_citations": finalized.citations == (),
        "no_confidence": finalized.confidence is None,
    }
    return _result(
        case,
        checks,
        {
            "status": finalized.status,
            "content": finalized.content,
            "citation_count": len(finalized.citations),
            "confidence": finalized.confidence,
        },
    )


def evaluate_conversation_case(
    case: ConversationBenchmarkCase, workspace: Path
) -> dict[str, Any]:
    """Evaluate one case through the same helpers used by the application."""
    if case.category in {"follow_up_rewrite", "ambiguous_reference"}:
        return _evaluate_rewrite(case)
    if case.category in {"unsupported_answer", "citation_correctness"}:
        return _evaluate_answer(case)
    if case.category == "conversation_isolation":
        return _evaluate_isolation(case, workspace)
    return _evaluate_incomplete_generation(case, workspace)


def evaluate_conversation_benchmark(
    name: str,
    cases: list[ConversationBenchmarkCase],
    workspace: Path,
) -> dict[str, Any]:
    """Run every case and return a deterministic, gate-friendly report."""
    if not cases:
        raise ValueError("At least one conversation benchmark case is required.")
    workspace.mkdir(parents=True, exist_ok=True)
    results = [evaluate_conversation_case(case, workspace) for case in cases]
    category_accuracy = {
        category: mean(
            1.0 if result["passed"] else 0.0
            for result in results
            if result["category"] == category
        )
        for category in sorted({result["category"] for result in results})
    }
    passed_count = sum(1 for result in results if result["passed"])
    return {
        "schema_version": 1,
        "benchmark": name,
        "summary": {
            "case_count": len(results),
            "passed_count": passed_count,
            "pass_rate": passed_count / len(results),
            "category_accuracy": category_accuracy,
        },
        "results": results,
    }


def write_conversation_report(path: Path, report: dict[str, Any]) -> None:
    """Write a stable human-readable benchmark report."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
