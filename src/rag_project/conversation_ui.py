"""Deterministic helpers shared by the Version 4 conversation interface."""

from src.rag_project.citations import cited_claims, cited_source_numbers
from src.rag_project.conversation_store import Conversation, MessageCitation
from src.rag_project.search import SearchResult
from src.rag_project.source_selection import reconcile_source_selection


def choose_active_conversation(
    conversations: list[Conversation], requested_id: str | None
) -> str | None:
    """Keep a valid requested conversation, otherwise choose the most recent."""
    available = {conversation.conversation_id for conversation in conversations}
    if requested_id in available:
        return requested_id
    return conversations[0].conversation_id if conversations else None


def conversation_title(question: str, *, maximum_length: int = 56) -> str:
    """Create a compact deterministic title from the first user question."""
    if maximum_length < 4:
        raise ValueError("Conversation title length must be at least four characters.")
    cleaned = " ".join(question.split())
    if not cleaned:
        return "New conversation"
    if len(cleaned) <= maximum_length:
        return cleaned
    return cleaned[: maximum_length - 1].rstrip(" ,.;:-") + "…"


def reconcile_conversation_sources(
    *,
    conversation_id: str,
    loaded_conversation_id: str | None,
    saved_keys: tuple[str, ...],
    session_keys: list[str],
    available_keys: list[str],
    newly_added_keys: list[str] | None = None,
) -> list[str]:
    """Load a switched conversation's scope, then maintain its session scope."""
    starting_keys = (
        list(saved_keys)
        if loaded_conversation_id != conversation_id
        else session_keys
    )
    return reconcile_source_selection(
        starting_keys,
        available_keys,
        newly_added_keys,
    )


def message_citations(
    answer: str, results: list[SearchResult]
) -> tuple[MessageCitation, ...]:
    """Convert answer source labels into durable conversation citations."""
    citations: list[MessageCitation] = []
    claims = cited_claims(answer, len(results))
    if claims:
        for claim in claims:
            for number in claim.source_numbers:
                chunk = results[number - 1].embedded_chunk.chunk
                citations.append(
                    MessageCitation(
                        source_name=chunk.source_name,
                        source_type=chunk.source_type.value,
                        location=chunk.location.label(),
                        page_number=chunk.location.page_number,
                        url=chunk.location.url,
                        chunk_number=chunk.chunk_number,
                        source_number=number,
                        excerpt=chunk.text,
                        claim_number=claim.claim_number,
                        claim_text=claim.text,
                    )
                )
    else:
        for number in cited_source_numbers(answer, len(results)):
            chunk = results[number - 1].embedded_chunk.chunk
            citations.append(
                MessageCitation(
                    source_name=chunk.source_name,
                    source_type=chunk.source_type.value,
                    location=chunk.location.label(),
                    page_number=chunk.location.page_number,
                    url=chunk.location.url,
                    chunk_number=chunk.chunk_number,
                    source_number=number,
                    excerpt=chunk.text,
                    claim_text=answer,
                )
            )
    return tuple(citations)
