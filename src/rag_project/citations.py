"""Helpers for resolving the numbered sources cited in generated answers."""

from dataclasses import dataclass
import re


@dataclass(frozen=True)
class CitedClaim:
    """One answer claim and its ordered, valid evidence references."""

    claim_number: int
    text: str
    source_numbers: tuple[int, ...]


CLAIM_PATTERN = re.compile(
    r"(?P<claim>.*?\S)\s+(?P<citations>(?:\[Source\s+\d+\]\s*)+)(?=\n|$)",
    re.IGNORECASE | re.MULTILINE,
)

STOPWORDS = frozenset(
    {
        "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
        "in", "is", "it", "of", "on", "or", "that", "the", "this", "to",
        "was", "were", "with",
    }
)


def cited_source_numbers(answer: str, available_count: int) -> tuple[int, ...]:
    """Return unique, valid source numbers in the order they are cited."""
    if available_count < 0:
        raise ValueError("The available source count cannot be negative.")
    numbers: list[int] = []
    for value in re.findall(r"\[Source\s+(\d+)\]", answer, re.I):
        number = int(value)
        if 1 <= number <= available_count and number not in numbers:
            numbers.append(number)
    return tuple(numbers)


def cited_claims(answer: str, available_count: int) -> tuple[CitedClaim, ...]:
    """Parse newline-delimited claims whose citations immediately follow them."""
    if available_count < 0:
        raise ValueError("The available source count cannot be negative.")
    claims: list[CitedClaim] = []
    for match in CLAIM_PATTERN.finditer(answer.strip()):
        numbers = cited_source_numbers(match.group("citations"), available_count)
        text = " ".join(match.group("claim").split())
        if text and numbers:
            claims.append(CitedClaim(len(claims) + 1, text, numbers))
    return tuple(claims)


def _content_terms(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9][a-z0-9+./-]*", text.casefold())
        if token not in STOPWORDS
    }


def calibrated_confidence(answer: str, evidence_texts: dict[int, str]) -> str:
    """Return a discrete evidence-coverage state, never a probability score."""
    normalized = " ".join(answer.casefold().replace("'", "").split())
    refusal_markers = (
        "cannot find", "cant find", "could not find", "no information",
        "not mentioned", "not provided", "not stated", "not specified",
    )
    if any(marker in normalized for marker in refusal_markers):
        return "unsupported"
    claims = cited_claims(answer, max(evidence_texts, default=0))
    if not claims:
        return "unsupported"
    weak_claim = False
    for claim in claims:
        claim_terms = _content_terms(claim.text)
        evidence_terms: set[str] = set()
        for number in claim.source_numbers:
            evidence_terms.update(_content_terms(evidence_texts.get(number, "")))
        overlap = len(claim_terms.intersection(evidence_terms))
        required_overlap = 1 if len(claim_terms) <= 2 else 2
        if overlap == 0:
            return "unsupported"
        coverage = overlap / max(len(claim_terms), 1)
        if overlap < required_overlap or coverage < 0.5:
            weak_claim = True
    return "uncertain" if weak_claim else "supported"
