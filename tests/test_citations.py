import pytest

from src.rag_project.citations import (
    calibrated_confidence,
    cited_claims,
    cited_source_numbers,
)


def test_returns_unique_valid_sources_in_answer_order() -> None:
    assert cited_source_numbers(
        "Policy A [Source 2], policy B [Source 1], again [Source 2].",
        2,
    ) == (2, 1)


def test_ignores_missing_and_out_of_range_sources() -> None:
    assert cited_source_numbers("No citation [Source 0] [Source 4].", 3) == ()


def test_rejects_negative_source_count() -> None:
    with pytest.raises(ValueError, match="cannot be negative"):
        cited_source_numbers("Answer [Source 1]", -1)


def test_parses_source_relationships_for_each_claim() -> None:
    claims = cited_claims(
        "Remote work is allowed. [Source 2]\nLeave is 25 days. "
        "[Source 1] [Source 2]",
        2,
    )

    assert [(claim.text, claim.source_numbers) for claim in claims] == [
        ("Remote work is allowed.", (2,)),
        ("Leave is 25 days.", (1, 2)),
    ]


def test_confidence_is_discrete_and_based_on_claim_evidence_overlap() -> None:
    evidence = {1: "Employees receive 25 days of annual leave."}

    assert calibrated_confidence(
        "Employees receive 25 days of annual leave. [Source 1]", evidence
    ) == "supported"
    assert calibrated_confidence(
        "Employees receive a generous documented allowance. [Source 1]", evidence
    ) == "uncertain"
    assert calibrated_confidence(
        "Pizza is served on Friday. [Source 1]", evidence
    ) == "unsupported"
    assert calibrated_confidence(
        "I cannot find this in the supplied sources.", evidence
    ) == "unsupported"
