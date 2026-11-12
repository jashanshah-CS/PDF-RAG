import pytest

from src.rag_project.citations import cited_source_numbers


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
