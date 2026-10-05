from datetime import UTC, datetime

import pytest

from src.rag_project.csv_loader import (
    CsvLoadError,
    answer_csv_question,
    load_csv_file,
    looks_like_csv_calculation,
)
from src.rag_project.documents import SourceLocation, SourceType


CSV_BYTES = (
    b"office,employees,support_team,city\n"
    b"London,120,18,London\n"
    b"Manchester,75,12,Manchester\n"
    b"Birmingham,90,15,Birmingham\n"
)
ADDED_AT = datetime(2026, 10, 5, 12, 30, tzinfo=UTC)


def table():
    return load_csv_file(CSV_BYTES, "offices.csv", added_at=ADDED_AT)


def test_loads_csv_and_preserves_exact_row_metadata() -> None:
    loaded = table()
    documents = loaded.to_documents()

    assert loaded.columns == ("office", "employees", "support_team", "city")
    assert len(loaded.rows) == 3
    assert loaded.preview(1) == [
        {
            "office": "London",
            "employees": "120",
            "support_team": "18",
            "city": "London",
        }
    ]
    assert documents[0].source_type == SourceType.CSV
    assert documents[0].source_name == "offices.csv"
    assert documents[0].location == SourceLocation(row_number=2)
    assert documents[0].added_at == ADDED_AT
    assert documents[0].text == (
        "office: London | employees: 120 | support_team: 18 | city: London"
    )


@pytest.mark.parametrize(
    ("content", "message"),
    [
        (b"", "empty"),
        (b",employees\nLondon,120\n", "header"),
        (b"office,Office\nLondon,London\n", "unique"),
        (b"office,employees\nLondon\n", "expected 2"),
        (b"office,employees\n", "data rows"),
        (b"office\n\xff\n", "UTF-8"),
    ],
)
def test_rejects_malformed_csv(content: bytes, message: str) -> None:
    with pytest.raises(CsvLoadError, match=message):
        load_csv_file(content, "bad.csv")


def test_answers_maximum_with_exact_row_citation() -> None:
    answer = answer_csv_question(
        "Which office has the largest support team?",
        [table()],
    )

    assert answer is not None
    assert answer.answer == "London has the highest support_team at 18."
    assert [citation.row_number for citation in answer.citations] == [2]


def test_answers_minimum_average_and_total() -> None:
    minimum = answer_csv_question("Which office has the fewest employees?", [table()])
    average = answer_csv_question("What is the average employees?", [table()])
    total = answer_csv_question("What is the total employees?", [table()])

    assert minimum is not None and "Manchester" in minimum.answer
    assert average is not None and average.answer == "The average employees is 95."
    assert total is not None and total.answer == "The total employees is 285."
    assert len(average.citations) == 3
    assert len(total.citations) == 3


def test_filters_rows_with_numeric_comparison() -> None:
    answer = answer_csv_question(
        "Which offices have more than 14 support team?",
        [table()],
    )

    assert answer is not None
    assert "London (18)" in answer.answer
    assert "Birmingham (15)" in answer.answer
    assert [citation.row_number for citation in answer.citations] == [2, 4]


def test_counts_distinct_entities() -> None:
    answer = answer_csv_question("How many offices are listed?", [table()])

    assert answer is not None
    assert answer.answer == "There are 3 distinct office values."


def test_how_many_numeric_values_uses_exact_sum() -> None:
    answer = answer_csv_question("How many employees are there?", [table()])

    assert answer is not None
    assert answer.answer == "There are 285 employees in total."


def test_looks_up_one_row_without_model_arithmetic() -> None:
    answer = answer_csv_question(
        "What is the support team for Manchester?",
        [table()],
    )

    assert answer is not None
    assert answer.answer == "For Manchester, support_team is 12."
    assert answer.citations[0].label() == "offices.csv — row 3"


def test_unknown_question_falls_back_to_semantic_search() -> None:
    assert answer_csv_question("Describe the available office data", [table()]) is None


def test_calculation_detection_does_not_match_substrings() -> None:
    assert looks_like_csv_calculation("What is the total employees?") is True
    assert looks_like_csv_calculation("This is almost complete") is False
