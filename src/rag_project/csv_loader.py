"""Load CSV sources and answer exact structured-data questions."""

import csv
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from io import StringIO
import re

from src.rag_project.documents import (
    Document,
    SourceLocation,
    SourceType,
    create_document_id,
)


MAX_CSV_BYTES = 2 * 1024 * 1024
MAX_CSV_COLUMNS = 100
MAX_CSV_ROWS = 10_000


class CsvLoadError(ValueError):
    """Raised when uploaded CSV data is unsafe or malformed."""


@dataclass(frozen=True)
class CsvTable:
    """Validated tabular data with stable source metadata."""

    document_id: str
    source_name: str
    columns: tuple[str, ...]
    rows: tuple[tuple[str, ...], ...]
    added_at: datetime

    def row_dict(self, index: int) -> dict[str, str]:
        return dict(zip(self.columns, self.rows[index], strict=True))

    def preview(self, limit: int = 10) -> list[dict[str, str]]:
        return [self.row_dict(index) for index in range(min(limit, len(self.rows)))]

    def to_documents(self) -> list[Document]:
        """Represent each data row as searchable text with an exact row citation."""
        documents: list[Document] = []
        for index, row in enumerate(self.rows):
            row_number = index + 2
            text = " | ".join(
                f"{column}: {value}"
                for column, value in zip(self.columns, row, strict=True)
                if value
            )
            documents.append(
                Document(
                    document_id=self.document_id,
                    source_type=SourceType.CSV,
                    source_name=self.source_name,
                    text=text,
                    location=SourceLocation(row_number=row_number),
                    added_at=self.added_at,
                    metadata={"columns": self.columns},
                )
            )
        return documents


@dataclass(frozen=True)
class CsvCitation:
    """One exact CSV row used by a structured answer."""

    source_name: str
    row_number: int
    values: dict[str, str]

    def label(self) -> str:
        return f"{self.source_name} — row {self.row_number}"


@dataclass(frozen=True)
class StructuredCsvAnswer:
    """A deterministic answer and the precise rows supporting it."""

    answer: str
    citations: tuple[CsvCitation, ...]


def load_csv_file(
    csv_bytes: bytes,
    source_name: str,
    *,
    added_at: datetime | None = None,
) -> CsvTable:
    """Validate and parse one UTF-8 CSV file."""
    if not csv_bytes:
        raise CsvLoadError("The uploaded CSV file is empty.")
    if len(csv_bytes) > MAX_CSV_BYTES:
        raise CsvLoadError("The CSV file is larger than the 2 MB limit.")

    try:
        text = csv_bytes.decode("utf-8-sig")
    except UnicodeDecodeError as error:
        raise CsvLoadError("The CSV file must use UTF-8 text encoding.") from error
    if "\0" in text:
        raise CsvLoadError("The CSV file contains invalid null characters.")

    reader = csv.reader(StringIO(text, newline=""))
    try:
        raw_columns = next(reader)
    except StopIteration as error:
        raise CsvLoadError("The CSV file does not contain a header row.") from error

    columns = tuple(column.strip() for column in raw_columns)
    if not columns or any(not column for column in columns):
        raise CsvLoadError("Every CSV column must have a header.")
    if len(columns) > MAX_CSV_COLUMNS:
        raise CsvLoadError(f"CSV files can contain at most {MAX_CSV_COLUMNS} columns.")
    normalized_columns = [_normalize(column) for column in columns]
    if len(set(normalized_columns)) != len(normalized_columns):
        raise CsvLoadError("CSV column headers must be unique.")

    rows: list[tuple[str, ...]] = []
    for row_number, raw_row in enumerate(reader, start=2):
        if not any(value.strip() for value in raw_row):
            continue
        if len(raw_row) != len(columns):
            raise CsvLoadError(
                f"CSV row {row_number} has {len(raw_row)} values; "
                f"expected {len(columns)}."
            )
        rows.append(tuple(value.strip() for value in raw_row))
        if len(rows) > MAX_CSV_ROWS:
            raise CsvLoadError(f"CSV files can contain at most {MAX_CSV_ROWS:,} rows.")

    if not rows:
        raise CsvLoadError("The CSV file does not contain any data rows.")

    return CsvTable(
        document_id=create_document_id(SourceType.CSV, source_name, csv_bytes),
        source_name=source_name,
        columns=columns,
        rows=tuple(rows),
        added_at=added_at or datetime.now(UTC),
    )


def _normalize(value: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", value.lower()))


def _variants(value: str) -> set[str]:
    normalized = _normalize(value)
    variants = {normalized, normalized + "s"}
    if normalized.endswith("y"):
        variants.add(normalized[:-1] + "ies")
    if normalized.endswith("ies"):
        variants.add(normalized[:-3] + "y")
    elif normalized.endswith("s"):
        variants.add(normalized[:-1])
    return {variant for variant in variants if variant}


def _mentioned_columns(question: str, table: CsvTable) -> list[str]:
    normalized_question = _normalize(question)
    return [
        column
        for column in table.columns
        if any(
            re.search(rf"\b{re.escape(variant)}\b", normalized_question)
            for variant in _variants(column)
        )
    ]


def _parse_number(value: str) -> Decimal | None:
    normalized = value.strip()
    if not normalized:
        return None
    negative = normalized.startswith("(") and normalized.endswith(")")
    normalized = normalized.strip("()").replace(",", "")
    normalized = re.sub(r"^[£$€]\s*", "", normalized)
    normalized = normalized.rstrip("%").strip()
    try:
        number = Decimal(normalized)
    except InvalidOperation:
        return None
    return -number if negative else number


def _numeric_columns(table: CsvTable) -> list[str]:
    numeric: list[str] = []
    for column_index, column in enumerate(table.columns):
        populated = [row[column_index] for row in table.rows if row[column_index]]
        if populated and all(_parse_number(value) is not None for value in populated):
            numeric.append(column)
    return numeric


def _format_number(number: Decimal) -> str:
    if number == number.to_integral():
        return f"{int(number):,}"
    return f"{number.quantize(Decimal('0.01')):,.2f}".rstrip("0").rstrip(".")


def _citation(table: CsvTable, row_index: int) -> CsvCitation:
    return CsvCitation(
        source_name=table.source_name,
        row_number=row_index + 2,
        values=table.row_dict(row_index),
    )


def looks_like_csv_calculation(question: str) -> bool:
    """Detect questions that should never be delegated to approximate generation."""
    normalized = _normalize(question)
    phrases = (
        "average", "mean", "total", "sum", "largest", "highest", "most",
        "smallest", "lowest", "fewest", "minimum", "maximum", "how many",
        "more than", "greater than", "less than", "above", "below",
    )
    return any(
        re.search(rf"\b{re.escape(phrase)}\b", normalized)
        for phrase in phrases
    )


def answer_csv_question(
    question: str,
    tables: list[CsvTable],
) -> StructuredCsvAnswer | None:
    """Answer supported table questions with exact deterministic operations."""
    if len(tables) != 1 or not question.strip():
        return None

    table = tables[0]
    normalized_question = _normalize(question)
    mentioned = _mentioned_columns(question, table)
    numeric_columns = _numeric_columns(table)
    mentioned_numeric = [column for column in mentioned if column in numeric_columns]
    entity_columns = [column for column in table.columns if column not in numeric_columns]
    entity_column = next(
        (column for column in mentioned if column in entity_columns),
        entity_columns[0] if entity_columns else table.columns[0],
    )

    metric = mentioned_numeric[-1] if mentioned_numeric else None
    if metric and any(
        phrase in normalized_question
        for phrase in ("largest", "highest", "most", "maximum")
    ):
        return _extreme_answer(table, entity_column, metric, maximum=True)
    if metric and any(
        phrase in normalized_question
        for phrase in ("smallest", "lowest", "fewest", "minimum")
    ):
        return _extreme_answer(table, entity_column, metric, maximum=False)
    if metric and any(word in normalized_question for word in ("average", "mean")):
        values = _column_numbers(table, metric)
        if values:
            average = sum(value for _, value in values) / Decimal(len(values))
            citations = tuple(_citation(table, index) for index, _ in values)
            return StructuredCsvAnswer(
                f"The average {metric} is {_format_number(average)}.", citations
            )
    if metric and any(word in normalized_question for word in ("total", "sum")):
        values = _column_numbers(table, metric)
        if values:
            total = sum(value for _, value in values)
            citations = tuple(_citation(table, index) for index, _ in values)
            return StructuredCsvAnswer(
                f"The total {metric} is {_format_number(total)}.", citations
            )

    if metric and "how many" in normalized_question:
        values = _column_numbers(table, metric)
        if values:
            total = sum(value for _, value in values)
            return StructuredCsvAnswer(
                f"There are {_format_number(total)} {metric} in total.",
                tuple(_citation(table, index) for index, _ in values),
            )

    comparison = re.search(
        r"\b(more than|greater than|over|above|less than|below|under)\s+"
        r"([£$€]?\s*[0-9][0-9,.]*(?:\.[0-9]+)?%?)",
        question,
        re.IGNORECASE,
    )
    if metric and comparison:
        threshold = _parse_number(comparison.group(2))
        if threshold is not None:
            greater = comparison.group(1).lower() in {
                "more than", "greater than", "over", "above",
            }
            matches = [
                (index, value)
                for index, value in _column_numbers(table, metric)
                if (value > threshold if greater else value < threshold)
            ]
            if not matches:
                return StructuredCsvAnswer(
                    f"No rows have {metric} {'above' if greater else 'below'} "
                    f"{_format_number(threshold)}.",
                    (),
                )
            entities = [
                f"{table.row_dict(index)[entity_column]} ({_format_number(value)})"
                for index, value in matches
            ]
            return StructuredCsvAnswer(
                f"{entity_column} with {metric} "
                f"{'above' if greater else 'below'} {_format_number(threshold)}: "
                + ", ".join(entities)
                + ".",
                tuple(_citation(table, index) for index, _ in matches),
            )

    if "how many" in normalized_question:
        count_column = next((column for column in mentioned if column in entity_columns), None)
        if count_column:
            column_index = table.columns.index(count_column)
            count = len({row[column_index] for row in table.rows if row[column_index]})
            return StructuredCsvAnswer(
                f"There are {count:,} distinct {count_column} values.",
                tuple(_citation(table, index) for index in range(len(table.rows))),
            )
        if any(word in normalized_question for word in ("row", "record", "entry")):
            return StructuredCsvAnswer(
                f"The CSV contains {len(table.rows):,} data rows.",
                tuple(_citation(table, index) for index in range(len(table.rows))),
            )

    target_columns = [column for column in mentioned if column != entity_column]
    for row_index, row in enumerate(table.rows):
        values = table.row_dict(row_index)
        matched_entity = next(
            (
                column
                for column in entity_columns
                if values[column]
                and _normalize(values[column]) in normalized_question
            ),
            None,
        )
        if matched_entity and target_columns:
            target = target_columns[-1]
            return StructuredCsvAnswer(
                f"For {values[matched_entity]}, {target} is {values[target]}.",
                (_citation(table, row_index),),
            )

    return None


def _column_numbers(table: CsvTable, column: str) -> list[tuple[int, Decimal]]:
    column_index = table.columns.index(column)
    values: list[tuple[int, Decimal]] = []
    for row_index, row in enumerate(table.rows):
        number = _parse_number(row[column_index])
        if number is not None:
            values.append((row_index, number))
    return values


def _extreme_answer(
    table: CsvTable,
    entity_column: str,
    metric: str,
    *,
    maximum: bool,
) -> StructuredCsvAnswer | None:
    values = _column_numbers(table, metric)
    if not values:
        return None
    target_value = (max if maximum else min)(value for _, value in values)
    matches = [(index, value) for index, value in values if value == target_value]
    entities = [table.row_dict(index)[entity_column] for index, _ in matches]
    direction = "highest" if maximum else "lowest"
    return StructuredCsvAnswer(
        f"{', '.join(entities)} has the {direction} {metric} at "
        f"{_format_number(target_value)}.",
        tuple(_citation(table, index) for index, _ in matches),
    )
