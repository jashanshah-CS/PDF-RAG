"""Transparent evaluation helpers for the Version 1 RAG pipeline."""

from collections.abc import Callable
import csv
from dataclasses import asdict, dataclass
from pathlib import Path
import re
from time import perf_counter

from src.rag_project.embeddings import EmbeddedChunk, TextEncoder
from src.rag_project.generator import generate_grounded_answer, is_refusal_answer
from src.rag_project.search import SearchResult, semantic_search


@dataclass(frozen=True)
class EvaluationCase:
    """One question and its expected, human-authored checks."""

    question: str
    expected_keywords: tuple[str, ...]
    expected_page: int | None
    should_refuse: bool


@dataclass(frozen=True)
class EvaluationResult:
    """Recorded output and automatic checks for one question."""

    question: str
    expected_keywords: str
    expected_page: int | None
    should_refuse: bool
    actual_answer: str
    retrieved_pages: str
    cited_pages: str
    retrieval_hit: bool | None
    keyword_hit: bool | None
    citation_hit: bool | None
    refusal_detected: bool
    refusal_correct: bool
    elapsed_seconds: float
    passed: bool
    error: str


def parse_bool(value: str) -> bool:
    """Parse a human-friendly CSV boolean."""
    normalized = value.strip().lower()
    if normalized in {"true", "yes", "1"}:
        return True
    if normalized in {"false", "no", "0"}:
        return False
    raise ValueError(f"Invalid boolean value: {value!r}")


def load_evaluation_cases(path: Path) -> list[EvaluationCase]:
    """Load and validate evaluation questions from a CSV file."""
    cases: list[EvaluationCase] = []

    with path.open(encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        required = {
            "question",
            "expected_keywords",
            "expected_page",
            "should_refuse",
        }
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing CSV columns: {', '.join(sorted(missing))}")

        for row_number, row in enumerate(reader, start=2):
            question = row["question"].strip()
            if not question:
                raise ValueError(f"Question is empty on CSV row {row_number}.")

            keywords = tuple(
                keyword.strip()
                for keyword in row["expected_keywords"].split("|")
                if keyword.strip()
            )
            page_text = row["expected_page"].strip()

            cases.append(
                EvaluationCase(
                    question=question,
                    expected_keywords=keywords,
                    expected_page=int(page_text) if page_text else None,
                    should_refuse=parse_bool(row["should_refuse"]),
                )
            )

    if not cases:
        raise ValueError("The evaluation CSV contains no questions.")

    return cases


def cited_pages(answer: str, results: list[SearchResult]) -> list[int]:
    """Resolve model citations such as [Source 2] to PDF page numbers."""
    source_numbers = {
        int(match) for match in re.findall(r"\[Source\s+(\d+)\]", answer, re.I)
    }
    return sorted(
        {
            results[number - 1].embedded_chunk.chunk.page_number
            for number in source_numbers
            if 1 <= number <= len(results)
        }
    )


def normalize_for_keyword_check(text: str) -> str:
    """Normalize punctuation so equivalent forms such as multi-tenant match."""
    normalized = text.lower().replace("£", " gbp ")
    normalized = re.sub(r"[^a-z0-9+%]+", " ", normalized)
    return " ".join(normalized.split())


def evaluate_case(
    case: EvaluationCase,
    embedded_chunks: list[EmbeddedChunk],
    model: TextEncoder,
    answer_question: Callable[[str, list[SearchResult]], str] = (
        generate_grounded_answer
    ),
) -> EvaluationResult:
    """Run one question and calculate deterministic quality checks."""
    started = perf_counter()

    results: list[SearchResult] = []
    answer = ""
    error = ""

    try:
        results = semantic_search(case.question, embedded_chunks, model)
    except Exception as exception:
        error = str(exception)

    if not error:
        try:
            answer = answer_question(case.question, results)
        except Exception as exception:
            error = str(exception)

    elapsed = perf_counter() - started
    retrieved = sorted(
        {result.embedded_chunk.chunk.page_number for result in results}
    )
    cited = cited_pages(answer, results)
    answer_lower = answer.lower()
    normalized_answer = normalize_for_keyword_check(answer)
    refusal_detected = is_refusal_answer(answer)
    refusal_correct = refusal_detected == case.should_refuse

    retrieval_hit = (
        case.expected_page in retrieved if case.expected_page is not None else None
    )
    keyword_hit = (
        all(
            normalize_for_keyword_check(keyword) in normalized_answer
            for keyword in case.expected_keywords
        )
        if case.expected_keywords
        else None
    )
    citation_hit = (
        case.expected_page in cited if case.expected_page is not None else None
    )

    if case.should_refuse:
        passed = refusal_correct and not error
    else:
        checks = [retrieval_hit, keyword_hit, citation_hit, not refusal_detected]
        passed = all(check is not False for check in checks) and not error

    return EvaluationResult(
        question=case.question,
        expected_keywords="|".join(case.expected_keywords),
        expected_page=case.expected_page,
        should_refuse=case.should_refuse,
        actual_answer=answer,
        retrieved_pages="|".join(str(page) for page in retrieved),
        cited_pages="|".join(str(page) for page in cited),
        retrieval_hit=retrieval_hit,
        keyword_hit=keyword_hit,
        citation_hit=citation_hit,
        refusal_detected=refusal_detected,
        refusal_correct=refusal_correct,
        elapsed_seconds=round(elapsed, 3),
        passed=passed,
        error=error,
    )


def write_results(path: Path, results: list[EvaluationResult]) -> None:
    """Write detailed evaluation results to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [asdict(result) for result in results]

    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
