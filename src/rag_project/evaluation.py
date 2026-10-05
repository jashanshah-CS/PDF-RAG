"""Transparent evaluation helpers for PDF and website RAG pipelines."""

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
    expected_sources: tuple[str, ...] = ()
    expected_pages: tuple[int, ...] = ()
    expected_urls: tuple[str, ...] = ()


@dataclass(frozen=True)
class EvaluationResult:
    """Recorded output and automatic checks for one question."""

    question: str
    expected_keywords: str
    expected_page: int | None
    expected_sources: str
    expected_pages: str
    expected_urls: str
    should_refuse: bool
    actual_answer: str
    retrieved_sources: str
    retrieved_pages: str
    retrieved_urls: str
    cited_sources: str
    cited_pages: str
    cited_urls: str
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
        required = {"question", "expected_keywords", "should_refuse"}
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
            page_text = (row.get("expected_page") or "").strip()
            expected_pages = tuple(
                int(page.strip())
                for page in (row.get("expected_pages") or "").split("|")
                if page.strip()
            )
            expected_sources = tuple(
                source.strip()
                for source in (row.get("expected_sources") or "").split("|")
                if source.strip()
            )
            expected_urls = tuple(
                url.strip()
                for url in (row.get("expected_urls") or "").split("|")
                if url.strip()
            )

            cases.append(
                EvaluationCase(
                    question=question,
                    expected_keywords=keywords,
                    expected_page=int(page_text) if page_text else None,
                    should_refuse=parse_bool(row["should_refuse"]),
                    expected_sources=expected_sources,
                    expected_pages=expected_pages,
                    expected_urls=expected_urls,
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
    pages = {
        results[number - 1].embedded_chunk.chunk.location.page_number
        for number in source_numbers
        if 1 <= number <= len(results)
    }
    return sorted(page for page in pages if page is not None)


def cited_results(answer: str, results: list[SearchResult]) -> list[SearchResult]:
    """Resolve model source labels to the retrieved results they reference."""
    source_numbers = sorted(
        {int(match) for match in re.findall(r"\[Source\s+(\d+)\]", answer, re.I)}
    )
    return [results[number - 1] for number in source_numbers if 1 <= number <= len(results)]


def _source_values(results: list[SearchResult]) -> tuple[list[str], list[int], list[str]]:
    sources = sorted({result.embedded_chunk.chunk.source_name for result in results})
    pages = sorted(
        page
        for page in {
            result.embedded_chunk.chunk.location.page_number for result in results
        }
        if page is not None
    )
    urls = sorted(
        url
        for url in {
            result.embedded_chunk.chunk.location.url for result in results
        }
        if url is not None
    )
    return sources, pages, urls


def _contains_all(actual: list[str], expected: tuple[str, ...]) -> bool:
    normalized_actual = [value.casefold().rstrip("/") for value in actual]
    return all(
        any(
            expected_value.casefold().rstrip("/") in actual_value
            for actual_value in normalized_actual
        )
        for expected_value in expected
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
    retrieved_sources, retrieved_pages, retrieved_urls = _source_values(results)
    cited_sources, cited_page_values, cited_urls = _source_values(
        cited_results(answer, results)
    )
    normalized_answer = normalize_for_keyword_check(answer)
    refusal_detected = is_refusal_answer(answer)
    refusal_correct = refusal_detected == case.should_refuse

    expected_pages = case.expected_pages or (
        (case.expected_page,) if case.expected_page is not None else ()
    )
    has_source_expectation = bool(
        case.expected_sources or expected_pages or case.expected_urls
    )
    retrieval_hit = (
        _contains_all(retrieved_sources, case.expected_sources)
        and all(page in retrieved_pages for page in expected_pages)
        and _contains_all(retrieved_urls, case.expected_urls)
        if has_source_expectation
        else None
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
        _contains_all(cited_sources, case.expected_sources)
        and all(page in cited_page_values for page in expected_pages)
        and _contains_all(cited_urls, case.expected_urls)
        if has_source_expectation
        else None
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
        expected_sources="|".join(case.expected_sources),
        expected_pages="|".join(str(page) for page in expected_pages),
        expected_urls="|".join(case.expected_urls),
        should_refuse=case.should_refuse,
        actual_answer=answer,
        retrieved_sources="|".join(retrieved_sources),
        retrieved_pages="|".join(str(page) for page in retrieved_pages),
        retrieved_urls="|".join(retrieved_urls),
        cited_sources="|".join(cited_sources),
        cited_pages="|".join(str(page) for page in cited_page_values),
        cited_urls="|".join(cited_urls),
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
