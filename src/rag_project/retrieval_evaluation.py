"""Deterministic metrics for measuring retrieval quality before generation."""

from collections.abc import Callable
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import re
from statistics import mean, median
from time import perf_counter
from typing import Any

from src.rag_project.search import SearchResult


SUPPORTED_CATEGORIES = {
    "semantic",
    "exact",
    "context",
    "multi_source",
    "unsupported",
}


@dataclass(frozen=True)
class ExpectedLocation:
    """One source location that should be represented in the search results."""

    source: str
    page: int | None = None
    url: str | None = None
    evidence: tuple[str, ...] = ()


@dataclass(frozen=True)
class RetrievalCase:
    """A benchmark question and its human-authored relevance judgements."""

    case_id: str
    category: str
    question: str
    expected_locations: tuple[ExpectedLocation, ...]

    @property
    def is_supported(self) -> bool:
        return bool(self.expected_locations)


@dataclass(frozen=True)
class RetrievedLocation:
    """Serializable view of one ranked search result."""

    rank: int
    source: str
    page: int | None
    url: str | None
    score: float
    text_preview: str


@dataclass(frozen=True)
class RetrievalCaseResult:
    """Detailed metrics for a single benchmark question."""

    case_id: str
    category: str
    question: str
    supported: bool
    expected_count: int
    matched_count: int
    source_recall: float | None
    location_recall: float | None
    evidence_recall: float | None
    reciprocal_rank: float | None
    latency_ms: float
    top_score: float | None
    retrieved: tuple[RetrievedLocation, ...]


def _required_text(value: Any, field: str, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string in {context}.")
    return value.strip()


def load_retrieval_cases(path: Path) -> list[RetrievalCase]:
    """Load and strictly validate the versioned JSON benchmark format."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exception:
        raise ValueError(f"Invalid benchmark JSON: {exception.msg}.") from exception

    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError("Benchmark schema_version must be 1.")
    raw_cases = payload.get("cases")
    if not isinstance(raw_cases, list) or not raw_cases:
        raise ValueError("Benchmark cases must be a non-empty list.")

    cases: list[RetrievalCase] = []
    seen_ids: set[str] = set()
    for index, raw_case in enumerate(raw_cases, start=1):
        context = f"case {index}"
        if not isinstance(raw_case, dict):
            raise ValueError(f"{context} must be an object.")
        case_id = _required_text(raw_case.get("id"), "id", context)
        if case_id in seen_ids:
            raise ValueError(f"Duplicate benchmark case id: {case_id}.")
        seen_ids.add(case_id)

        category = _required_text(raw_case.get("category"), "category", context)
        if category not in SUPPORTED_CATEGORIES:
            raise ValueError(f"Unsupported category {category!r} in {context}.")
        question = _required_text(raw_case.get("question"), "question", context)

        raw_locations = raw_case.get("expected_locations", [])
        if not isinstance(raw_locations, list):
            raise ValueError(f"expected_locations must be a list in {context}.")
        if category == "unsupported" and raw_locations:
            raise ValueError(f"Unsupported {context} cannot have expected locations.")
        if category != "unsupported" and not raw_locations:
            raise ValueError(f"Supported {context} needs at least one expected location.")

        locations: list[ExpectedLocation] = []
        for location_index, raw_location in enumerate(raw_locations, start=1):
            location_context = f"{context}, location {location_index}"
            if not isinstance(raw_location, dict):
                raise ValueError(f"{location_context} must be an object.")
            source = _required_text(
                raw_location.get("source"), "source", location_context
            )
            page = raw_location.get("page")
            if page is not None and (not isinstance(page, int) or page <= 0):
                raise ValueError(f"page must be a positive integer in {location_context}.")
            url = raw_location.get("url")
            if url is not None:
                url = _required_text(url, "url", location_context)
            raw_evidence = raw_location.get("evidence", [])
            if not isinstance(raw_evidence, list) or not all(
                isinstance(item, str) and item.strip() for item in raw_evidence
            ):
                raise ValueError(
                    f"evidence must be a list of non-empty strings in {location_context}."
                )
            locations.append(
                ExpectedLocation(
                    source=source,
                    page=page,
                    url=url,
                    evidence=tuple(item.strip() for item in raw_evidence),
                )
            )

        cases.append(
            RetrievalCase(
                case_id=case_id,
                category=category,
                question=question,
                expected_locations=tuple(locations),
            )
        )
    return cases


def _normalize(value: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9]+", " ", value.casefold()).split())


def _same_source(result: SearchResult, expected: ExpectedLocation) -> bool:
    return result.embedded_chunk.chunk.source_name.casefold() == expected.source.casefold()


def _same_location(result: SearchResult, expected: ExpectedLocation) -> bool:
    location = result.embedded_chunk.chunk.location
    return (
        _same_source(result, expected)
        and (expected.page is None or location.page_number == expected.page)
        and (
            expected.url is None
            or (location.url or "").casefold().rstrip("/")
            == expected.url.casefold().rstrip("/")
        )
    )


def _contains_evidence(result: SearchResult, expected: ExpectedLocation) -> bool:
    if not _same_location(result, expected):
        return False
    actual = _normalize(result.embedded_chunk.chunk.text)
    return all(_normalize(fragment) in actual for fragment in expected.evidence)


def _coverage(
    expected: tuple[ExpectedLocation, ...],
    results: list[SearchResult],
    matcher: Callable[[SearchResult, ExpectedLocation], bool],
) -> float:
    return mean(
        any(matcher(result, target) for result in results) for target in expected
    )


def evaluate_retrieval_case(
    case: RetrievalCase,
    search: Callable[[str], list[SearchResult]],
) -> RetrievalCaseResult:
    """Run one search and calculate source, location, evidence, and rank metrics."""
    started = perf_counter()
    results = search(case.question)
    latency_ms = (perf_counter() - started) * 1000

    retrieved = tuple(
        RetrievedLocation(
            rank=rank,
            source=result.embedded_chunk.chunk.source_name,
            page=result.embedded_chunk.chunk.location.page_number,
            url=result.embedded_chunk.chunk.location.url,
            score=round(result.score, 6),
            text_preview=" ".join(result.embedded_chunk.chunk.text.split())[:240],
        )
        for rank, result in enumerate(results, start=1)
    )
    top_score = results[0].score if results else None

    if not case.is_supported:
        return RetrievalCaseResult(
            case_id=case.case_id,
            category=case.category,
            question=case.question,
            supported=False,
            expected_count=0,
            matched_count=0,
            source_recall=None,
            location_recall=None,
            evidence_recall=None,
            reciprocal_rank=None,
            latency_ms=round(latency_ms, 3),
            top_score=round(top_score, 6) if top_score is not None else None,
            retrieved=retrieved,
        )

    first_relevant_rank = next(
        (
            rank
            for rank, result in enumerate(results, start=1)
            if any(_contains_evidence(result, target) for target in case.expected_locations)
        ),
        None,
    )
    matched_count = sum(
        any(_contains_evidence(result, target) for result in results)
        for target in case.expected_locations
    )
    return RetrievalCaseResult(
        case_id=case.case_id,
        category=case.category,
        question=case.question,
        supported=True,
        expected_count=len(case.expected_locations),
        matched_count=matched_count,
        source_recall=round(
            _coverage(case.expected_locations, results, _same_source), 6
        ),
        location_recall=round(
            _coverage(case.expected_locations, results, _same_location), 6
        ),
        evidence_recall=round(matched_count / len(case.expected_locations), 6),
        reciprocal_rank=(
            round(1 / first_relevant_rank, 6) if first_relevant_rank else 0.0
        ),
        latency_ms=round(latency_ms, 3),
        top_score=round(top_score, 6) if top_score is not None else None,
        retrieved=retrieved,
    )


def _percentile(values: list[float], proportion: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, round(proportion * (len(ordered) - 1))))
    return ordered[index]


def summarize_results(results: list[RetrievalCaseResult], top_k: int) -> dict[str, Any]:
    """Aggregate macro retrieval metrics and latency statistics."""
    if not results:
        raise ValueError("At least one retrieval result is required.")
    supported = [result for result in results if result.supported]
    unsupported = [result for result in results if not result.supported]
    multi_source = [
        result
        for result in supported
        if result.category == "multi_source" or result.expected_count > 1
    ]
    latency_values = [result.latency_ms for result in results]
    unsupported_scores = [
        result.top_score for result in unsupported if result.top_score is not None
    ]
    return {
        "case_count": len(results),
        "supported_case_count": len(supported),
        "unsupported_case_count": len(unsupported),
        f"recall_at_{top_k}": round(mean(r.evidence_recall for r in supported), 6),
        "mean_reciprocal_rank": round(mean(r.reciprocal_rank for r in supported), 6),
        f"source_accuracy_at_{top_k}": round(mean(r.source_recall for r in supported), 6),
        f"location_accuracy_at_{top_k}": round(
            mean(r.location_recall for r in supported), 6
        ),
        f"multi_source_coverage_at_{top_k}": (
            round(mean(r.source_recall for r in multi_source), 6)
            if multi_source
            else None
        ),
        "latency_ms": {
            "mean": round(mean(latency_values), 3),
            "median": round(median(latency_values), 3),
            "p95": round(_percentile(latency_values, 0.95), 3),
            "max": round(max(latency_values), 3),
        },
        "unsupported_top_score_mean": (
            round(mean(unsupported_scores), 6) if unsupported_scores else None
        ),
    }


def build_report(
    results: list[RetrievalCaseResult],
    *,
    benchmark_name: str,
    top_k: int,
    corpus: list[str],
    retriever: str | None = None,
) -> dict[str, Any]:
    """Create a JSON-serializable report with explicit benchmark metadata."""
    report = {
        "schema_version": 1,
        "benchmark": benchmark_name,
        "top_k": top_k,
        "corpus": corpus,
        "summary": summarize_results(results, top_k),
        "results": [asdict(result) for result in results],
    }
    if retriever is not None:
        report["retriever"] = retriever
    return report


def write_report(path: Path, report: dict[str, Any]) -> None:
    """Write a stable, human-readable JSON report."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
