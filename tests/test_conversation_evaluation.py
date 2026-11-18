import json
from pathlib import Path

import pytest

from src.rag_project.conversation_evaluation import (
    ConversationBenchmarkCase,
    evaluate_conversation_benchmark,
    load_conversation_benchmark,
    write_conversation_report,
)
from src.rag_project.quality_gates import evaluate_quality_gates, load_json_object


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CASES_PATH = PROJECT_ROOT / "evaluation" / "conversation_v4_cases.json"
POLICY_PATH = PROJECT_ROOT / "evaluation" / "conversation_v4_quality_gates.json"


def test_committed_conversation_benchmark_passes_every_quality_gate(
    tmp_path: Path,
) -> None:
    name, cases = load_conversation_benchmark(CASES_PATH)
    report = evaluate_conversation_benchmark(name, cases, tmp_path)
    policy = load_json_object(POLICY_PATH, label="conversation quality policy")

    assert report["summary"] == {
        "case_count": 14,
        "passed_count": 14,
        "pass_rate": 1.0,
        "category_accuracy": {
            "ambiguous_reference": 1.0,
            "citation_correctness": 1.0,
            "conversation_isolation": 1.0,
            "follow_up_rewrite": 1.0,
            "incomplete_generation": 1.0,
            "unsupported_answer": 1.0,
        },
    }
    assert evaluate_quality_gates(report, policy) == []


def test_report_contains_resolved_citations_and_isolation_evidence(
    tmp_path: Path,
) -> None:
    name, cases = load_conversation_benchmark(CASES_PATH)
    report = evaluate_conversation_benchmark(name, cases, tmp_path)
    by_id = {item["id"]: item for item in report["results"]}

    citation = by_id["citation-multiple-claims"]
    isolation = by_id["isolation-private-topic"]

    assert citation["checks"]["citations_resolve"] is True
    assert citation["actual"]["source_numbers"] == [1, 2]
    assert isolation["checks"]["other_history_absent"] is True
    assert isolation["checks"]["stored_histories_isolated"] is True


def test_regression_is_reported_instead_of_hidden(tmp_path: Path) -> None:
    broken = ConversationBenchmarkCase(
        case_id="broken-follow-up",
        category="follow_up_rewrite",
        data={
            "question": "What about contractors?",
            "history": [
                {"role": "user", "content": "Explain remote work."},
            ],
            "model_response": "Does remote work include contractors?",
            "expected_standalone_query": "A deliberately wrong expectation",
            "expected_rewritten": True,
        },
    )

    report = evaluate_conversation_benchmark("regression", [broken], tmp_path)

    assert report["summary"]["passed_count"] == 0
    assert report["summary"]["pass_rate"] == 0.0
    assert report["results"][0]["checks"]["standalone_query"] is False


def test_loader_rejects_wrong_schema_duplicate_ids_and_unknown_category(
    tmp_path: Path,
) -> None:
    invalid = tmp_path / "invalid.json"
    invalid.write_text('{"schema_version": 2}', encoding="utf-8")
    with pytest.raises(ValueError, match="schema_version must be 1"):
        load_conversation_benchmark(invalid)

    duplicate_payload = json.loads(CASES_PATH.read_text(encoding="utf-8"))
    duplicate_payload["cases"].append(duplicate_payload["cases"][0])
    invalid.write_text(json.dumps(duplicate_payload), encoding="utf-8")
    with pytest.raises(ValueError, match="Duplicate conversation benchmark"):
        load_conversation_benchmark(invalid)

    duplicate_payload["cases"][-1] = {
        "id": "unknown",
        "category": "magic",
    }
    invalid.write_text(json.dumps(duplicate_payload), encoding="utf-8")
    with pytest.raises(ValueError, match="Unsupported conversation category"):
        load_conversation_benchmark(invalid)


def test_loader_rejects_invalid_category_specific_fields(tmp_path: Path) -> None:
    payload = {
        "schema_version": 1,
        "name": "invalid",
        "cases": [
            {
                "id": "bad-status",
                "category": "incomplete_generation",
                "terminal_status": "completed",
                "draft_content": "draft",
                "final_content": "final",
            }
        ],
    }
    path = tmp_path / "invalid-fields.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="failed or cancelled"):
        load_conversation_benchmark(path)


def test_report_writer_creates_stable_json(tmp_path: Path) -> None:
    report = {"schema_version": 1, "summary": {"pass_rate": 1.0}}
    output = tmp_path / "nested" / "report.json"

    write_conversation_report(output, report)

    assert json.loads(output.read_text(encoding="utf-8")) == report
    assert output.read_text(encoding="utf-8").endswith("\n")


def test_evaluator_rejects_empty_case_list(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="At least one"):
        evaluate_conversation_benchmark("empty", [], tmp_path)
