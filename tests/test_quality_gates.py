import json
from pathlib import Path

import pytest

from src.rag_project.quality_gates import evaluate_quality_gates, load_json_object


PROJECT_ROOT = Path(__file__).resolve().parents[1]
POLICY_PATH = PROJECT_ROOT / "evaluation" / "retrieval_v3_quality_gates.json"
REPORT_PATH = PROJECT_ROOT / "evaluation" / "retrieval_v3_reranked_results.json"


def policy() -> dict:
    return load_json_object(POLICY_PATH, label="quality-gate policy")


def report() -> dict:
    return load_json_object(REPORT_PATH, label="retrieval report")


def test_committed_reranker_report_passes_quality_gates() -> None:
    assert evaluate_quality_gates(report(), policy()) == []


def test_reports_every_regressed_or_missing_metric() -> None:
    regressed = report()
    regressed["summary"]["recall_at_5"] = 0.95
    del regressed["summary"]["mean_reciprocal_rank"]

    failures = evaluate_quality_gates(regressed, policy())

    assert "summary.recall_at_5: 0.95 < minimum 1.0" in failures
    assert "summary.mean_reciprocal_rank: missing (minimum 1.0)" in failures


def test_rejects_wrong_retriever_and_case_count() -> None:
    wrong_report = report()
    wrong_report["retriever"] = "semantic"
    wrong_report["summary"]["case_count"] = 19

    failures = evaluate_quality_gates(wrong_report, policy())

    assert "retriever: 'semantic' != required 'reranked'" in failures
    assert "summary.case_count: 19 != required 20" in failures


def test_rejects_non_numeric_measured_value() -> None:
    invalid_report = report()
    invalid_report["summary"]["recall_at_5"] = "perfect"

    failures = evaluate_quality_gates(invalid_report, policy())

    assert "summary.recall_at_5: 'perfect' is not numeric" in failures


def test_rejects_invalid_policy_schema() -> None:
    invalid_policy = policy()
    invalid_policy["schema_version"] = 2

    with pytest.raises(ValueError, match="schema_version must be 1"):
        evaluate_quality_gates(report(), invalid_policy)


def test_load_json_object_explains_invalid_json(tmp_path: Path) -> None:
    invalid = tmp_path / "invalid.json"
    invalid.write_text("{", encoding="utf-8")

    with pytest.raises(ValueError, match="Invalid test JSON"):
        load_json_object(invalid, label="test")


def test_policy_file_is_valid_json() -> None:
    json.dumps(policy())
