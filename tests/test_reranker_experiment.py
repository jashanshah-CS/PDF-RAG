import pytest

from src.rag_project.reranker_experiment import compare_reranker_reports


def report(retriever: str, recall: float, mrr: float, latency: float) -> dict:
    return {
        "retriever": retriever,
        "top_k": 5,
        "summary": {
            "recall_at_5": recall,
            "mean_reciprocal_rank": mrr,
            "latency_ms": {"mean": latency},
        },
    }


def test_adopts_only_meaningful_quality_gain_with_bounded_latency() -> None:
    decision = compare_reranker_reports(
        report("reranked", 0.9, 0.8, 100),
        report("cross-encoder", 0.9, 0.85, 250),
    )

    assert decision["adopt_cross_encoder"] is True
    assert decision["decision"] == "adopt"


def test_retains_deterministic_when_quality_does_not_improve() -> None:
    decision = compare_reranker_reports(
        report("reranked", 1.0, 1.0, 100),
        report("cross-encoder", 1.0, 1.0, 500),
    )

    assert decision["adopt_cross_encoder"] is False
    assert "MRR gain" in " ".join(decision["reasons"])
    assert "latency" in " ".join(decision["reasons"])


def test_rejects_reports_for_wrong_retrievers() -> None:
    with pytest.raises(ValueError, match="deterministic"):
        compare_reranker_reports(
            report("semantic", 1.0, 1.0, 100),
            report("cross-encoder", 1.0, 1.0, 100),
        )
