"""Deterministic adoption decision for the optional learned reranker."""

from typing import Any


def compare_reranker_reports(
    deterministic: dict[str, Any],
    learned: dict[str, Any],
    *,
    minimum_mrr_gain: float = 0.01,
    maximum_latency_ratio: float = 3.0,
) -> dict[str, Any]:
    """Adopt only for measurable quality gain without recall regression."""
    if deterministic.get("retriever") != "reranked":
        raise ValueError("The deterministic report must use the reranked retriever.")
    if learned.get("retriever") != "cross-encoder":
        raise ValueError("The learned report must use the cross-encoder retriever.")
    base = deterministic["summary"]
    candidate = learned["summary"]
    top_k = int(deterministic["top_k"])
    recall_key = f"recall_at_{top_k}"
    base_recall = float(base[recall_key])
    candidate_recall = float(candidate[recall_key])
    base_mrr = float(base["mean_reciprocal_rank"])
    candidate_mrr = float(candidate["mean_reciprocal_rank"])
    base_latency = float(base["latency_ms"]["mean"])
    candidate_latency = float(candidate["latency_ms"]["mean"])
    latency_ratio = candidate_latency / base_latency if base_latency else float("inf")
    mrr_gain = candidate_mrr - base_mrr
    adopted = (
        candidate_recall >= base_recall
        and mrr_gain >= minimum_mrr_gain
        and latency_ratio <= maximum_latency_ratio
    )
    reasons = []
    if candidate_recall < base_recall:
        reasons.append("recall regressed")
    if mrr_gain < minimum_mrr_gain:
        reasons.append("MRR gain was below the minimum meaningful improvement")
    if latency_ratio > maximum_latency_ratio:
        reasons.append("latency exceeded the allowed multiplier")
    return {
        "adopt_cross_encoder": adopted,
        "decision": "adopt" if adopted else "retain-deterministic",
        "deterministic_recall_at_k": base_recall,
        "cross_encoder_recall_at_k": candidate_recall,
        "deterministic_mrr": base_mrr,
        "cross_encoder_mrr": candidate_mrr,
        "mrr_gain": mrr_gain,
        "deterministic_mean_latency_ms": base_latency,
        "cross_encoder_mean_latency_ms": candidate_latency,
        "latency_ratio": latency_ratio,
        "minimum_mrr_gain": minimum_mrr_gain,
        "maximum_latency_ratio": maximum_latency_ratio,
        "reasons": reasons,
    }
