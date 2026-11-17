"""Compare deterministic and learned reranker benchmark reports."""

import argparse
import json
from pathlib import Path

from src.rag_project.reranker_experiment import compare_reranker_reports


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--deterministic",
        type=Path,
        default=Path("evaluation/retrieval_v3_reranked_results.json"),
    )
    parser.add_argument(
        "--cross-encoder",
        type=Path,
        default=Path("evaluation/retrieval_v4_cross_encoder_results.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("evaluation/retrieval_v4_reranker_decision.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    deterministic = json.loads(args.deterministic.read_text(encoding="utf-8"))
    learned = json.loads(args.cross_encoder.read_text(encoding="utf-8"))
    decision = compare_reranker_reports(deterministic, learned)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(decision, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Decision: {decision['decision']}")
    print(f"MRR gain: {decision['mrr_gain']:.3f}")
    print(f"Latency ratio: {decision['latency_ratio']:.2f}x")
    for reason in decision["reasons"]:
        print(f"- {reason}")
    print(f"Report: {args.output.resolve()}")


if __name__ == "__main__":
    main()
