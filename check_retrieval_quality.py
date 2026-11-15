"""Check a retrieval report against the committed Version 3 quality policy."""

import argparse
from pathlib import Path

from src.rag_project.quality_gates import evaluate_quality_gates, load_json_object


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("evaluation/retrieval_v3_reranked_results.json"),
        help="Retrieval benchmark report to validate",
    )
    parser.add_argument(
        "--policy",
        type=Path,
        default=Path("evaluation/retrieval_v3_quality_gates.json"),
        help="Versioned quality-gate policy",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        report = load_json_object(args.report, label="retrieval report")
        policy = load_json_object(args.policy, label="quality-gate policy")
        failures = evaluate_quality_gates(report, policy)
    except ValueError as exception:
        raise SystemExit(f"Quality gate configuration error: {exception}") from exception

    if failures:
        details = "\n".join(f"  - {failure}" for failure in failures)
        raise SystemExit(f"Retrieval quality gates failed:\n{details}")

    print(
        f"Retrieval quality gates passed: {args.report} "
        f"against {args.policy}"
    )


if __name__ == "__main__":
    main()
