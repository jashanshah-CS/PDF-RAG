"""Check a Version 4 conversation report against its committed quality policy."""

import argparse
from pathlib import Path

from src.rag_project.quality_gates import evaluate_quality_gates, load_json_object


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("evaluation/conversation_v4_results.json"),
        help="Conversation benchmark report to validate",
    )
    parser.add_argument(
        "--policy",
        type=Path,
        default=Path("evaluation/conversation_v4_quality_gates.json"),
        help="Versioned conversational quality-gate policy",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        report = load_json_object(args.report, label="conversation report")
        policy = load_json_object(args.policy, label="conversation quality policy")
        failures = evaluate_quality_gates(report, policy)
    except ValueError as exception:
        raise SystemExit(
            f"Conversation quality gate configuration error: {exception}"
        ) from exception
    if failures:
        details = "\n".join(f"  - {failure}" for failure in failures)
        raise SystemExit(f"Conversation quality gates failed:\n{details}")
    print(
        f"Conversation quality gates passed: {args.report} "
        f"against {args.policy}"
    )


if __name__ == "__main__":
    main()
