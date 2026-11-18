"""Run the deterministic Version 4 conversational benchmark."""

import argparse
from pathlib import Path
from tempfile import TemporaryDirectory

from src.rag_project.conversation_evaluation import (
    evaluate_conversation_benchmark,
    load_conversation_benchmark,
    write_conversation_report,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cases",
        type=Path,
        default=Path("evaluation/conversation_v4_cases.json"),
        help="Versioned conversational benchmark cases",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("evaluation/conversation_v4_results.json"),
        help="Benchmark report destination",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        name, cases = load_conversation_benchmark(args.cases)
        with TemporaryDirectory(
            prefix="rag-conversation-evaluation-",
            ignore_cleanup_errors=True,
        ) as directory:
            report = evaluate_conversation_benchmark(name, cases, Path(directory))
        write_conversation_report(args.output, report)
    except ValueError as exception:
        raise SystemExit(f"Conversation benchmark error: {exception}") from exception

    summary = report["summary"]
    print(
        f"Conversation benchmark: {summary['passed_count']}/"
        f"{summary['case_count']} cases passed"
    )
    print(f"Report: {args.output}")


if __name__ == "__main__":
    main()
