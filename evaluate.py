"""Command-line evaluator for the complete local PDF RAG pipeline."""

import argparse
from pathlib import Path

from src.rag_project.chunker import chunk_pdf_pages
from src.rag_project.embeddings import embed_chunks, load_embedding_model
from src.rag_project.evaluation import (
    evaluate_case,
    load_evaluation_cases,
    write_results,
)
from src.rag_project.pdf_loader import extract_pdf_pages


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf", required=True, type=Path, help="PDF to evaluate")
    parser.add_argument(
        "--questions",
        type=Path,
        default=Path("evaluation/questions.csv"),
        help="CSV containing the expected evaluation cases",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("evaluation/results.csv"),
        help="Destination for detailed CSV results",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cases = load_evaluation_cases(args.questions)
    pages = extract_pdf_pages(args.pdf.read_bytes(), args.pdf.name)
    chunks = chunk_pdf_pages(pages)
    model = load_embedding_model()
    embedded_chunks = embed_chunks(chunks, model)

    results = []
    for number, case in enumerate(cases, start=1):
        print(f"[{number}/{len(cases)}] {case.question}")
        result = evaluate_case(case, embedded_chunks, model)
        results.append(result)
        print(f"  {'PASS' if result.passed else 'FAIL'}: {result.actual_answer or result.error}")

    write_results(args.output, results)
    passed = sum(result.passed for result in results)
    average_seconds = sum(result.elapsed_seconds for result in results) / len(results)

    print(f"\nPassed: {passed}/{len(results)} ({passed / len(results):.0%})")
    print(f"Average response time: {average_seconds:.2f} seconds")
    print(f"Detailed results: {args.output.resolve()}")


if __name__ == "__main__":
    main()

