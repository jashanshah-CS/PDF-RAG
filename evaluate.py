"""Command-line evaluator for the local PDF and website RAG pipeline."""

import argparse
from pathlib import Path

from src.rag_project.chunker import chunk_documents
from src.rag_project.embeddings import embed_chunks, load_embedding_model
from src.rag_project.evaluation import (
    evaluate_case,
    load_evaluation_cases,
    write_results,
)
from src.rag_project.pdf_loader import extract_pdf_files
from src.rag_project.website_loader import crawl_website


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pdf",
        action="append",
        default=[],
        type=Path,
        help="PDF to evaluate; repeat for multiple PDFs",
    )
    parser.add_argument(
        "--website",
        action="append",
        default=[],
        help="Website start URL to crawl; repeat for multiple approved sites",
    )
    parser.add_argument(
        "--website-pages",
        type=int,
        default=5,
        choices=range(1, 11),
        metavar="1-10",
        help="Maximum unique pages to crawl per website (default: 5)",
    )
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
    args = parser.parse_args()
    if not args.pdf and not args.website:
        parser.error("provide at least one --pdf or --website source")
    return args


def main() -> None:
    args = parse_args()
    cases = load_evaluation_cases(args.questions)
    documents = []
    if args.pdf:
        documents.extend(
            extract_pdf_files((pdf.name, pdf.read_bytes()) for pdf in args.pdf)
        )
    for website_url in args.website:
        documents.extend(
            crawl_website(website_url, max_pages=args.website_pages)
        )

    chunks = chunk_documents(documents)
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
