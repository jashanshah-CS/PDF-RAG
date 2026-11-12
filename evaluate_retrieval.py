"""Run the Version 3 retrieval-only benchmark against a local PDF corpus."""

import argparse
import json
from pathlib import Path

from src.rag_project.chunker import chunk_documents
from src.rag_project.embeddings import embed_chunks, load_embedding_model
from src.rag_project.pdf_loader import extract_pdf_files
from src.rag_project.retrieval_evaluation import (
    build_report,
    evaluate_retrieval_case,
    load_retrieval_cases,
    write_report,
)
from src.rag_project.search import semantic_search


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pdf-dir",
        action="append",
        type=Path,
        help="Directory containing benchmark PDFs; repeat for multiple directories",
    )
    parser.add_argument(
        "--questions",
        type=Path,
        default=Path("evaluation/retrieval_v3_cases.json"),
        help="Versioned JSON relevance judgements",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("evaluation/retrieval_v3_baseline_results.json"),
        help="Destination for the JSON report",
    )
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()
    if args.top_k <= 0:
        parser.error("--top-k must be greater than zero")
    return args


def main() -> None:
    args = parse_args()
    pdf_directories = args.pdf_dir or [
        Path("output/pdf/quality-test-10"),
        Path("output/pdf/version2-evaluation"),
    ]
    pdf_paths = sorted(
        {
            path
            for directory in pdf_directories
            for path in directory.glob("*.pdf")
        }
    )
    if not pdf_paths:
        searched = ", ".join(str(path.resolve()) for path in pdf_directories)
        raise SystemExit(f"No PDFs found in: {searched}")

    cases = load_retrieval_cases(args.questions)
    benchmark_payload = json.loads(args.questions.read_text(encoding="utf-8"))
    documents = extract_pdf_files((path.name, path.read_bytes()) for path in pdf_paths)
    chunks = chunk_documents(documents)
    # The baseline must be reproducible and must not change because a remote
    # model repository happened to be available during one run.
    model = load_embedding_model(local_only=True)
    embedded_chunks = embed_chunks(chunks, model)

    results = []
    for number, case in enumerate(cases, start=1):
        print(f"[{number}/{len(cases)}] {case.case_id}: {case.question}")
        result = evaluate_retrieval_case(
            case,
            lambda question: semantic_search(
                question, embedded_chunks, model, top_k=args.top_k
            ),
        )
        results.append(result)
        if result.supported:
            print(
                f"  evidence recall={result.evidence_recall:.0%}, "
                f"reciprocal rank={result.reciprocal_rank:.3f}, "
                f"latency={result.latency_ms:.1f} ms"
            )
        else:
            print(
                f"  unsupported probe, top score={result.top_score}, "
                f"latency={result.latency_ms:.1f} ms"
            )

    report = build_report(
        results,
        benchmark_name=benchmark_payload.get("name", args.questions.stem),
        top_k=args.top_k,
        corpus=[path.name for path in pdf_paths],
    )
    write_report(args.output, report)
    summary = report["summary"]
    print(f"\nRecall@{args.top_k}: {summary[f'recall_at_{args.top_k}']:.1%}")
    print(f"MRR: {summary['mean_reciprocal_rank']:.3f}")
    print(f"Mean latency: {summary['latency_ms']['mean']:.1f} ms")
    print(f"Report: {args.output.resolve()}")


if __name__ == "__main__":
    main()
