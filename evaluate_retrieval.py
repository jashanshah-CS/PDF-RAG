"""Run the Version 3 retrieval-only benchmark against a local PDF corpus."""

import argparse
import json
from pathlib import Path

from src.rag_project.bm25 import BM25Index
from src.rag_project.chunker import chunk_documents
from src.rag_project.cross_encoder_reranker import (
    CrossEncoderRetriever,
    load_cross_encoder,
)
from src.rag_project.embeddings import (
    EmbeddedChunk,
    embed_chunks,
    load_embedding_model,
)
from src.rag_project.hybrid import HybridRetriever
from src.rag_project.pdf_loader import extract_pdf_files
from src.rag_project.retrieval_evaluation import (
    build_report,
    evaluate_retrieval_case,
    load_retrieval_cases,
    write_report,
)
from src.rag_project.reranker import RerankedRetriever
from src.rag_project.search import semantic_search
from src.rag_project.sentence_window import SentenceWindowRetriever


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
        help="Destination for the JSON report; defaults according to retriever",
    )
    parser.add_argument(
        "--retriever",
        choices=(
            "semantic",
            "bm25",
            "hybrid",
            "hybrid-window",
            "reranked",
            "cross-encoder",
        ),
        default="semantic",
        help="Retrieval implementation to benchmark (default: semantic)",
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--allow-model-download",
        action="store_true",
        help="Allow the optional cross-encoder benchmark model to be downloaded",
    )
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
    retriever_config = None
    if args.retriever in {
        "semantic",
        "hybrid",
        "hybrid-window",
        "reranked",
        "cross-encoder",
    }:
        # The baseline must be reproducible and must not change because a remote
        # model repository happened to be available during one run.
        model = load_embedding_model(local_only=True)
        embedded_chunks = embed_chunks(chunks, model)
        if args.retriever == "cross-encoder":
            try:
                cross_encoder = load_cross_encoder(
                    local_only=not args.allow_model_download
                )
            except Exception as error:
                raise SystemExit(
                    "Cross-encoder model unavailable. Run once with "
                    "--allow-model-download, or use --retriever reranked. "
                    f"Details: {error}"
                ) from error
            learned_retriever = CrossEncoderRetriever(
                embedded_chunks,
                model,
                cross_encoder,
                fallback_on_error=False,
            )
            retriever_config = learned_retriever.configuration
            search = lambda question: learned_retriever.search(
                question, top_k=args.top_k
            )
        elif args.retriever == "reranked":
            reranked_retriever = RerankedRetriever(embedded_chunks, model)
            retriever_config = reranked_retriever.configuration
            search = lambda question: reranked_retriever.search(
                question, top_k=args.top_k
            )
        elif args.retriever == "hybrid-window":
            sentence_window_retriever = SentenceWindowRetriever(
                embedded_chunks, model
            )
            retriever_config = sentence_window_retriever.configuration
            search = lambda question: sentence_window_retriever.search(
                question, top_k=args.top_k
            )
        elif args.retriever == "hybrid":
            hybrid_retriever = HybridRetriever(embedded_chunks, model)
            retriever_config = hybrid_retriever.configuration
            search = lambda question: hybrid_retriever.search(
                question, top_k=args.top_k
            )
        else:
            search = lambda question: semantic_search(
                question, embedded_chunks, model, top_k=args.top_k
            )
    else:
        # BM25 needs text and metadata, not embedding vectors. The shared
        # wrapper keeps its output compatible with the existing result model.
        bm25_index = BM25Index([EmbeddedChunk(chunk, ()) for chunk in chunks])
        search = lambda question: bm25_index.search(question, top_k=args.top_k)

    results = []
    for number, case in enumerate(cases, start=1):
        print(f"[{number}/{len(cases)}] {case.case_id}: {case.question}")
        result = evaluate_retrieval_case(
            case,
            search,
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
        retriever=args.retriever,
        retriever_config=retriever_config,
    )
    output_path = args.output or Path(
        {
            "semantic": "evaluation/retrieval_v3_baseline_results.json",
            "bm25": "evaluation/retrieval_v3_bm25_results.json",
            "hybrid": "evaluation/retrieval_v3_hybrid_results.json",
            "hybrid-window": "evaluation/retrieval_v3_sentence_window_results.json",
            "reranked": "evaluation/retrieval_v3_reranked_results.json",
            "cross-encoder": "evaluation/retrieval_v4_cross_encoder_results.json",
        }[args.retriever]
    )
    write_report(output_path, report)
    summary = report["summary"]
    print(f"\nRecall@{args.top_k}: {summary[f'recall_at_{args.top_k}']:.1%}")
    print(f"MRR: {summary['mean_reciprocal_rank']:.3f}")
    print(f"Mean latency: {summary['latency_ms']['mean']:.1f} ms")
    print(f"Report: {output_path.resolve()}")


if __name__ == "__main__":
    main()
