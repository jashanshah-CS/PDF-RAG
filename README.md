# Local PDF RAG

A beginner-friendly, local retrieval-augmented generation (RAG) project for
asking questions about a PDF and receiving answers with page-level citations.

The project is being built gradually so that every stage can be understood and
tested before the next feature is added.

Version 3 is feature-complete on the `version-3` branch. It adds benchmarked
hybrid retrieval, focused context, deterministic reranking, and automated
quality gates. See [`RELEASE_NOTES_V3.md`](RELEASE_NOTES_V3.md) for the final
architecture, measurements, verification, and known limitations.

## Current features

- Upload and search one or more text-based PDFs through a Streamlit interface
- Add up to five linked pages from one approved public website and search them
  alongside uploaded PDFs
- Extract text while retaining the filename and page number
- Detect image-only PDF pages and recognize them locally with Tesseract OCR
- Label OCR-derived chunks so recognized text is distinguishable in the UI
- Represent extracted content with a source-neutral document model
- Assign stable, content-based document identifiers and ingestion timestamps
- Split pages into chunks of up to 150 words with a 30-word overlap
- Create a normalized 384-dimensional embedding for every chunk
- Persist chunks, embeddings, and citation metadata in local ChromaDB storage
- Automatically reload indexed sources after restarting the application
- Add, update, refresh, and remove saved PDF and website sources
- Save questions, answers, citations, and timestamps in local SQLite history
- Search, inspect, export, and clear question history through the interface
- Search semantic and BM25 candidate lists, then combine their ranks with
  deterministic weighted reciprocal rank fusion
- Focus retrieved evidence on two relevant sentence anchors and their immediate
  neighbours, with semantic fallback for paraphrased questions
- Rerank the top five focused candidates using original, semantic, and BM25
  ranks before sending evidence to the answer model
- Select exactly which saved PDFs and websites may be searched for each question
- Keep previous uploads in a compact document-library selector and automatically
  select newly added or updated sources
- Generate an evidence-grounded answer with local Qwen3 8B through Ollama
- Support alternative Ollama chat models with reasoning disabled for fast document Q&A
- Validate structured model output and render citations programmatically
- Show cited evidence in expandable source cards with page, section, excerpt,
  and original webpage links when available
- Refuse questions unsupported by the retrieved text
- Evaluate retrieval, answers, citations, refusals, and response time from CSV
- Switch between a high-contrast editorial Light theme and default Dark theme
- Process documents locally on the computer

The embedding model is
[`sentence-transformers/all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2).
It is downloaded once and stored in the local `.model-cache` directory.

## How it currently works

```text
PDF uploads and approved webpage
    ↓
Selectable-text extraction or local Tesseract OCR
    ↓
Unified documents with source metadata and extraction method
    ↓
Source-neutral chunks with inherited metadata
    ↓
384-dimensional local embeddings
    ↓
Persistent ChromaDB index in data/chroma
    ↓
Hybrid semantic and exact-term search
    ↓
Confidence-aware sentence windows
    ↓
Deterministic local reranking
    ↓
Top five evidence chunks
    ↓
Local Qwen3 8B answer with source labels
    ↓
Readable question history in data/history.db
```

Version 2 starts from the tested Version 1 pipeline. Its unified document and
chunk models support multiple PDFs and approved webpages. The current interface
can search multiple PDFs and up to five linked pages from one approved public
website together.

## Requirements

- Windows, macOS, or Linux
- [`uv`](https://docs.astral.sh/uv/) for Python and dependency management

The project uses Python 3.12. `uv` can download a project-specific Python
runtime automatically if Python is not already installed.

Install the default local chat model before starting the app:

```powershell
ollama pull qwen3:8b
```

Scanned PDF support also requires the local
[`Tesseract OCR`](https://github.com/tesseract-ocr/tesseract) engine. On Windows,
the project supports the standard UB Mannheim installation location, a
`tesseract` executable on `PATH`, or a custom executable configured with
`TESSERACT_CMD`.

## Run locally

Clone the repository and enter its directory, then run:

```powershell
uv sync
uv run streamlit run app.py
```

Open `http://localhost:8501` if the browser does not open automatically.
The first PDF upload may take longer while the free embedding model downloads.

## Run with Docker Compose

Docker is the recommended deployment for Version 2 because it packages the app,
Tesseract OCR, Ollama, Qwen3 8B, and persistent local storage together.

Requirements:

- Docker Desktop on Windows or macOS, or Docker Engine with Compose on Linux;
- at least 12 GB of available memory recommended for Qwen3 8B plus embeddings;
- several gigabytes of free disk space for container images and model files.

Start from a fresh clone:

```powershell
Copy-Item .env.example .env
docker compose up --build
```

Open `http://localhost:8501`. The first start takes longer because Compose pulls
the Ollama image, downloads `qwen3:8b`, builds the Python application, and caches
the MiniLM embedding model. Later starts reuse all three named volumes.

Stop the services without deleting documents, history, or models:

```powershell
docker compose down
```

Start them again with:

```powershell
docker compose up
```

The Compose services are:

- `app`: Streamlit, the RAG code, ChromaDB, SQLite, and Tesseract OCR;
- `ollama`: the local model server, reachable only inside the Compose network;
- `ollama-model`: a one-time setup job that ensures the configured model exists.

The named volumes are:

- `rag-data`: ChromaDB indexes and SQLite question history;
- `embedding-model`: the downloaded MiniLM embedding model;
- `ollama-models`: Qwen and other Ollama model files.

Do not run `docker compose down --volumes` unless you intentionally want to
delete all three persistent volumes.

### Configuration

Copy `.env.example` to `.env` and change only the values you need:

| Variable | Default | Purpose |
| --- | --- | --- |
| `OLLAMA_MODEL` | `qwen3:8b` | Local chat model used for answers and review |
| `OLLAMA_CHAT_URL` | `http://localhost:11434/api/chat` | Ollama API for non-Docker runs; Compose overrides it |
| `MAX_UPLOAD_MB` | `200` | Maximum uploaded file size |
| `RAG_DATA_DIR` | `data` | Parent directory for ChromaDB and SQLite |
| `MODEL_CACHE` | `.model-cache` | Sentence-transformer download cache |
| `EMBEDDING_LOCAL_ONLY` | `false` | Prevent downloads only when the model is already cached |
| `TESSERACT_CMD` | auto-detected | Optional custom Tesseract executable for local Windows runs |

The sidebar's **System status** panel checks writable storage, Ollama connectivity,
the configured Ollama model, Tesseract, and the Python environment. Storage
failure stops the app with an actionable message; Ollama and optional OCR issues
remain visible so document management can still be used.

## Run the tests

```powershell
uv run pytest -q
```

## Evaluate Version 2

The repository includes a reproducible ten-document quality benchmark. Its
PDFs are in `output/pdf/quality-test-10/`, its questions are in
`evaluation/quality_test_10.csv`, and its latest detailed results are in
`evaluation/quality_test_10_results.csv`. Regenerate the fictional documents
with `scripts/create_ten_quality_test_pdfs.py`. The latest local Qwen3 8B run
passed all 10 questions with an average response time of 8.13 seconds.

The harder benchmark in `evaluation/quality_test_10_hard.csv` covers unsupported
questions, near-duplicate numbers, cross-document comparisons, conflicting
policies, and misspellings. The latest deterministic Qwen3 8B run passes 8 of
10 cases. Its detailed output is saved in
`evaluation/quality_test_10_hard_results.csv`; the retained failures document
current limitations rather than being removed from the benchmark.

Copy `evaluation/questions.example.csv` to `evaluation/questions.csv`, then
replace the examples with questions about your PDFs and approved website.

Each CSV row contains:

- `question`: the question sent to the RAG pipeline;
- `expected_keywords`: required answer terms separated by `|`;
- `expected_sources`: required PDF filenames or webpage titles separated by `|`;
- `expected_pages`: required PDF page numbers separated by `|`;
- `expected_urls`: required webpage URLs separated by `|`;
- `should_refuse`: `true` when none of the supplied sources contains the answer.

Run multiple PDFs and a bounded website crawl together:

```powershell
uv run python evaluate.py `
  --pdf "C:\path\to\handbook.pdf" `
  --pdf "C:\path\to\benefits.pdf" `
  --website "https://example.com/start"
```

You may evaluate PDFs alone or a website alone. Repeat `--pdf` or `--website`
to add sources. Each website crawl is limited to five pages by default; use
`--website-pages 1` through `--website-pages 10` to change that bound.

The report is written to `evaluation/results.csv`. Local questions, results,
and PDFs are ignored by Git so private evaluation material is not published.

To use another installed Ollama model without changing the code:

```powershell
$env:OLLAMA_MODEL = "llama3.2:latest"
uv run python evaluate.py --pdf "C:\path\to\document.pdf" `
  --website "https://example.com/start" `
  --output "evaluation/results-llama32.csv"
```

The chat request sets Ollama's `think` option to `false`. This avoids long
reasoning traces for models such as Qwen3, where short evidence-based answers
are more useful than extended internal reasoning.

Multi-part questions receive one additional deterministic completeness review.
The reviewer checks every requested source, entity, condition, and sub-question
against the retrieved evidence. It rewrites an incomplete draft once, or returns
a safe refusal when the available evidence cannot support a complete answer.
Initial refusals also receive this single review so obvious spelling mistakes can
be reconsidered without repeatedly calling the model. Simple supported questions
remain single-pass.

## Evaluate Version 3 retrieval

Version 3 starts with a retrieval-only baseline so search changes can be
measured independently from Ollama answer generation. The benchmark uses JSON,
not CSV. Its 20 human-authored cases cover semantic paraphrases, exact terms,
context, multi-source questions, and unsupported probes across the synthetic
PDF corpus.

Run it after the embedding model has been downloaded once:

```powershell
uv run python evaluate_retrieval.py
```

The command searches the PDFs in `output/pdf/quality-test-10` and
`output/pdf/version2-evaluation`, then writes a detailed report to
`evaluation/retrieval_v3_baseline_results.json`. It reports Recall@5, mean
reciprocal rank, source accuracy, exact location accuracy, multi-source
coverage, and query latency. Unsupported probes are excluded from relevance
metrics; their highest similarity scores are retained for future threshold
calibration.

Run the standalone BM25 keyword retriever against the identical cases and
corpus:

```powershell
uv run python evaluate_retrieval.py --retriever bm25
```

BM25 uses an inverted index with inverse document frequency, term-frequency
saturation, and document-length normalization. The latest run preserved 100%
Recall@5, improved mean reciprocal rank from 0.958 to 1.000, and reduced mean
query time from 13.4 ms to approximately 0.1 ms. These timings exclude one-time
corpus loading and index construction. Its detailed report is stored in
`evaluation/retrieval_v3_bm25_results.json`.

Run the production hybrid retriever with:

```powershell
uv run python evaluate_retrieval.py --retriever hybrid
```

Hybrid retrieval takes the top 20 pure-vector and BM25 candidates, removes
duplicate chunks, and combines their positions with weighted reciprocal rank
fusion. The frozen configuration uses RRF `k=5`, vector weight `1`, and BM25
weight `3`; the values and candidate count are written into every report for
reproducibility. It achieved 100% Recall@5 and 1.000 MRR on the committed
benchmark. The detailed result is stored in
`evaluation/retrieval_v3_hybrid_results.json`.

Run the sentence-window postprocessor with:

```powershell
uv run python evaluate_retrieval.py --retriever hybrid-window
```

After hybrid retrieval, high-confidence exact matches select two sentence
anchors and one neighbouring sentence on each side. Ambiguous paraphrases use a
batched semantic fallback. The source, page, chunk identity, ranking score, and
original embedding remain unchanged. The committed benchmark retained 100%
Recall@5 and 1.000 MRR while reducing retrieved context by 21.9%. The broader
private PDF benchmark retained 100% Recall@5 and 0.871 MRR while reducing
context by 33.5%. The private run added about 0.89 seconds to retrieval; this is
accepted because it removes roughly one third of the text sent to local Qwen.
Detailed committed results are in
`evaluation/retrieval_v3_sentence_window_results.json`.

Run the production reranker with:

```powershell
uv run python evaluate_retrieval.py --retriever reranked
```

The offline reranker re-evaluates the five focused candidates using their
original hybrid rank, vector similarity rank, and focused-text BM25 rank. It
combines those lists with deterministic RRF using weights `3`, `1`, and `2`
respectively. The committed benchmark retained 100% Recall@5 and 1.000 MRR.
On the broader private 24-case PDF benchmark, it retained 100% Recall@5 and
improved MRR from 0.871 to 0.879. Mean retrieval time increased from about
0.96 to 1.28 seconds, a roughly 0.32-second cost for the improved ordering.
Detailed committed results are in
`evaluation/retrieval_v3_reranked_results.json`.

### Automated retrieval quality gates

Run the committed quality policy against the reranker report with:

```powershell
uv run python check_retrieval_quality.py
```

GitHub Actions also runs the complete Pytest suite, generates a fresh reranker
report, and fails the build if deterministic retrieval quality drops below the
committed thresholds. Recall@5, MRR, source accuracy, location accuracy,
multi-source coverage, benchmark case counts, and context reduction are gated.
Latency is measured but not gated because CI runner speed varies. See
[`docs/RETRIEVAL_QUALITY_GATES.md`](docs/RETRIEVAL_QUALITY_GATES.md) for every
threshold, the local workflow, failure behaviour, and CI design.

Use one or more custom directories when comparing another corpus:

```powershell
uv run python evaluate_retrieval.py `
  --pdf-dir "C:\path\to\first-corpus" `
  --pdf-dir "C:\path\to\second-corpus"
```

## Local database

Version 2 uses **ChromaDB** as an embedded vector database. It runs inside the
Python application and does not require a separate database server or account.
Its files are stored in `data/chroma-v2`, which is excluded from Git.

For every searchable chunk, ChromaDB stores:

- the extracted text;
- its 384-dimensional MiniLM embedding;
- the PDF filename and page number, or webpage title and URL;
- the source type, ingestion time, and chunk number;
- a management key used to update or remove the complete source.

The original PDF file is not copied into ChromaDB. Re-uploading a PDF with the
same filename replaces its existing chunks. Adding the same website start URL
again recrawls its pages and replaces the previous crawl. The **Saved sources**
section can remove either source type and all of its indexed chunks.

### Why ChromaDB and SQLite are separate

ChromaDB is used for source retrieval because it is a vector database: it can
store embedding arrays alongside text and metadata, then support similarity
search as the source collection grows. These records represent the current
searchable knowledge index rather than a chronological conversation.

SQLite is used for question history because questions and answers are ordinary,
ordered records. It provides reliable transactions, timestamps, text filtering,
and portable exports without encoding conversations as vectors. The app stores
SQLite history in `data/history.db`, including each successful question, answer,
its cited sources, and the creation time. Both databases are local and excluded
from Git.

Each history entry also records the sources that were selected when the question
was asked. This is distinct from citations: the selected-source list shows the
allowed search scope, while citations show which retrieved sources actually
supported the final answer.

## Scanned PDFs and OCR

OCR means **Optical Character Recognition**. A scanned PDF usually contains a
photograph of text rather than selectable characters, so ordinary PDF text
extraction returns nothing. For each page with fewer than 20 extracted
characters, the app renders that page at 200 DPI and asks the local Tesseract
engine to recognize its words. The resulting text retains the original PDF
filename and page number, then follows the same chunking, embedding, ChromaDB,
retrieval, and citation pipeline as normal text.

Tesseract runs locally and does not upload page images. OCR can still make
mistakes on handwriting, low-resolution scans, skewed pages, unusual fonts, or
complex tables, so OCR-labelled evidence should be checked against the source.

## Version 1 roadmap

- [x] Upload and extract one PDF
- [x] Split pages into overlapping chunks
- [x] Create embeddings locally
- [x] Retrieve the chunks most relevant to a question
- [x] Generate an answer using only retrieved evidence
- [x] Display filename and page citations
- [x] Add a repeatable evaluation runner
- [ ] Achieve acceptable results on 15–20 prepared questions

## Version 2 roadmap

- [x] Introduce unified document and chunk models
- [x] Add stable document IDs, source locations, and ingestion timestamps
- [x] Make embeddings, retrieval, generation, citations, and evaluation source-neutral
- [x] Upload and search multiple PDFs together
- [x] Ingest up to five linked pages from one approved public website
- [x] Evaluate PDF, website, cross-source, and unsupported questions
- [x] Persist embeddings and source metadata with local ChromaDB
- [x] Reload, update, refresh, list, and remove indexed sources
- [x] Persist searchable question, answer, citation, and timestamp history
- [x] Recognize image-only PDF pages locally with Tesseract OCR
- [x] Filter retrieval to user-selected PDF and website sources
- [x] Hide saved sources in a compact document-library selector
- [x] Automatically select newly uploaded or refreshed sources
- [x] Show expandable evidence cards for current and saved answers
- [x] Provide accessible Light and Dark interface themes

## Version 3 roadmap

- [x] Establish a versioned JSON retrieval benchmark and baseline report
- [x] Add a proper, independently benchmarked BM25 keyword retriever
- [x] Fuse semantic and keyword rankings with deterministic weighted RRF
- [x] Add confidence-aware sentence-window context after retrieval
- [x] Add a reranking stage and compare it against the frozen baseline
- [x] Add regression thresholds to automated tests and CI

## Version 4 roadmap

Version 4 focuses on conversational, trustworthy document Q&A. The complete
scope and acceptance criteria are in
[`VERSION_4_PLAN.md`](VERSION_4_PLAN.md).

- [ ] Add persistent conversation and message storage
- [ ] Build conversation management and chat timeline UI
- [ ] Rewrite follow-up questions into standalone retrieval queries
- [ ] Stream local Ollama answers with cancellation and failure handling
- [ ] Add claim-level citations and calibrated answer confidence
- [ ] Benchmark an optional local cross-encoder reranker
- [ ] Add a JSON conversational benchmark and CI quality gates

## Limitations

- OCR is limited to English in the current version.
- Handwriting, damaged scans, and complex layouts may reduce OCR accuracy.
- Website ingestion supports static HTML only and does not execute JavaScript.
- Website discovery follows only same-hostname HTML links and stops after five
  unique pages.
- The current embedding model is intended primarily for English text.
- Answer quality depends on whether hybrid retrieval finds the right passage.
- A small local model can still make mistakes, so the visible evidence and
  citations should always be checked.
- Severe misspellings can cause an answerable question to be refused, and a
  multi-document question may occasionally answer only one requested part.
- Extracted PDF text and embeddings persist locally, but the original PDF file
  is not retained. Refreshing a PDF therefore requires uploading it again.

## Privacy and cost

The current pipeline runs locally and uses free, open-source software. Uploaded
PDFs, the local ChromaDB index, and the downloaded model are excluded from Git.
A hosted model API or paid cloud deployment may be added later, but neither is
required for Version 3.
