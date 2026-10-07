# Local PDF RAG

A beginner-friendly, local retrieval-augmented generation (RAG) project for
asking questions about a PDF and receiving answers with page-level citations.

The project is being built gradually so that every stage can be understood and
tested before the next feature is added.

## Current features

- Upload and search one or more text-based PDFs through a Streamlit interface
- Add up to five linked pages from one approved public website and search them
  alongside uploaded PDFs
- Extract text while retaining the filename and page number
- Represent extracted content with a source-neutral document model
- Assign stable, content-based document identifiers and ingestion timestamps
- Split pages into chunks of up to 150 words with a 30-word overlap
- Create a normalized 384-dimensional embedding for every chunk
- Persist chunks, embeddings, and citation metadata in local ChromaDB storage
- Automatically reload indexed sources after restarting the application
- Add, update, refresh, and remove saved PDF and website sources
- Save questions, answers, citations, and timestamps in local SQLite history
- Search, inspect, export, and clear question history through the interface
- Search for the five chunks most relevant to a natural-language question
  using combined semantic similarity and exact-term matching
- Show similarity scores and page-level source information
- Generate an evidence-grounded answer with local Qwen3 8B through Ollama
- Support alternative Ollama chat models with reasoning disabled for fast document Q&A
- Validate structured model output and render citations programmatically
- Refuse questions unsupported by the retrieved text
- Evaluate retrieval, answers, citations, refusals, and response time from CSV
- Preview original pages, chunks, and a sample embedding
- Process documents locally on the computer

The embedding model is
[`sentence-transformers/all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2).
It is downloaded once and stored in the local `.model-cache` directory.

## How it currently works

```text
PDF uploads and approved webpage
    ↓
Unified documents with source metadata
    ↓
Source-neutral chunks with inherited metadata
    ↓
384-dimensional local embeddings
    ↓
Persistent ChromaDB index in data/chroma
    ↓
Hybrid semantic and exact-term search
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

## Run locally

Clone the repository and enter its directory, then run:

```powershell
uv sync
uv run streamlit run app.py
```

Open `http://localhost:8501` if the browser does not open automatically.
The first PDF upload may take longer while the free embedding model downloads.

## Run the tests

```powershell
uv run pytest -q
```

## Evaluate Version 2

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

## Limitations

- Scanned or image-only PDFs are not supported yet because they require OCR.
- Website ingestion supports static HTML only and does not execute JavaScript.
- Website discovery follows only same-hostname HTML links and stops after five
  unique pages.
- The current embedding model is intended primarily for English text.
- Answer quality depends on whether semantic search retrieves the right passage.
- A small local model can still make mistakes, so the visible evidence and
  citations should always be checked.
- Extracted PDF text and embeddings persist locally, but the original PDF file
  is not retained. Refreshing a PDF therefore requires uploading it again.

## Privacy and cost

The current pipeline runs locally and uses free, open-source software. Uploaded
PDFs, the local ChromaDB index, and the downloaded model are excluded from Git.
A hosted model API or paid cloud deployment may be added later, but neither is
required for Version 2.
