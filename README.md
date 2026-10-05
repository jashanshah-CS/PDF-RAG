# Local PDF RAG

A beginner-friendly, local retrieval-augmented generation (RAG) project for
asking questions about a PDF and receiving answers with page-level citations.

The project is being built gradually so that every stage can be understood and
tested before the next feature is added.

## Current features

- Upload and search one or more text-based PDFs through a Streamlit interface
- Extract text while retaining the filename and page number
- Represent extracted content with a source-neutral document model
- Assign stable, content-based document identifiers and ingestion timestamps
- Split pages into chunks of up to 150 words with a 30-word overlap
- Create a normalized 384-dimensional embedding for every chunk
- Search for the three chunks most relevant to a natural-language question
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
PDF upload
    ↓
Unified documents with source metadata
    ↓
Source-neutral chunks with inherited metadata
    ↓
384-dimensional local embeddings
    ↓
Hybrid semantic and exact-term search
    ↓
Top three evidence chunks
    ↓
Local Qwen3 8B answer with source labels
```

Version 2 starts from the tested Version 1 pipeline. Its unified document and
chunk models are the foundation for multiple PDFs, approved webpages, and CSV
sources. The current interface can search multiple PDFs together.

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

## Evaluate Version 1

Copy `evaluation/questions.example.csv` to `evaluation/questions.csv`, then
replace the examples with 15–20 questions about your PDF.

Each CSV row contains:

- `question`: the question sent to the RAG pipeline;
- `expected_keywords`: required answer terms separated by `|`;
- `expected_page`: the page that should be retrieved and cited;
- `should_refuse`: `true` when the PDF does not contain the answer.

Run the full PDF pipeline and save a detailed report:

```powershell
uv run python evaluate.py --pdf "C:\path\to\document.pdf"
```

The report is written to `evaluation/results.csv`. Local questions, results,
and PDFs are ignored by Git so private evaluation material is not published.

To use another installed Ollama model without changing the code:

```powershell
$env:OLLAMA_MODEL = "llama3.2:latest"
uv run python evaluate.py --pdf "C:\path\to\document.pdf" --output "evaluation/results-llama32.csv"
```

The chat request sets Ollama's `think` option to `false`. This avoids long
reasoning traces for models such as Qwen3, where short evidence-based answers
are more useful than extended internal reasoning.

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
- [ ] Ingest an approved website or small list of pages
- [ ] Ingest CSV data and use structured querying for calculations
- [ ] Evaluate PDF, website, CSV, cross-source, and unsupported questions

## Limitations

- Scanned or image-only PDFs are not supported yet because they require OCR.
- The current embedding model is intended primarily for English text.
- Answer quality depends on whether semantic search retrieves the right passage.
- A small local model can still make mistakes, so the visible evidence and
  citations should always be checked.
- Uploaded documents are processed in memory and are not intentionally saved.

## Privacy and cost

The current pipeline runs locally and uses free, open-source software. Uploaded
PDFs and the downloaded model are excluded from Git. A hosted model API or paid
cloud deployment may be added later, but neither is required for Version 1.
