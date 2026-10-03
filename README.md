# Local PDF RAG

A beginner-friendly, local retrieval-augmented generation (RAG) project for
asking questions about a PDF and receiving answers with page-level citations.

The project is being built gradually so that every stage can be understood and
tested before the next feature is added.

## Current features

- Upload one text-based PDF through a Streamlit interface
- Extract text while retaining the filename and page number
- Split pages into chunks of up to 150 words with a 30-word overlap
- Create a normalized 384-dimensional embedding for every chunk
- Preview original pages, chunks, and a sample embedding
- Process documents locally on the computer

The embedding model is
[`sentence-transformers/all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2).
It is downloaded once and stored in the local `.model-cache` directory.

## How it currently works

```text
PDF upload
    ↓
Page-by-page text extraction
    ↓
150-word chunks with 30-word overlap
    ↓
384-dimensional local embeddings
```

Semantic search and answer generation are the next stages.

## Requirements

- Windows, macOS, or Linux
- [`uv`](https://docs.astral.sh/uv/) for Python and dependency management

The project uses Python 3.12. `uv` can download a project-specific Python
runtime automatically if Python is not already installed.

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

## Version 1 roadmap

- [x] Upload and extract one PDF
- [x] Split pages into overlapping chunks
- [x] Create embeddings locally
- [ ] Retrieve the chunks most relevant to a question
- [ ] Generate an answer using only retrieved evidence
- [ ] Display filename and page citations
- [ ] Evaluate with 15–20 prepared questions

## Limitations

- Scanned or image-only PDFs are not supported yet because they require OCR.
- The current embedding model is intended primarily for English text.
- The app does not answer questions yet; retrieval and generation are still on
  the roadmap.
- Uploaded documents are processed in memory and are not intentionally saved.

## Privacy and cost

The current pipeline runs locally and uses free, open-source software. Uploaded
PDFs and the downloaded model are excluded from Git. A hosted model API or paid
cloud deployment may be added later, but neither is required for Version 1.

