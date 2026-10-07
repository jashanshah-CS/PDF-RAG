# Version 2.0.0

Version 2 turns the original single-PDF learning project into a persistent,
multi-source local RAG application.

## Highlights

- Upload and search multiple PDFs together.
- Crawl and index up to five same-site website pages.
- Select exactly which saved sources are searched for each question.
- Persist searchable embeddings and source metadata in ChromaDB.
- Persist searchable question-and-answer history in SQLite.
- Read image-only PDF pages using local Tesseract OCR.
- Generate grounded answers with local Qwen3 8B through Ollama.
- Review multi-part answers once for completeness before displaying them.
- Run reproducible basic and difficult answer-quality benchmarks.
- Configure model names, storage locations, upload limits, and service URLs
  through environment variables.
- Run the application, Ollama, Qwen, and Tesseract using Docker Compose.

## Verification

- All 86 automated tests pass.
- The basic ten-question quality benchmark passes 10/10.
- The difficult ten-question benchmark passes 8/10 and retains its two known
  failures to document current model limitations.
- The Compose YAML and frozen dependency lock were validated.

## Known limitations

- A severe spelling error can cause an answerable question to be refused.
- A small local model can occasionally alter an exact timing condition even
  after the completeness review.
- Website crawling supports static, same-host HTML and does not execute
  JavaScript.
- OCR currently uses English and may struggle with handwriting, damaged scans,
  or complex layouts.
- An end-to-end Docker runtime smoke test has not been performed on the release
  workstation because Docker is not installed there.
