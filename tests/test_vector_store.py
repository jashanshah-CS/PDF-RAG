from datetime import UTC, datetime

import pytest

from src.rag_project.documents import DocumentChunk, SourceLocation, SourceType
from src.rag_project.embeddings import EmbeddedChunk
from src.rag_project.vector_store import (
    PersistentVectorStore,
    pdf_source_key,
    website_source_key,
)


def embedded(
    text: str,
    *,
    name: str = "handbook.pdf",
    page: int | None = 1,
    url: str | None = None,
    root_url: str | None = None,
    extraction_method: str | None = None,
) -> EmbeddedChunk:
    source_type = SourceType.WEBSITE if url else SourceType.PDF
    chunk = DocumentChunk(
        document_id=f"document-{name}-{page or url}",
        source_type=source_type,
        source_name=name,
        chunk_number=1,
        text=text,
        location=SourceLocation(page_number=page if not url else None, url=url),
        added_at=datetime(2026, 10, 6, 10, 30, tzinfo=UTC),
        metadata={
            key: value
            for key, value in {
                "root_url": root_url,
                "extraction_method": extraction_method,
            }.items()
            if value
        },
    )
    return EmbeddedChunk(chunk, (0.1, 0.2, 0.3))


def test_store_survives_reopening(tmp_path) -> None:
    path = tmp_path / "chroma"
    store = PersistentVectorStore(path)
    store.replace_source(pdf_source_key("handbook.pdf"), [embedded("Leave is 25 days.")])

    restored = PersistentVectorStore(path).load_all()

    assert len(restored) == 1
    assert restored[0].chunk.text == "Leave is 25 days."
    assert restored[0].chunk.location.page_number == 1
    assert restored[0].embedding == pytest.approx((0.1, 0.2, 0.3))


def test_reupload_replaces_pdf_with_same_filename(tmp_path) -> None:
    store = PersistentVectorStore(tmp_path / "chroma")
    key = pdf_source_key("Handbook.pdf")
    store.replace_source(key, [embedded("Old policy")])
    store.replace_source(key, [embedded("New policy")])

    assert [item.chunk.text for item in store.load_all()] == ["New policy"]


def test_store_preserves_ocr_metadata(tmp_path) -> None:
    store = PersistentVectorStore(tmp_path / "chroma")
    store.replace_source(
        pdf_source_key("scan.pdf"),
        [embedded("Recognized text", name="scan.pdf", extraction_method="ocr")],
    )

    restored = store.load_all()
    sources = store.list_sources()

    assert restored[0].chunk.metadata["extraction_method"] == "ocr"
    assert sources[0].ocr_chunk_count == 1


def test_lists_and_deletes_website_crawl_as_one_source(tmp_path) -> None:
    store = PersistentVectorStore(tmp_path / "chroma")
    root_url = "https://example.com/start"
    key = website_source_key(root_url)
    store.replace_source(
        key,
        [
            embedded(
                "Start page",
                name="Start",
                page=None,
                url=root_url,
                root_url=root_url,
            ),
            embedded(
                "About page",
                name="About",
                page=None,
                url="https://example.com/about",
                root_url=root_url,
            ),
        ],
    )

    sources = store.list_sources()
    assert len(sources) == 1
    assert sources[0].root_url == root_url
    assert sources[0].location_count == 2
    assert sources[0].chunk_count == 2

    store.delete_source(key)
    assert store.load_all() == []
