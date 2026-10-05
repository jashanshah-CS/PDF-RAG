"""Persistent local ChromaDB storage for embedded source chunks."""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings

from src.rag_project.documents import DocumentChunk, SourceLocation, SourceType
from src.rag_project.embeddings import EmbeddedChunk


DEFAULT_STORE_PATH = Path(__file__).resolve().parents[2] / "data" / "chroma-v2"
COLLECTION_NAME = "rag_sources"


@dataclass(frozen=True)
class StoredSource:
    """Summary information for one user-managed source."""

    source_key: str
    source_type: SourceType
    source_name: str
    added_at: datetime
    chunk_count: int
    location_count: int
    root_url: str | None = None


def pdf_source_key(filename: str) -> str:
    """Return the stable management key used when replacing a PDF."""
    return f"pdf:{filename.strip().casefold()}"


def website_source_key(root_url: str) -> str:
    """Return the stable management key used when refreshing a crawl."""
    return f"website:{root_url.strip().rstrip('/').casefold()}"


def _chunk_id(source_key: str, chunk: DocumentChunk) -> str:
    location = chunk.location
    position = location.url or str(location.page_number or location.row_number or 0)
    return f"{source_key}|{chunk.document_id}|{position}|{chunk.chunk_number}"


def _metadata(source_key: str, chunk: DocumentChunk) -> dict[str, Any]:
    location = chunk.location
    metadata: dict[str, Any] = {
        "source_key": source_key,
        "document_id": chunk.document_id,
        "source_type": chunk.source_type.value,
        "source_name": chunk.source_name,
        "chunk_number": chunk.chunk_number,
        "added_at": chunk.added_at.isoformat(),
    }
    if location.page_number is not None:
        metadata["page_number"] = location.page_number
    if location.row_number is not None:
        metadata["row_number"] = location.row_number
    if location.url:
        metadata["url"] = location.url
    if location.section:
        metadata["section"] = location.section
    root_url = chunk.metadata.get("root_url")
    if isinstance(root_url, str) and root_url:
        metadata["root_url"] = root_url
    return metadata


class PersistentVectorStore:
    """Store and restore embeddings plus citation metadata with ChromaDB."""

    def __init__(self, path: Path = DEFAULT_STORE_PATH) -> None:
        path.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.client = chromadb.PersistentClient(
            path=str(path),
            settings=Settings(anonymized_telemetry=False),
        )
        self.collection = self.client.get_or_create_collection(
            COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

    def replace_source(
        self,
        source_key: str,
        chunks: list[EmbeddedChunk],
    ) -> None:
        """Atomically replace the indexed chunks belonging to one source key."""
        if not source_key.strip():
            raise ValueError("A source key is required.")
        if not chunks:
            raise ValueError("At least one embedded chunk is required.")

        self.collection.delete(where={"source_key": source_key})
        self.collection.add(
            ids=[_chunk_id(source_key, item.chunk) for item in chunks],
            documents=[item.chunk.text for item in chunks],
            embeddings=[list(item.embedding) for item in chunks],
            metadatas=[_metadata(source_key, item.chunk) for item in chunks],
        )

    def delete_source(self, source_key: str) -> None:
        """Remove every stored chunk for one managed source."""
        self.collection.delete(where={"source_key": source_key})

    def load_all(self) -> list[EmbeddedChunk]:
        """Recreate the in-memory search records from persistent storage."""
        stored = self.collection.get(
            include=["documents", "embeddings", "metadatas"]
        )
        documents = stored.get("documents") or []
        embeddings = stored.get("embeddings")
        metadatas = stored.get("metadatas") or []
        if embeddings is None:
            return []

        restored: list[EmbeddedChunk] = []
        for text, embedding, metadata in zip(
            documents, embeddings, metadatas, strict=True
        ):
            source_type = SourceType(str(metadata["source_type"]))
            extra_metadata: dict[str, Any] = {}
            root_url = metadata.get("root_url")
            if root_url:
                extra_metadata["root_url"] = str(root_url)
            chunk = DocumentChunk(
                document_id=str(metadata["document_id"]),
                source_type=source_type,
                source_name=str(metadata["source_name"]),
                chunk_number=int(metadata["chunk_number"]),
                text=str(text),
                location=SourceLocation(
                    page_number=(
                        int(metadata["page_number"])
                        if "page_number" in metadata
                        else None
                    ),
                    row_number=(
                        int(metadata["row_number"])
                        if "row_number" in metadata
                        else None
                    ),
                    url=str(metadata["url"]) if metadata.get("url") else None,
                    section=(
                        str(metadata["section"])
                        if metadata.get("section")
                        else None
                    ),
                ),
                added_at=datetime.fromisoformat(str(metadata["added_at"])),
                metadata=extra_metadata,
            )
            restored.append(
                EmbeddedChunk(chunk, tuple(float(value) for value in embedding))
            )
        return restored

    def list_sources(self) -> list[StoredSource]:
        """Return one summary row for each user-managed source."""
        stored = self.collection.get(include=["metadatas"])
        grouped: dict[str, list[dict[str, Any]]] = {}
        for metadata in stored.get("metadatas") or []:
            grouped.setdefault(str(metadata["source_key"]), []).append(metadata)

        sources: list[StoredSource] = []
        for source_key, entries in grouped.items():
            first = entries[0]
            locations = {
                str(entry.get("url") or entry.get("page_number") or "source")
                for entry in entries
            }
            sources.append(
                StoredSource(
                    source_key=source_key,
                    source_type=SourceType(str(first["source_type"])),
                    source_name=str(first["source_name"]),
                    added_at=max(
                        datetime.fromisoformat(str(entry["added_at"]))
                        for entry in entries
                    ),
                    chunk_count=len(entries),
                    location_count=len(locations),
                    root_url=(
                        str(first["root_url"])
                        if first.get("root_url")
                        else None
                    ),
                )
            )
        return sorted(sources, key=lambda source: source.source_name.casefold())
