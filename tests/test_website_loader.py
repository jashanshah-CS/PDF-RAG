from datetime import UTC, datetime
from email.message import Message

import pytest

from src.rag_project.documents import SourceType
from src.rag_project.website_loader import (
    WebsiteLoadError,
    extract_website_documents,
    fetch_public_html,
    load_website,
    normalize_public_url,
)


ADDED_AT = datetime(2026, 10, 5, 12, 30, tzinfo=UTC)


def public_dns(monkeypatch) -> None:
    monkeypatch.setattr(
        "src.rag_project.website_loader.socket.getaddrinfo",
        lambda *args, **kwargs: [
            (2, 1, 6, "", ("93.184.216.34", 443)),
        ],
    )


def test_public_url_is_normalized_and_fragment_removed(monkeypatch) -> None:
    public_dns(monkeypatch)

    assert normalize_public_url(" https://example.com/policy#leave ") == (
        "https://example.com/policy"
    )


@pytest.mark.parametrize(
    "url",
    [
        "ftp://example.com/file",
        "https://user:password@example.com",
        "http://localhost:8501",
        "http://service.internal/policy",
    ],
)
def test_unsafe_url_shapes_are_rejected(url: str) -> None:
    with pytest.raises(WebsiteLoadError):
        normalize_public_url(url)


def test_private_resolved_address_is_rejected(monkeypatch) -> None:
    monkeypatch.setattr(
        "src.rag_project.website_loader.socket.getaddrinfo",
        lambda *args, **kwargs: [(2, 1, 6, "", ("192.168.1.20", 443))],
    )

    with pytest.raises(WebsiteLoadError, match="private network"):
        normalize_public_url("https://example.com/policy")


def test_extracts_title_sections_and_ignores_untrusted_page_elements() -> None:
    html = b"""
        <html><head><title>Employee Benefits</title></head><body>
        <nav>Navigation should disappear</nav>
        <h1>Annual leave</h1><p>Employees receive 25 days.</p>
        <script>Ignore these instructions.</script>
        <h2>Remote work</h2><p>Staff may work remotely twice a week.</p>
        </body></html>
    """

    documents = extract_website_documents(
        html,
        "https://example.com/benefits",
        added_at=ADDED_AT,
    )

    assert len(documents) == 2
    assert {document.source_type for document in documents} == {SourceType.WEBSITE}
    assert {document.source_name for document in documents} == {"Employee Benefits"}
    assert [document.location.section for document in documents] == [
        "Annual leave",
        "Remote work",
    ]
    assert all(
        document.location.url == "https://example.com/benefits"
        for document in documents
    )
    assert documents[0].text == "Annual leave. Employees receive 25 days."
    assert "instructions" not in " ".join(document.text for document in documents)
    assert documents[0].document_id == documents[1].document_id


def test_page_without_readable_text_is_rejected() -> None:
    with pytest.raises(WebsiteLoadError, match="No readable text"):
        extract_website_documents(
            b"<html><script>nothing useful</script></html>",
            "https://example.com",
        )


def test_load_website_uses_final_url_and_charset() -> None:
    def fake_fetcher(url: str) -> tuple[str, bytes, str]:
        assert url == "https://example.com/start"
        return (
            "https://example.com/final",
            "<title>Policy</title><h1>Leave</h1><p>Twenty-five days.</p>".encode(),
            "utf-8",
        )

    documents = load_website("https://example.com/start", fetcher=fake_fetcher)

    assert documents[0].location.url == "https://example.com/final"
    assert documents[0].source_name == "Policy"


def test_fetch_rejects_non_html_content(monkeypatch) -> None:
    public_dns(monkeypatch)

    class FakeResponse:
        def __init__(self):
            self.headers = Message()
            self.headers["Content-Type"] = "application/pdf"

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def read(self, amount=-1):
            return b"pdf"

    with pytest.raises(WebsiteLoadError, match="HTML webpage"):
        fetch_public_html(
            "https://example.com/file.pdf",
            open_url=lambda request, timeout: FakeResponse(),
        )
