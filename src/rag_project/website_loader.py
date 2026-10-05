"""Safely load a public webpage into the unified document model."""

from collections.abc import Callable
from datetime import UTC, datetime
from html.parser import HTMLParser
from http.client import HTTPMessage
from ipaddress import ip_address
import re
import socket
from typing import Protocol
from urllib.error import HTTPError, URLError
from urllib.parse import urldefrag, urljoin, urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

from src.rag_project.documents import (
    Document,
    SourceLocation,
    SourceType,
    create_document_id,
)


MAX_WEBPAGE_BYTES = 2 * 1024 * 1024
MAX_REDIRECTS = 3
USER_AGENT = "Local-PDF-RAG/2.0"


class WebsiteLoadError(ValueError):
    """Raised when a webpage cannot be loaded safely."""


class WebResponse(Protocol):
    """The response behaviour used by the website loader."""

    headers: HTTPMessage

    def read(self, amount: int = -1) -> bytes: ...
    def __enter__(self) -> "WebResponse": ...
    def __exit__(self, *args: object) -> None: ...


class _NoRedirectHandler(HTTPRedirectHandler):
    """Expose redirects so each destination can be safety-checked."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def normalize_public_url(url: str) -> str:
    """Validate and normalize a public HTTP(S) URL."""
    normalized, _ = urldefrag(url.strip())
    try:
        parsed = urlsplit(normalized)
        port = parsed.port
    except ValueError as error:
        raise WebsiteLoadError("Enter a valid website URL.") from error

    if parsed.scheme.lower() not in {"http", "https"}:
        raise WebsiteLoadError("Website URLs must use http or https.")
    if not parsed.hostname:
        raise WebsiteLoadError("Enter a website URL with a hostname.")
    if parsed.username or parsed.password:
        raise WebsiteLoadError("Website URLs cannot contain credentials.")

    hostname = parsed.hostname.rstrip(".").lower()
    if hostname == "localhost" or hostname.endswith(".localhost"):
        raise WebsiteLoadError("Local and private network URLs are not allowed.")
    if hostname.endswith(".local") or hostname.endswith(".internal"):
        raise WebsiteLoadError("Local and private network URLs are not allowed.")

    try:
        addresses = {
            address[4][0]
            for address in socket.getaddrinfo(
                hostname,
                port or (443 if parsed.scheme.lower() == "https" else 80),
                type=socket.SOCK_STREAM,
            )
        }
    except socket.gaierror as error:
        raise WebsiteLoadError("The website hostname could not be resolved.") from error

    if not addresses:
        raise WebsiteLoadError("The website hostname could not be resolved.")
    if any(not ip_address(address).is_global for address in addresses):
        raise WebsiteLoadError("Local and private network URLs are not allowed.")

    return normalized


def fetch_public_html(
    url: str,
    *,
    open_url: Callable[..., WebResponse] | None = None,
) -> tuple[str, bytes, str]:
    """Download bounded HTML while validating every redirect destination."""
    opener = open_url or build_opener(_NoRedirectHandler()).open
    current_url = normalize_public_url(url)

    for redirect_count in range(MAX_REDIRECTS + 1):
        request = Request(
            current_url,
            headers={"User-Agent": USER_AGENT, "Accept": "text/html"},
        )
        try:
            with opener(request, timeout=15) as response:
                content_type = response.headers.get_content_type()
                if content_type not in {"text/html", "application/xhtml+xml"}:
                    raise WebsiteLoadError("The URL did not return an HTML webpage.")

                content_length = response.headers.get("Content-Length")
                if content_length:
                    try:
                        declared_size = int(content_length)
                    except ValueError as error:
                        raise WebsiteLoadError(
                            "The website returned an invalid content length."
                        ) from error
                    if declared_size > MAX_WEBPAGE_BYTES:
                        raise WebsiteLoadError("The webpage is larger than the 2 MB limit.")

                content = response.read(MAX_WEBPAGE_BYTES + 1)
                if len(content) > MAX_WEBPAGE_BYTES:
                    raise WebsiteLoadError("The webpage is larger than the 2 MB limit.")

                charset = response.headers.get_content_charset() or "utf-8"
                return current_url, content, charset
        except HTTPError as error:
            if error.code not in {301, 302, 303, 307, 308}:
                raise WebsiteLoadError(
                    f"The website returned HTTP {error.code}."
                ) from error
            location = error.headers.get("Location")
            if not location:
                raise WebsiteLoadError("The website returned an invalid redirect.") from error
            if redirect_count == MAX_REDIRECTS:
                raise WebsiteLoadError("The website redirected too many times.") from error
            current_url = normalize_public_url(urljoin(current_url, location))
        except (URLError, TimeoutError, OSError) as error:
            raise WebsiteLoadError("The website could not be downloaded.") from error

    raise WebsiteLoadError("The website redirected too many times.")


class _ReadableHTMLParser(HTMLParser):
    """Collect page title and readable text grouped by headings."""

    IGNORED_TAGS = {"script", "style", "noscript", "svg", "nav", "footer", "form"}
    HEADING_TAGS = {"h1", "h2", "h3", "h4", "h5", "h6"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.title_parts: list[str] = []
        self.sections: list[tuple[str | None, str]] = []
        self._ignored_depth = 0
        self._in_title = False
        self._in_heading = False
        self._heading_parts: list[str] = []
        self._current_heading: str | None = None
        self._text_parts: list[str] = []

    def handle_starttag(self, tag: str, attrs) -> None:
        tag = tag.lower()
        if tag in self.IGNORED_TAGS:
            self._ignored_depth += 1
            return
        if self._ignored_depth:
            return
        if tag == "title":
            self._in_title = True
        elif tag in self.HEADING_TAGS:
            self._flush_section()
            self._in_heading = True
            self._heading_parts = []

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in self.IGNORED_TAGS:
            if self._ignored_depth:
                self._ignored_depth -= 1
            return
        if self._ignored_depth:
            return
        if tag == "title":
            self._in_title = False
        elif tag in self.HEADING_TAGS and self._in_heading:
            heading = _clean_text(" ".join(self._heading_parts))
            self._current_heading = heading or self._current_heading
            self._in_heading = False

    def handle_data(self, data: str) -> None:
        if self._ignored_depth:
            return
        text = _clean_text(data)
        if not text:
            return
        if self._in_title:
            self.title_parts.append(text)
        elif self._in_heading:
            self._heading_parts.append(text)
        else:
            self._text_parts.append(text)

    def close(self) -> None:
        super().close()
        self._flush_section()

    def _flush_section(self) -> None:
        text = _clean_text(" ".join(self._text_parts))
        if text:
            self.sections.append((self._current_heading, text))
        self._text_parts = []


def _clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def extract_website_documents(
    html_bytes: bytes,
    url: str,
    *,
    charset: str = "utf-8",
    added_at: datetime | None = None,
) -> list[Document]:
    """Convert HTML into source-aware document sections."""
    try:
        html = html_bytes.decode(charset, errors="replace")
    except LookupError as error:
        raise WebsiteLoadError("The webpage uses an unsupported text encoding.") from error

    parser = _ReadableHTMLParser()
    parser.feed(html)
    parser.close()

    title = _clean_text(" ".join(parser.title_parts)) or urlsplit(url).hostname or url
    document_id = create_document_id(SourceType.WEBSITE, url, html_bytes)
    ingestion_time = added_at or datetime.now(UTC)
    documents = [
        Document(
            document_id=document_id,
            source_type=SourceType.WEBSITE,
            source_name=title,
            text=f"{section}. {text}" if section else text,
            location=SourceLocation(url=url, section=section or "Page content"),
            added_at=ingestion_time,
            metadata={"media_type": "text/html"},
        )
        for section, text in parser.sections
    ]
    if not documents:
        raise WebsiteLoadError("No readable text was found on the webpage.")
    return documents


def load_website(
    url: str,
    *,
    fetcher: Callable[[str], tuple[str, bytes, str]] = fetch_public_html,
) -> list[Document]:
    """Download one approved URL and convert it into unified documents."""
    final_url, content, charset = fetcher(url)
    return extract_website_documents(content, final_url, charset=charset)
