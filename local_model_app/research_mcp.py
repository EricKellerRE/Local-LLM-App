from __future__ import annotations

import hashlib
import ipaddress
import io
import os
import re
import socket
import urllib.request
from html import unescape
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

MAX_PDF_BYTES = 50_000_000
MAX_ORIGINAL_BYTES = 50_000_000
ARCHIVE_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,128}$")

try:
    from mcp.server.mcpserver import MCPServer as _Server
except ImportError:  # MCP Python SDK 1.x compatibility
    from mcp.server.fastmcp import FastMCP as _Server


server = _Server(
    "Local Model Web Research",
    instructions=(
        "Search the public web, then fetch promising sources in chunks. Preserve exact source URLs "
        "in research notes and distinguish evidence from inference."
    ),
)


def _require_public_url(url: str) -> str:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("Only public http and https URLs are supported.")
    hostname = parsed.hostname.rstrip(".").lower()
    if hostname == "localhost" or hostname.endswith(".local"):
        raise ValueError("Local and private network addresses are not available to web research.")
    try:
        addresses = {item[4][0] for item in socket.getaddrinfo(hostname, parsed.port or 443)}
    except socket.gaierror as exc:
        raise ValueError(f"Could not resolve {hostname}.") from exc
    if not addresses or any(not ipaddress.ip_address(address).is_global for address in addresses):
        raise ValueError("Local and private network addresses are not available to web research.")
    return url


META_PATTERN = re.compile(
    r"<meta\s+[^>]*(?:name|property)=[\"']([^\"']+)[\"'][^>]*content=[\"']([^\"']*)[\"'][^>]*>|"
    r"<meta\s+[^>]*content=[\"']([^\"']*)[\"'][^>]*(?:name|property)=[\"']([^\"']+)[\"'][^>]*>",
    re.IGNORECASE,
)


def _scholarly_metadata(html: str, url: str) -> dict[str, Any]:
    values: dict[str, list[str]] = {}
    for match in META_PATTERN.finditer(html):
        name = (match.group(1) or match.group(4) or "").strip().lower()
        content = unescape(match.group(2) or match.group(3) or "").strip()
        if name and content:
            values.setdefault(name, []).append(content)

    def first(*names: str) -> str | None:
        return next((values[name][0] for name in names if values.get(name)), None)

    date = first("citation_publication_date", "citation_date", "article:published_time", "dc.date")
    year_match = re.search(r"\b(?:19|20)\d{2}\b", date or "")
    # Only scholarly metadata fields are called abstracts. Generic HTML/OpenGraph
    # descriptions remain clearly labelled fallback context for the selector.
    abstract = first("citation_abstract", "dc.description", "dcterms.abstract")
    abstract_source = next((name for name in (
        "citation_abstract", "dc.description", "dcterms.abstract"
    ) if values.get(name)), None)
    description = first("description", "og:description", "twitter:description")
    description_source = next((name for name in (
        "description", "og:description", "twitter:description"
    ) if values.get(name)), None)
    return {
        "url": url,
        "title": first("citation_title", "dc.title", "og:title", "twitter:title"),
        "authors": values.get("citation_author") or values.get("dc.creator") or [],
        "year": int(year_match.group(0)) if year_match else None,
        "venue": first("citation_journal_title", "citation_conference_title", "dc.source"),
        "abstract": abstract,
        "abstract_source": abstract_source,
        "description": description,
        "description_source": description_source,
    }


def _extract_pdf_bytes(raw: bytes) -> str:
    """Deterministically extract page text from a downloaded PDF."""
    from pypdf import PdfReader

    if len(raw) > MAX_PDF_BYTES:
        raise ValueError(f"PDF exceeds the {MAX_PDF_BYTES // 1_000_000} MB extraction limit.")
    if not raw.lstrip().startswith(b"%PDF-"):
        raise ValueError("The PDF URL did not return a PDF document.")
    reader = PdfReader(io.BytesIO(raw))
    pages: list[str] = []
    for number, page in enumerate(reader.pages, 1):
        text = str(page.extract_text() or "").strip()
        if text:
            pages.append(f"## Page {number}\n\n{text}")
    extracted = "\n\n".join(pages).strip()
    if not extracted:
        raise ValueError("The PDF contains no extractable text; it may require OCR.")
    return extracted


def _download_document(url: str, max_bytes: int = MAX_ORIGINAL_BYTES) -> tuple[bytes, str, str]:
    """Download a public document once without placing its bytes in an MCP response."""
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "LocalModelResearch/1.0 (+local desktop research client)"},
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        raw = response.read(max_bytes + 1)
        final_url = str(response.geturl() or url)
        content_type = str(response.headers.get("Content-Type") or "application/octet-stream")
    if len(raw) > max_bytes:
        raise ValueError(f"Source document exceeds the {max_bytes // 1_000_000} MB archive limit.")
    _require_public_url(final_url)
    return raw, final_url, content_type


def _extract_pdf_text(url: str) -> str:
    """Download a public PDF and deterministically extract page text."""
    raw, _, _ = _download_document(url, MAX_PDF_BYTES)
    return _extract_pdf_bytes(raw)


def _original_extension(raw: bytes, content_type: str, final_url: str) -> str:
    if raw.lstrip().startswith(b"%PDF-") or "application/pdf" in content_type.lower():
        return ".pdf"
    if "html" in content_type.lower() or raw.lstrip().lower().startswith((b"<!doctype html", b"<html")):
        return ".html"
    suffix = Path(urlparse(final_url).path).suffix.lower()
    if re.fullmatch(r"\.[a-z0-9]{1,8}", suffix):
        return suffix
    return ".bin"


def _store_original_document(
    raw: bytes,
    final_url: str,
    content_type: str,
    task_id: str,
    source_id: str,
) -> dict[str, Any]:
    """Persist original source bytes under the configured app-data directory."""
    if not ARCHIVE_ID_PATTERN.fullmatch(task_id) or not ARCHIVE_ID_PATTERN.fullmatch(source_id):
        raise ValueError("Archive task and source identifiers contain unsupported characters.")
    configured_root = os.getenv("LOCAL_APP_DATA_DIRECTORY", "").strip()
    if not configured_root:
        raise ValueError("The app data directory is not configured for source archiving.")
    data_root = Path(configured_root).expanduser().resolve()
    archive_directory = (data_root / "research" / task_id / "originals").resolve()
    if data_root not in archive_directory.parents:
        raise ValueError("The source archive path escaped the app data directory.")
    archive_directory.mkdir(parents=True, exist_ok=True)
    extension = _original_extension(raw, content_type, final_url)
    destination = archive_directory / f"{source_id}{extension}"
    temporary = archive_directory / f".{source_id}{extension}.tmp"
    temporary.write_bytes(raw)
    os.replace(temporary, destination)
    return {
        "original_file": destination.relative_to(data_root).as_posix(),
        "original_url": final_url,
        "original_content_type": content_type,
        "original_bytes": len(raw),
        "original_sha256": hashlib.sha256(raw).hexdigest(),
    }


@server.tool(structured_output=True)
def search_web(
    query: str,
    max_results: int = 10,
    page: int = 1,
    timelimit: str | None = None,
) -> dict[str, Any]:
    """Search multiple public web indexes and return titles, snippets, and exact source URLs."""
    from ddgs import DDGS

    count = max(1, min(int(max_results), 25))
    page_number = max(1, min(int(page), 20))
    results = DDGS(timeout=20).text(
        query,
        max_results=count,
        page=page_number,
        timelimit=timelimit,
        backend="auto",
    )
    return {"query": query, "page": page_number, "results": results}


@server.tool(structured_output=True)
def fetch_url(
    url: str,
    start_index: int = 0,
    max_length: int = 30_000,
    archive_task_id: str | None = None,
    archive_source_id: str | None = None,
) -> dict[str, Any]:
    """Fetch a public source as text, optionally retaining its original bytes outside model context."""
    from ddgs import DDGS

    safe_url = _require_public_url(url)
    start = max(0, int(start_index))
    length = max(1_000, min(int(max_length), 100_000))
    if bool(archive_task_id) != bool(archive_source_id):
        raise ValueError("Both archive_task_id and archive_source_id are required for source archiving.")
    archive: dict[str, Any] = {}
    archived_raw: bytes | None = None
    archived_url = safe_url
    archived_content_type = ""
    if start == 0 and archive_task_id and archive_source_id:
        try:
            archived_raw, archived_url, archived_content_type = _download_document(safe_url)
            archive = _store_original_document(
                archived_raw,
                archived_url,
                archived_content_type,
                archive_task_id,
                archive_source_id,
            )
        except Exception as exc:
            archive = {"original_archive_error": str(exc)[:500]}
    is_pdf_url = urlparse(safe_url).path.lower().endswith(".pdf")
    archived_pdf = bool(archived_raw and archived_raw.lstrip().startswith(b"%PDF-"))
    if archived_pdf or is_pdf_url:
        content = _extract_pdf_bytes(archived_raw) if archived_pdf else _extract_pdf_text(safe_url)
        extracted_url = archived_url if archived_pdf else safe_url
        content_kind = "pdf_text"
    else:
        extracted = DDGS(timeout=30).extract(safe_url, fmt="text_markdown")
        content = str(extracted.get("content") or "")
        extracted_url = str(extracted.get("url") or safe_url)
        content_kind = "markdown"
        if content.lstrip().startswith("%PDF-"):
            content = _extract_pdf_text(extracted_url)
            content_kind = "pdf_text"
    end = min(len(content), start + length)
    return {
        "url": extracted_url,
        "start_index": start,
        "content": content[start:end],
        "content_kind": content_kind,
        "total_characters": len(content),
        "next_start": end if end < len(content) else None,
        "complete": end >= len(content),
        **archive,
    }


@server.tool(structured_output=True)
def fetch_scholarly_metadata(url: str) -> dict[str, Any]:
    """Fetch lightweight scholarly metadata fields without downloading and extracting a full paper."""
    safe_url = _require_public_url(url)
    request = urllib.request.Request(
        safe_url,
        headers={"User-Agent": "LocalModelResearch/1.0 (+local desktop research client)"},
    )
    with urllib.request.urlopen(request, timeout=20) as response:
        content_type = str(response.headers.get("Content-Type") or "")
        if "html" not in content_type.lower():
            return _scholarly_metadata("", str(response.geturl() or safe_url))
        raw = response.read(2_000_000)
        charset = response.headers.get_content_charset() or "utf-8"
        html = raw.decode(charset, errors="replace")
        return _scholarly_metadata(html, str(response.geturl() or safe_url))


def main() -> None:
    server.run(transport="stdio")


if __name__ == "__main__":
    main()
