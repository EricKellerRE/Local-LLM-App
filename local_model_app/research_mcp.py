from __future__ import annotations

import ipaddress
import re
import socket
import urllib.request
from html import unescape
from typing import Any
from urllib.parse import urlparse

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
def fetch_url(url: str, start_index: int = 0, max_length: int = 30_000) -> dict[str, Any]:
    """Fetch a public webpage as Markdown. Use next_start to continue reading long pages in chunks."""
    from ddgs import DDGS

    safe_url = _require_public_url(url)
    start = max(0, int(start_index))
    length = max(1_000, min(int(max_length), 100_000))
    extracted = DDGS(timeout=30).extract(safe_url, fmt="text_markdown")
    content = str(extracted.get("content") or "")
    end = min(len(content), start + length)
    return {
        "url": str(extracted.get("url") or safe_url),
        "start_index": start,
        "content": content[start:end],
        "total_characters": len(content),
        "next_start": end if end < len(content) else None,
        "complete": end >= len(content),
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
