from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable
from urllib.parse import parse_qsl, urlencode, urljoin, urlparse, urlunparse

from local_model_app.mcp_manager import McpPluginManager
from local_model_app.task_models import WorkItemOutcome


@dataclass(frozen=True)
class CandidateSelection:
    keep_numbers: set[int]
    needs_abstract_numbers: set[int]


ReferenceClassifier = Callable[[str, list[dict[str, Any]]], Awaitable[CandidateSelection]]
AsyncGenerator = Callable[[list[dict[str, Any]]], Awaitable[str]]
TokenCounter = Callable[[str], int]
MIN_SOURCE_CHARACTERS = 600
MIN_SOURCE_WORDS = 100

URL_PATTERN = re.compile(r"https?://[^\s<>\]\[\"']+", re.IGNORECASE)
MARKDOWN_LINK_PATTERN = re.compile(r"\[([^\]]{1,500})\]\((https?://[^\s)]+(?:\([^)]*\)[^\s)]*)?)\)", re.IGNORECASE)
DOI_PATTERN = re.compile(r"(?<![\w/])(10\.\d{4,9}/[-._;()/:A-Z0-9]+)", re.IGNORECASE)
REFERENCE_HEADING_PATTERN = re.compile(
    r"^#{1,6}\s+(?:references|bibliography|works cited|literature cited)\s*$",
    re.IGNORECASE | re.MULTILINE,
)
TRACKING_KEYS = {"fbclid", "gclid", "mc_cid", "mc_eid"}
STOP_WORDS = {
    "about", "after", "again", "against", "also", "among", "and", "are", "because", "been",
    "before", "being", "between", "both", "but", "can", "could", "does", "each", "for", "from",
    "have", "into", "its", "long", "more", "most", "not", "our", "should", "that", "the", "their",
    "then", "there", "these", "they", "this", "those", "through", "under", "using", "what", "when",
    "where", "which", "while", "with", "would", "your", "report", "research", "topic",
}


class SourceRejectedError(ValueError):
    """The fetched response is deterministically unusable as research evidence."""


DEFAULT_SOURCE_POLICY: dict[str, Any] = {
    "core_types": [],
    "supplemental_types": [],
    "disallowed_types": [],
    "unknown_role": "core",
    "max_supplemental_sources": 0,
    "follow_supplemental_references": False,
}


def source_type_hint(candidate: dict[str, Any], content: str = "") -> str:
    """Classify obvious document roles from host-visible metadata without judging relevance."""
    url = str(candidate.get("url") or "")
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()
    path = parsed.path.lower()
    title = str(candidate.get("title") or "").lower()
    head = " ".join(content[:5000].lower().split())
    combined = f"{title} {head}"

    if "research-starters" in path or "research starter" in combined:
        return "tertiary_overview"
    if host == "wikipedia.org" or host.endswith(".wikipedia.org"):
        return "reference_work"
    if host == "amazon.com" or host.endswith(".amazon.com"):
        return "retail"
    if any(value in path for value in ("/subjects/", "/topics/", "/topic/")):
        return "topic_index"
    if any(value in path for value in ("/blog/", "/news/", "/press-release/")):
        return "popular_summary"
    if "systematic review" in combined or "meta-analysis" in combined or "meta analysis" in combined:
        return "systematic_review"
    if "review" in title and any(candidate.get(field) for field in ("authors", "year", "venue", "abstract")):
        return "authoritative_review"
    if "/books/" in path or "bookshelf" in title:
        return "reference_work"
    if any(candidate.get(field) for field in ("authors", "year", "venue", "abstract")):
        return "scholarly_article"
    if host == "doi.org" or "doi.org/10." in url.lower():
        return "scholarly_article"
    scholarly_hosts = (
        "pubmed.ncbi.nlm.nih.gov", "pmc.ncbi.nlm.nih.gov", "nature.com", "science.org",
        "sciencedirect.com", "cell.com", "jbc.org", "springer.com", "link.springer.com",
        "karger.com", "frontiersin.org", "plos.org", "wiley.com", "tandfonline.com",
    )
    scholarly_paths = ("/article/", "/articles/", "/science/article/", "/fulltext/", "/chapter/")
    if any(host == value or host.endswith(f".{value}") for value in scholarly_hosts):
        if any(value in path for value in scholarly_paths) or host in {
            "pubmed.ncbi.nlm.nih.gov", "pmc.ncbi.nlm.nih.gov"
        }:
            return "scholarly_candidate"
    if re.search(r"(?m)^#{1,3}\s+(?:abstract|methods?|materials and methods|results|references)\b", content, re.I):
        return "scholarly_article"
    return "unknown_web"


def evidence_role(source_type: str, policy: dict[str, Any] | None) -> str:
    if not policy:
        return "core"
    if source_type in set(policy.get("core_types") or []):
        return "core"
    if source_type in set(policy.get("supplemental_types") or []):
        return "supplemental"
    if source_type in set(policy.get("disallowed_types") or []):
        return "disallowed"
    return str(policy.get("unknown_role") or "disallowed")


def canonical_url(url: str, base_url: str | None = None) -> str:
    value = urljoin(base_url or "", url.strip()).rstrip(".,;:'\"")
    doi = DOI_PATTERN.search(value)
    if doi and ("doi.org" in value.lower() or not urlparse(value).scheme):
        return f"https://doi.org/{doi.group(1).rstrip('.,;').lower()}"
    parsed = urlparse(value)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return ""
    hostname = (parsed.hostname or "").lower()
    port = parsed.port
    netloc = hostname if port in {None, 80, 443} else f"{hostname}:{port}"
    query = urlencode([
        (key, item)
        for key, item in parse_qsl(parsed.query, keep_blank_values=True)
        if not key.lower().startswith("utm_") and key.lower() not in TRACKING_KEYS
    ])
    path = parsed.path.rstrip("/") or "/"
    return urlunparse((parsed.scheme.lower(), netloc, path, "", query, ""))


def source_id(url: str) -> str:
    return hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]


def citation_id(citation: str) -> str:
    return "citation-" + hashlib.sha256(citation.encode("utf-8")).hexdigest()[:12]


def _plain_words(value: str) -> set[str]:
    return {
        word for word in re.findall(r"[a-z][a-z0-9-]{2,}", value.lower())
        if word not in STOP_WORDS
    }


def lexical_relevance(goal: str, candidate: dict[str, str]) -> bool:
    goal_words = _plain_words(goal)
    text_words = _plain_words(" ".join([
        candidate.get("title", ""), candidate.get("context", ""), candidate.get("url", "")
    ]))
    if goal_words & text_words:
        return True
    url = candidate.get("url", "").lower()
    return "doi.org/10." in url or any(host in url for host in ("pubmed.ncbi.nlm.nih.gov", "nature.com", "science.org"))


def extract_reference_candidates(markdown: str, parent_url: str) -> list[dict[str, str]]:
    """Extract traceable citation candidates, preferring a document's reference section."""
    heading = REFERENCE_HEADING_PATTERN.search(markdown)
    reference_text = markdown[heading.start():] if heading else markdown
    candidates: dict[str, dict[str, str]] = {}

    for match in MARKDOWN_LINK_PATTERN.finditer(reference_text):
        title = " ".join(match.group(1).split())
        url = canonical_url(match.group(2), parent_url)
        if not url or url == canonical_url(parent_url):
            continue
        context = " ".join(reference_text[max(0, match.start() - 220):match.end() + 220].split())
        candidates.setdefault(url, {"id": source_id(url), "url": url, "title": title, "context": context})

    for match in DOI_PATTERN.finditer(reference_text):
        url = canonical_url(f"https://doi.org/{match.group(1)}")
        if not url:
            continue
        context = " ".join(reference_text[max(0, match.start() - 260):match.end() + 180].split())
        candidates.setdefault(url, {"id": source_id(url), "url": url, "title": context[:300], "context": context})

    # Some extractors emit bare URLs instead of Markdown links.
    for match in URL_PATTERN.finditer(reference_text):
        url = canonical_url(match.group(0).rstrip(")"), parent_url)
        if not url or url == canonical_url(parent_url):
            continue
        context = " ".join(reference_text[max(0, match.start() - 220):match.end() + 220].split())
        candidates.setdefault(url, {"id": source_id(url), "url": url, "title": context[:220], "context": context})

    # Resolve bibliography entries that contain no link or DOI in a later, bounded search step.
    for raw_line in reference_text.splitlines():
        line = " ".join(raw_line.strip().lstrip("-* ").split())
        if len(line) < 45 or not re.search(r"\b(?:19|20)\d{2}\b", line):
            continue
        if URL_PATTERN.search(line) or DOI_PATTERN.search(line):
            continue
        identifier = citation_id(line)
        candidates.setdefault(identifier, {
            "id": identifier,
            "url": "",
            "title": line[:500],
            "context": line[:800],
        })
    return list(candidates.values())


def _structured(result: dict[str, Any]) -> dict[str, Any]:
    value = result.get("structuredContent") or result.get("structured_content") or {}
    return value if isinstance(value, dict) else {}


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    for attempt in range(5):
        try:
            temporary.replace(path)
            return
        except PermissionError:
            if attempt == 4:
                raise
            time.sleep(0.02 * (attempt + 1))


def split_source_sections(
    markdown: str,
    count_tokens: TokenCounter,
    max_tokens: int,
) -> list[dict[str, str]]:
    """Split a long source at headings, then paragraph boundaries, without losing text."""
    max_tokens = max(256, int(max_tokens))
    heading_parts = re.split(r"(?m)(?=^#{1,6}\s+\S)", markdown)
    logical_parts = [part.strip() for part in heading_parts if part.strip()]
    if not logical_parts:
        logical_parts = [markdown.strip()]
    chunks: list[str] = []
    current = ""
    for part in logical_parts:
        if count_tokens(part) > max_tokens:
            paragraphs = [value.strip() for value in re.split(r"\n\s*\n", part) if value.strip()]
        else:
            paragraphs = [part]
        bounded_paragraphs: list[str] = []
        for paragraph in paragraphs:
            if count_tokens(paragraph) <= max_tokens:
                bounded_paragraphs.append(paragraph)
                continue
            word_chunk: list[str] = []
            for word in paragraph.split():
                candidate = " ".join([*word_chunk, word])
                if word_chunk and count_tokens(candidate) > max_tokens:
                    bounded_paragraphs.append(" ".join(word_chunk))
                    word_chunk = [word]
                else:
                    word_chunk.append(word)
            if word_chunk:
                bounded_paragraphs.append(" ".join(word_chunk))
        for paragraph in bounded_paragraphs:
            candidate = f"{current}\n\n{paragraph}".strip()
            if current and count_tokens(candidate) > max_tokens:
                chunks.append(current)
                current = paragraph
            else:
                current = candidate
    if current:
        chunks.append(current)
    return [
        {"key": f"section-{index:04d}", "label": f"Document segment {index}", "content": chunk}
        for index, chunk in enumerate(chunks, 1)
    ]


class ResearchSourceProcessor:
    """Analyze one paper at a time and checkpoint every bounded model call."""

    def __init__(
        self,
        analyze_section: AsyncGenerator,
        build_dossier: AsyncGenerator,
        count_tokens: TokenCounter,
        data_directory: Path,
    ) -> None:
        self.analyze_section = analyze_section
        self.build_dossier = build_dossier
        self.count_tokens = count_tokens
        self.data_directory = data_directory

    def _path(self, task_id: str, source_key: str) -> Path:
        return self.data_directory / "research" / task_id / "source-analysis" / f"{source_key}.json"

    async def process(
        self,
        task: dict[str, Any],
        source: dict[str, Any],
        content: str,
    ) -> dict[str, Any]:
        metadata = task["definition"].get("metadata") or {}
        whole_limit = max(512, int(metadata.get("research_whole_source_max_tokens") or 8192))
        section_limit = max(512, min(whole_limit, int(metadata.get("research_section_input_tokens") or 6144)))
        path = self._path(str(task["id"]), source["id"])
        try:
            state = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, TypeError):
            state = {
                "version": 1,
                "source_id": source["id"],
                "title": source.get("title") or source["url"],
                "url": source["url"],
                "depth": source.get("depth", 0),
                "token_count": self.count_tokens(content),
                "mode": "whole_document" if self.count_tokens(content) <= whole_limit else "sections",
                "sections": [],
                "reduction_levels": [],
                "dossier": "",
                "status": "in_progress",
            }
        if state.get("status") == "completed" and str(state.get("dossier") or "").strip():
            return state

        if state["mode"] == "whole_document":
            sections = [{"key": "whole-document", "label": "Whole document", "content": content}]
        else:
            sections = split_source_sections(content, self.count_tokens, section_limit)
        saved_by_key = {row["key"]: row for row in state.get("sections") or []}
        for section in sections:
            if str(saved_by_key.get(section["key"], {}).get("notes") or "").strip():
                continue
            notes = await self.analyze_section([
                {"role": "system", "content": (
                    "Analyze one bounded portion of one scholarly source. Extract only source-grounded evidence: "
                    "specific findings, mechanisms, methods, study system or population, causal strength, boundary "
                    "conditions, limitations, disagreements, and important references mentioned in this portion. "
                    "Do not write the report or infer beyond the supplied text. Preserve the exact source URL."
                )},
                {"role": "user", "content": json.dumps({
                    "task_goal": task["definition"]["goal"],
                    "source": {
                        "id": source["id"], "title": source.get("title"), "url": source["url"],
                        "depth": source.get("depth", 0),
                    },
                    "segment": {"label": section["label"], "content": section["content"]},
                }, ensure_ascii=False)},
            ])
            saved_by_key[section["key"]] = {
                "key": section["key"], "label": section["label"], "notes": notes.strip(),
            }
            state["sections"] = [saved_by_key[value["key"]] for value in sections if value["key"] in saved_by_key]
            _atomic_json(path, state)

        note_blocks = [
            f"### {row['label']}\n{row['notes']}" for row in state["sections"] if str(row.get("notes") or "").strip()
        ]
        current_blocks = note_blocks
        reduction_levels = list(state.get("reduction_levels") or [])
        level = 0
        while self.count_tokens("\n\n".join(current_blocks)) > whole_limit and level < 8:
            reduction_chunks = split_source_sections(
                "\n\n".join(current_blocks), self.count_tokens, max(512, whole_limit // 2)
            )
            reductions = list(reduction_levels[level]) if level < len(reduction_levels) else []
            for index, chunk in enumerate(reduction_chunks):
                if index < len(reductions) and str(reductions[index]).strip():
                    continue
                reduced = await self.build_dossier([
                    {"role": "system", "content": (
                        "Compress these notes from one source without dropping methods, findings, mechanisms, "
                        "evidence strength, limitations, disagreements, or the exact source URL. Do not add claims."
                    )},
                    {"role": "user", "content": chunk["content"]},
                ])
                if index < len(reductions):
                    reductions[index] = reduced.strip()
                else:
                    reductions.append(reduced.strip())
                if level < len(reduction_levels):
                    reduction_levels[level] = reductions
                else:
                    reduction_levels.append(reductions)
                state["reduction_levels"] = reduction_levels
                _atomic_json(path, state)
            reduced_text = "\n\n".join(reductions)
            if self.count_tokens(reduced_text) >= self.count_tokens("\n\n".join(current_blocks)):
                current_blocks = reductions
                break
            current_blocks = reductions
            level += 1
        consolidation_input = "\n\n".join(current_blocks)

        dossier = await self.build_dossier([
            {"role": "system", "content": (
                "Create a reusable evidence dossier for exactly one scholarly source. Preserve the title and exact "
                "URL, then organize the source-grounded methods, study system or population, principal findings, "
                "mechanisms, causal strength, boundary conditions, limitations, disagreements, and useful cited leads. "
                "Distinguish what the source reports from interpretation. Do not write a multi-source report."
            )},
            {"role": "user", "content": json.dumps({
                "task_goal": task["definition"]["goal"],
                "source": {"id": source["id"], "title": source.get("title"), "url": source["url"]},
                "source_notes": consolidation_input,
            }, ensure_ascii=False)},
        ])
        state["dossier"] = dossier.strip()
        state["status"] = "completed"
        _atomic_json(path, state)
        return state


class ResearchDiscoveryExecutor:
    """Build a persisted breadth-first citation graph without spending model turns on navigation."""

    def __init__(
        self,
        manager: McpPluginManager,
        data_directory: Path,
        classify_references: ReferenceClassifier | None = None,
        source_processor: ResearchSourceProcessor | None = None,
    ) -> None:
        self.manager = manager
        self.data_directory = data_directory
        self.classify_references = classify_references
        self.source_processor = source_processor

    async def _call(self, native_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        tool = next(
            item for item in self.manager.tools_for_plugins(["local.web-research"])
            if item.native_name == native_name
        )
        return await self.manager.call_tool(tool.exposed_name, arguments)

    def _paths(self, task_id: str) -> tuple[Path, Path]:
        root = self.data_directory / "research" / task_id
        return root / "source-ledger.json", root / "sources"

    @staticmethod
    def _policy(metadata: dict[str, Any]) -> dict[str, Any]:
        configured = metadata.get("source_policy")
        return dict(configured) if isinstance(configured, dict) else dict(DEFAULT_SOURCE_POLICY)

    @staticmethod
    def _counts_toward_target(source: dict[str, Any]) -> bool:
        return source.get("evidence_role", "core") == "core" and source.get("status") != "dropped"

    @staticmethod
    def _annotate_candidate(candidate: dict[str, Any], policy: dict[str, Any], content: str = "") -> dict[str, Any]:
        source_type = source_type_hint(candidate, content)
        candidate["source_type"] = source_type
        candidate["evidence_role"] = evidence_role(source_type, policy)
        parsed = urlparse(str(candidate.get("url") or ""))
        candidate["source_locator"] = " / ".join(filter(None, [
            (parsed.hostname or "").lower(),
            next((part for part in parsed.path.split("/") if part), ""),
        ]))
        return candidate

    async def _enrich_candidate(self, candidate: dict[str, Any], policy: dict[str, Any]) -> dict[str, Any]:
        self._annotate_candidate(candidate, policy)
        if candidate["source_type"] != "unknown_web" or not candidate.get("url"):
            return candidate
        try:
            metadata = _structured(await self._call("fetch_scholarly_metadata", {"url": candidate["url"]}))
        except Exception:
            metadata = {}
        for field in ("title", "authors", "year", "venue", "abstract", "abstract_source"):
            if metadata.get(field):
                candidate[field] = metadata[field]
        self._annotate_candidate(candidate, policy)
        if candidate["source_type"] == "unknown_web" and candidate.get("evidence_role") == "needs_metadata":
            candidate["evidence_role"] = str(policy.get("unresolved_role") or "disallowed")
        return candidate

    @staticmethod
    def _source_rejection_reason(url: str, content: str) -> str | None:
        normalized = " ".join(content.lower().split())
        if not normalized:
            return "the source returned no readable text"
        boilerplate = (
            "access denied",
            "captcha",
            "javascript is disabled",
            "enable javascript to continue",
            "enable javascript to proceed",
            "please enable javascript",
            "required part of this site couldn't load",
            "required part of this site couldn’t load",
            "checking your browser",
            "verify you are human",
            "page not found",
        )
        if any(marker in normalized[:2000] for marker in boilerplate):
            return "the response is an access, JavaScript, or anti-bot interstitial"
        word_count = len(re.findall(r"\b[\w'-]+\b", content))
        if len(content) < MIN_SOURCE_CHARACTERS or word_count < MIN_SOURCE_WORDS:
            return (
                f"the response is too small to support source analysis "
                f"({len(content)} characters, {word_count} words)"
            )
        return None

    async def _fetch_complete(self, url: str, max_characters: int) -> tuple[str, bool]:
        chunks: list[str] = []
        start = 0
        complete = False
        chunk_size = min(30_000, max_characters)
        while start < max_characters:
            payload = _structured(await self._call("fetch_url", {
                "url": url,
                "start_index": start,
                "max_length": min(chunk_size, max_characters - start),
            }))
            content = str(payload.get("content") or "")
            if content:
                chunks.append(content)
            next_start = payload.get("next_start")
            complete = payload.get("complete") is not False or not isinstance(next_start, int)
            if complete or not content or next_start <= start:
                break
            start = next_start
        return "\n\n".join(chunks), complete

    @staticmethod
    def _new_ledger(goal: str, config: dict[str, int]) -> dict[str, Any]:
        return {
            "version": 1,
            "goal": goal,
            "config": config,
            "sources": [],
            "passes": [],
            "stop_reason": None,
        }

    async def _seed(self, ledger: dict[str, Any], target: int, policy: dict[str, Any]) -> None:
        goal = ledger["goal"]
        queries = [goal, f'"{goal}" review', f"{goal} primary study"]
        seen = {source["url"] for source in ledger["sources"]}
        considered = set(ledger.get("considered_seed_urls") or []) | seen
        pages = ledger.setdefault("seed_query_pages", {})
        page_history = ledger.setdefault("search_pages", [])
        for query in queries:
            while int(pages.get(query) or 0) < 20:
                active_count = sum(
                    source["depth"] == 0 and self._counts_toward_target(source)
                    for source in ledger["sources"]
                )
                if active_count >= target:
                    break
                page = int(pages.get(query) or 0) + 1
                result = _structured(await self._call("search_web", {
                    "query": query,
                    "max_results": min(25, max(target, 10)),
                    "page": page,
                }))
                pages[query] = page
                candidates: list[dict[str, str]] = []
                for row in result.get("results") or []:
                    url = canonical_url(str(row.get("href") or row.get("url") or ""))
                    if not url or url in considered:
                        continue
                    candidates.append({
                        "id": source_id(url),
                        "url": url,
                        "title": str(row.get("title") or url),
                        "context": str(row.get("body") or row.get("snippet") or ""),
                        "context_kind": "search_snippet",
                        "authors": row.get("authors") or [],
                        "year": row.get("year"),
                        "venue": row.get("venue"),
                        "abstract": row.get("abstract"),
                        "abstract_source": row.get("abstract_source"),
                    })
                for candidate in candidates:
                    self._annotate_candidate(candidate, policy)
                considered.update(candidate["url"] for candidate in candidates)
                ledger["considered_seed_urls"] = sorted(considered)
                eligible = [candidate for candidate in candidates if candidate["evidence_role"] != "disallowed"]
                selected_ids = await self._relevant(goal, eligible, policy)
                selected = [candidate for candidate in candidates if candidate["id"] in selected_ids]
                selected = [await self._enrich_candidate(candidate, policy) for candidate in selected]
                page_history.append({
                    "query": query,
                    "page": page,
                    "novel_candidates": len(candidates),
                    "relevant_selected": len(selected),
                    "presented_source_ids": [candidate["id"] for candidate in candidates],
                    "selected_source_ids": [candidate["id"] for candidate in selected],
                    "disallowed_source_ids": [
                        candidate["id"] for candidate in candidates if candidate["evidence_role"] == "disallowed"
                    ],
                })
                for candidate in selected:
                    role = candidate["evidence_role"]
                    if role == "disallowed":
                        continue
                    if role == "supplemental":
                        supplemental_count = sum(
                            source.get("evidence_role") == "supplemental" and source.get("status") != "dropped"
                            for source in ledger["sources"]
                        )
                        if supplemental_count >= int(policy.get("max_supplemental_sources") or 0):
                            continue
                    seen.add(candidate["url"])
                    ledger["sources"].append({
                        "id": candidate["id"],
                        "url": candidate["url"],
                        "title": candidate["title"],
                        "snippet": candidate["context"],
                        "depth": 0,
                        "parents": [],
                        "status": "queued",
                        "source_type": candidate["source_type"],
                        "evidence_role": role,
                        "authors": candidate.get("authors") or [],
                        "year": candidate.get("year"),
                        "venue": candidate.get("venue"),
                        "abstract": candidate.get("abstract"),
                        "abstract_source": candidate.get("abstract_source"),
                    })
                    if role == "core":
                        active_count += 1
                    if active_count >= target:
                        break
                if not selected:
                    break
            if sum(
                source["depth"] == 0 and self._counts_toward_target(source)
                for source in ledger["sources"]
            ) >= target:
                break

    async def _fetch_depth(
        self,
        task: dict[str, Any],
        ledger: dict[str, Any],
        sources_dir: Path,
        ledger_path: Path,
        depth: int,
        reference_cap: int,
        max_characters: int,
        policy: dict[str, Any],
    ) -> list[dict[str, str]]:
        candidates: dict[str, dict[str, str]] = {}
        for source in [
            item for item in ledger["sources"]
            if item["depth"] == depth and item["status"] != "dropped"
        ]:
            try:
                if source.get("content_file"):
                    content = (sources_dir / source["content_file"]).read_text(encoding="utf-8")
                else:
                    source["fetch_attempts"] = int(source.get("fetch_attempts") or 0) + 1
                    content, complete = await self._fetch_complete(source["url"], max_characters)
                    rejection_reason = self._source_rejection_reason(source["url"], content)
                    if rejection_reason:
                        raise SourceRejectedError(rejection_reason)
                    source_type = source_type_hint(source, content)
                    role = evidence_role(source_type, policy)
                    source["source_type"] = source_type
                    source["evidence_role"] = role
                    if role not in {"core", "supplemental"}:
                        raise SourceRejectedError(
                            f"the task source policy does not admit document type {source_type} as evidence"
                        )
                    sources_dir.mkdir(parents=True, exist_ok=True)
                    content_path = sources_dir / f"{source['id']}.md"
                    content_path.write_text(content, encoding="utf-8")
                    source["status"] = "fetched"
                    source["content_file"] = content_path.name
                    source["characters"] = len(content)
                    source["fetch_complete"] = complete
                role = source.get("evidence_role", "core")
                if role == "supplemental":
                    source["status"] = "supplemental"
                elif self.source_processor is not None and source.get("status") != "analyzed":
                    analysis = await self.source_processor.process(task, source, content)
                    source["analysis_file"] = str(
                        self.source_processor._path(str(task["id"]), source["id"]).relative_to(self.data_directory)
                    )
                    source["analysis_mode"] = analysis["mode"]
                    source["content_tokens"] = analysis["token_count"]
                    source["status"] = "analyzed"
                if role == "supplemental" and not bool(policy.get("follow_supplemental_references")):
                    source["references_found"] = 0
                    _atomic_json(ledger_path, ledger)
                    continue
                references = extract_reference_candidates(content, source["url"])
                source["references_found"] = len(references)
                for candidate in references[:reference_cap]:
                    self._annotate_candidate(candidate, policy)
                    key = candidate["url"] or candidate["id"]
                    current = candidates.setdefault(key, candidate)
                    parents = current.setdefault("parents", [])
                    if source["id"] not in parents:
                        parents.append(source["id"])
            except Exception as exc:
                if source.get("content_file"):
                    source["status"] = "fetched"
                elif isinstance(exc, SourceRejectedError):
                    source["status"] = "dropped"
                elif int(source.get("fetch_attempts") or 0) >= 2:
                    source["status"] = "dropped"
                else:
                    source["status"] = "failed"
                source["error"] = f"{type(exc).__name__}: {exc}"[:1000]
                _atomic_json(ledger_path, ledger)
                if source.get("content_file"):
                    raise
            _atomic_json(ledger_path, ledger)
        return list(candidates.values())

    async def _relevant(
        self,
        goal: str,
        candidates: list[dict[str, Any]],
        policy: dict[str, Any] | None = None,
    ) -> set[str]:
        if not candidates:
            return set()
        if self.classify_references is not None:
            try:
                selected: set[str] = set()
                for start in range(0, len(candidates), 40):
                    batch = candidates[start:start + 40]
                    choices: list[dict[str, Any]] = []
                    for number, candidate in enumerate(batch, 1):
                        title = URL_PATTERN.sub("", candidate.get("title", ""))
                        context = URL_PATTERN.sub("", candidate.get("context", ""))
                        abstract = str(candidate.get("abstract") or "").strip()
                        choice: dict[str, Any] = {
                            "number": number,
                            "title": " ".join(title.split())[:500],
                            "source_type_hint": candidate.get("source_type") or "unknown_web",
                            "evidence_role": candidate.get("evidence_role") or evidence_role(
                                str(candidate.get("source_type") or "unknown_web"), policy
                            ),
                            "source_locator": candidate.get("source_locator") or "",
                        }
                        for field in ("authors", "year", "venue"):
                            if candidate.get(field):
                                choice[field] = candidate[field]
                        if abstract:
                            choice["abstract"] = abstract[:4000]
                            choice["abstract_source"] = candidate.get("abstract_source") or "provided_metadata"
                        else:
                            choice["context"] = " ".join(context.split())[:800]
                            choice["context_kind"] = candidate.get("context_kind") or "citation_context"
                        choices.append(choice)
                    decision = await self.classify_references(goal, choices)
                    selected.update(
                        batch[number - 1]["id"]
                        for number in decision.keep_numbers
                        if 1 <= number <= len(batch)
                    )
                    ambiguous = [
                        (number, batch[number - 1])
                        for number in decision.needs_abstract_numbers
                        if 1 <= number <= len(batch) and number not in decision.keep_numbers
                    ]
                    if ambiguous:
                        enriched: list[dict[str, Any]] = []
                        source_by_number: dict[int, dict[str, str]] = {}
                        for enriched_number, (_, candidate) in enumerate(ambiguous, 1):
                            source_by_number[enriched_number] = candidate
                            metadata: dict[str, Any] = {}
                            if candidate.get("url"):
                                try:
                                    metadata = _structured(await self._call(
                                        "fetch_scholarly_metadata", {"url": candidate["url"]}
                                    ))
                                except Exception:
                                    metadata = {}
                            abstract = str(metadata.get("abstract") or "").strip()
                            description = str(metadata.get("description") or "").strip()
                            fallback_context = description or " ".join(
                                URL_PATTERN.sub("", candidate.get("context", "")).split()
                            )
                            enriched_candidate: dict[str, Any] = {
                                "number": enriched_number,
                                "title": str(metadata.get("title") or candidate.get("title") or "")[:500],
                                "authors": [str(value) for value in metadata.get("authors") or []][:12],
                                "year": metadata.get("year"),
                                "venue": metadata.get("venue"),
                                "abstract": abstract[:4000] if abstract else None,
                                "abstract_source": metadata.get("abstract_source") if abstract else None,
                                "context": fallback_context[:800] if not abstract else None,
                                "context_kind": (
                                    "page_description" if description else
                                    candidate.get("context_kind") or "citation_context"
                                ) if not abstract else None,
                                "source_type_hint": candidate.get("source_type") or "unknown_web",
                                "evidence_role": candidate.get("evidence_role") or "core",
                                "source_locator": candidate.get("source_locator") or "",
                            }
                            enriched.append(enriched_candidate)
                        second_decision = await self.classify_references(goal, enriched)
                        selected.update(
                            source_by_number[number]["id"]
                            for number in second_decision.keep_numbers
                            if number in source_by_number
                        )
                return selected
            except Exception:
                pass
        return {item["id"] for item in candidates if lexical_relevance(goal, item)}

    async def _resolve(self, candidates: list[dict[str, str]], limit: int) -> list[dict[str, str]]:
        resolved: list[dict[str, str]] = []
        for candidate in candidates:
            if len(resolved) >= limit:
                break
            if candidate.get("url"):
                resolved.append(candidate)
                continue
            result = _structured(await self._call("search_web", {
                "query": candidate.get("title") or candidate.get("context") or "",
                "max_results": 3,
                "page": 1,
            }))
            for row in result.get("results") or []:
                url = canonical_url(str(row.get("href") or row.get("url") or ""))
                if not url:
                    continue
                resolved.append({
                    **candidate,
                    "id": source_id(url),
                    "url": url,
                    "title": str(row.get("title") or candidate.get("title") or url),
                    "context": str(row.get("body") or row.get("snippet") or candidate.get("context") or ""),
                })
                break
        return resolved

    async def execute_work_item(
        self,
        task: dict[str, Any],
        item: dict[str, Any],
        completed_items: list[dict[str, Any]],
    ) -> WorkItemOutcome:
        del item, completed_items
        metadata = task["definition"].get("metadata") or {}
        policy = self._policy(metadata)
        config = {
            "seed_sources": int(metadata.get("research_seed_sources") or 12),
            "depth_passes": int(metadata.get("research_depth_passes") or 3),
            "max_sources": int(metadata.get("research_max_sources") or 80),
            "references_per_source": int(metadata.get("research_references_per_source") or 12),
            "source_max_characters": int(metadata.get("research_source_max_characters") or 160000),
        }
        ledger_path, sources_dir = self._paths(str(task["id"]))
        try:
            ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, TypeError):
            ledger = self._new_ledger(str(task["definition"]["goal"]), config)
        ledger["source_policy"] = policy

        await self.manager.ensure_started(["local.web-research"])
        active_seed_count = sum(
            source["depth"] == 0 and self._counts_toward_target(source)
            for source in ledger["sources"]
        )
        if active_seed_count < config["seed_sources"]:
            await self._seed(ledger, config["seed_sources"], policy)
            _atomic_json(ledger_path, ledger)
        active_seed_count = sum(
            source["depth"] == 0 and self._counts_toward_target(source)
            for source in ledger["sources"]
        )
        if active_seed_count < config["seed_sources"]:
            return WorkItemOutcome(
                outcome="retry",
                summary=(f"Seed discovery found only {active_seed_count} unique sources; "
                         f"{config['seed_sources']} are required before citation expansion."),
                wait_seconds=30,
            )

        known = {source["url"] for source in ledger["sources"]}
        existing_depths = {int(row["depth"]) for row in ledger.get("passes", [])}
        for depth in range(0, config["depth_passes"] + 1):
            if depth in existing_depths:
                continue
            candidates = await self._fetch_depth(
                task, ledger, sources_dir, ledger_path, depth, config["references_per_source"],
                config["source_max_characters"], policy
            )
            pending_fetches = [
                source for source in ledger["sources"]
                if source["depth"] == depth and source.get("status") == "failed"
            ]
            if pending_fetches:
                return WorkItemOutcome(
                    outcome="retry",
                    summary=(f"{len(pending_fetches)} selected source(s) could not be read on the first attempt; "
                             "their failures were checkpointed for one retry before replacement or exclusion."),
                    wait_seconds=30,
                )
            if depth == 0:
                readable_seeds = [
                    source for source in ledger["sources"]
                    if source["depth"] == 0 and source.get("evidence_role", "core") == "core"
                    and source["status"] in {"fetched", "analyzed"}
                ]
                if len(readable_seeds) < config["seed_sources"]:
                    for source in ledger["sources"]:
                        if (source["depth"] == 0 and source.get("status") == "failed" and
                                int(source.get("fetch_attempts") or 0) >= 2):
                            source["status"] = "dropped"
                    await self._seed(ledger, config["seed_sources"], policy)
                    _atomic_json(ledger_path, ledger)
                    return WorkItemOutcome(
                        outcome="retry",
                        summary=(f"Read {len(readable_seeds)} of {config['seed_sources']} required seed sources; "
                                 "failed candidates were checkpointed and replacements queued."),
                        wait_seconds=30,
                    )
            novel = [candidate for candidate in candidates if not candidate["url"] or candidate["url"] not in known]
            active_sources = sum(source.get("status") != "dropped" for source in ledger["sources"])
            capacity = max(0, config["max_sources"] - active_sources)
            selected_ids = (
                await self._relevant(ledger["goal"], novel, policy)
                if depth < config["depth_passes"] and capacity > 0 else set()
            )
            selected = [candidate for candidate in novel if candidate["id"] in selected_ids]
            resolved = await self._resolve(selected, capacity)
            resolved = [await self._enrich_candidate(candidate, policy) for candidate in resolved]
            accepted_by_url = {
                candidate["url"]: candidate for candidate in resolved
                if candidate["url"] and candidate["url"] not in known
                and candidate.get("evidence_role") != "disallowed"
            }
            accepted: list[dict[str, Any]] = []
            supplemental_count = sum(
                source.get("evidence_role") == "supplemental" and source.get("status") != "dropped"
                for source in ledger["sources"]
            )
            for candidate in accepted_by_url.values():
                if candidate.get("evidence_role") == "supplemental":
                    if supplemental_count >= int(policy.get("max_supplemental_sources") or 0):
                        continue
                    supplemental_count += 1
                accepted.append(candidate)
                if len(accepted) >= capacity:
                    break
            for candidate in accepted:
                known.add(candidate["url"])
                ledger["sources"].append({
                    "id": candidate["id"],
                    "url": candidate["url"],
                    "title": candidate.get("title") or candidate["url"],
                    "snippet": candidate.get("context") or "",
                    "depth": depth + 1,
                    "parents": candidate.get("parents") or [],
                    "status": "queued",
                    "source_type": candidate.get("source_type") or "unknown_web",
                    "evidence_role": candidate.get("evidence_role") or "core",
                    "authors": candidate.get("authors") or [],
                    "year": candidate.get("year"),
                    "venue": candidate.get("venue"),
                    "abstract": candidate.get("abstract"),
                    "abstract_source": candidate.get("abstract_source"),
                })
            ledger["passes"].append({
                "depth": depth,
                "sources_examined": sum(1 for source in ledger["sources"] if source["depth"] == depth),
                "references_found": len(candidates),
                "novel_references": len(novel),
                "relevant_references_added": len(accepted),
            })
            _atomic_json(ledger_path, ledger)
            if depth >= config["depth_passes"]:
                ledger["stop_reason"] = "depth_limit_reached"
                break
            if active_sources + len(accepted) >= config["max_sources"]:
                queued_deeper = any(
                    source["depth"] > depth and source["status"] in {"queued", "failed"}
                    for source in ledger["sources"]
                )
                if queued_deeper:
                    continue
                ledger["stop_reason"] = "source_limit_reached"
                break
            if not accepted:
                ledger["stop_reason"] = "no_novel_relevant_references"
                break

        _atomic_json(ledger_path, ledger)
        fetched = [
            source for source in ledger["sources"]
            if source.get("evidence_role", "core") == "core"
            and source["status"] in {"fetched", "analyzed"}
        ]
        if len([source for source in fetched if source["depth"] == 0]) < config["seed_sources"]:
            return WorkItemOutcome(
                outcome="retry",
                summary="Not all required seed sources could be read; the discovery checkpoint will retry.",
                wait_seconds=30,
            )
        source_map = "\n".join(
            f"- [depth {source['depth']}] {source['title']} — {source['url']}"
            for source in fetched
        )
        pass_summary = "; ".join(
            f"depth {row['depth']}: {row['sources_examined']} read, {row['novel_references']} novel, "
            f"{row['relevant_references_added']} added"
            for row in ledger["passes"]
        )
        return WorkItemOutcome(
            outcome="completed",
            summary=(f"Built a citation graph with {len(fetched)} readable sources from "
                     f"{config['seed_sources']} seeds; stopped because {ledger['stop_reason']}."),
            result=(f"Source ledger: {ledger_path}\nPasses: {pass_summary}\n\n{source_map}"),
            completion_evidence=[
                str(ledger_path),
                f"seed_sources={config['seed_sources']}",
                f"fetched_sources={len(fetched)}",
                f"supplemental_sources={sum(source.get('status') == 'supplemental' for source in ledger['sources'])}",
                f"depth_passes_completed={len(ledger['passes'])}",
                f"stop_reason={ledger['stop_reason']}",
            ],
        )


class ResearchNotesExecutor:
    """Read each source once in batches and persist reusable evidence notes for every report section."""

    def __init__(self, generate: AsyncGenerator, data_directory: Path) -> None:
        self.generate = generate
        self.data_directory = data_directory

    async def execute_work_item(
        self,
        task: dict[str, Any],
        item: dict[str, Any],
        completed_items: list[dict[str, Any]],
    ) -> WorkItemOutcome:
        del completed_items
        task_id = str(task["id"])
        root = self.data_directory / "research" / task_id
        ledger_path = root / "source-ledger.json"
        ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
        metadata = task["definition"].get("metadata") or {}
        batch_size = max(1, int(metadata.get("research_notes_batch_size") or 6))
        notes_path = root / "source-notes.json"
        try:
            saved = json.loads(notes_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, TypeError):
            saved = {"version": 1, "goal": task["definition"]["goal"], "batches": {}}

        sources = [source for source in ledger["sources"] if source.get("status") == "fetched"]
        for start in range(0, len(sources), batch_size):
            batch = sources[start:start + batch_size]
            batch_key = f"{start // batch_size:04d}"
            if batch_key in saved["batches"]:
                continue
            documents = []
            for source in batch:
                content = (root / "sources" / source["content_file"]).read_text(encoding="utf-8")
                documents.append({
                    "id": source["id"], "title": source["title"], "url": source["url"],
                    "depth": source["depth"], "content": content,
                })
            response = await self.generate([
                {"role": "system", "content": (
                    "Extract reusable research evidence from this bounded source batch. For every source ID, record "
                    "specific findings, mechanisms, methods, study system or population, evidential strength, "
                    "limitations, disagreements, and useful quotations only when essential. Preserve its exact URL. "
                    "Do not write the report and do not merge claims across sources. Return concise Markdown notes."
                )},
                {"role": "user", "content": json.dumps({
                    "task_goal": task["definition"]["goal"],
                    "checkpoint": item["instructions"],
                    "sources": documents,
                }, ensure_ascii=False)},
            ])
            inventory = "\n".join(
                f"- {source['id']} | {source['title']} | {source['url']} | depth {source['depth']}"
                for source in batch
            )
            saved["batches"][batch_key] = {
                "source_ids": [source["id"] for source in batch],
                "notes": f"### Source inventory\n{inventory}\n\n### Evidence notes\n{response}",
            }
            _atomic_json(notes_path, saved)

        combined = "\n\n".join(
            value["notes"] for _, value in sorted(saved["batches"].items())
        )
        urls = [source["url"] for source in sources]
        return WorkItemOutcome(
            outcome="completed",
            summary=(f"Extracted reusable evidence notes from {len(sources)} sources in "
                     f"{len(saved['batches'])} hardware-bounded batches."),
            result=f"Evidence notes: {notes_path}\n\n{combined}",
            completion_evidence=[str(notes_path), f"source_count={len(sources)}", *urls],
        )
