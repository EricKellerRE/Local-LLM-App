from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Awaitable, Callable
from urllib.parse import parse_qsl, urlencode, urljoin, urlparse, urlunparse

from local_model_app.mcp_manager import McpPluginManager
from local_model_app.task_models import WorkItemOutcome


ReferenceClassifier = Callable[[str, list[dict[str, str]]], Awaitable[set[str]]]
AsyncGenerator = Callable[[list[dict[str, Any]]], Awaitable[str]]

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
    temporary.replace(path)


class ResearchDiscoveryExecutor:
    """Build a persisted breadth-first citation graph without spending model turns on navigation."""

    def __init__(
        self,
        manager: McpPluginManager,
        data_directory: Path,
        classify_references: ReferenceClassifier | None = None,
    ) -> None:
        self.manager = manager
        self.data_directory = data_directory
        self.classify_references = classify_references

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
    def _new_ledger(goal: str, config: dict[str, int]) -> dict[str, Any]:
        return {
            "version": 1,
            "goal": goal,
            "config": config,
            "sources": [],
            "passes": [],
            "stop_reason": None,
        }

    async def _seed(self, ledger: dict[str, Any], target: int) -> None:
        goal = ledger["goal"]
        queries = [goal, f'"{goal}" review', f"{goal} primary study"]
        seen = {source["url"] for source in ledger["sources"]}
        pages = ledger.setdefault("seed_query_pages", {})
        for query in queries:
            active_seeds = [
                source for source in ledger["sources"]
                if source["depth"] == 0 and source.get("status") != "dropped"
            ]
            if len(active_seeds) >= target:
                break
            page = int(pages.get(query) or 0) + 1
            result = _structured(await self._call("search_web", {
                "query": query,
                "max_results": min(25, max(target, 10)),
                "page": page,
            }))
            pages[query] = page
            for row in result.get("results") or []:
                url = canonical_url(str(row.get("href") or row.get("url") or ""))
                if not url or url in seen:
                    continue
                seen.add(url)
                ledger["sources"].append({
                    "id": source_id(url),
                    "url": url,
                    "title": str(row.get("title") or url),
                    "snippet": str(row.get("body") or row.get("snippet") or ""),
                    "depth": 0,
                    "parents": [],
                    "status": "queued",
                })
                active_count = sum(
                    source["depth"] == 0 and source.get("status") != "dropped"
                    for source in ledger["sources"]
                )
                if active_count >= target:
                    break

    async def _fetch_depth(
        self,
        ledger: dict[str, Any],
        sources_dir: Path,
        ledger_path: Path,
        depth: int,
        reference_cap: int,
        max_characters: int,
    ) -> list[dict[str, str]]:
        candidates: dict[str, dict[str, str]] = {}
        for source in [
            item for item in ledger["sources"]
            if item["depth"] == depth and item["status"] != "dropped"
        ]:
            try:
                if source["status"] == "fetched":
                    content = (sources_dir / source["content_file"]).read_text(encoding="utf-8")
                else:
                    source["fetch_attempts"] = int(source.get("fetch_attempts") or 0) + 1
                    payload = _structured(await self._call("fetch_url", {
                        "url": source["url"],
                        "start_index": 0,
                        "max_length": max_characters,
                    }))
                    content = str(payload.get("content") or "")
                    if not content.strip():
                        raise ValueError("The source returned no readable text.")
                    sources_dir.mkdir(parents=True, exist_ok=True)
                    content_path = sources_dir / f"{source['id']}.md"
                    content_path.write_text(content, encoding="utf-8")
                    source["status"] = "fetched"
                    source["content_file"] = content_path.name
                    source["characters"] = len(content)
                references = extract_reference_candidates(content, source["url"])
                source["references_found"] = len(references)
                for candidate in references[:reference_cap]:
                    key = candidate["url"] or candidate["id"]
                    current = candidates.setdefault(key, candidate)
                    parents = current.setdefault("parents", [])
                    if source["id"] not in parents:
                        parents.append(source["id"])
            except Exception as exc:
                source["status"] = "failed"
                source["error"] = f"{type(exc).__name__}: {exc}"[:1000]
            _atomic_json(ledger_path, ledger)
        return list(candidates.values())

    async def _relevant(self, goal: str, candidates: list[dict[str, str]]) -> set[str]:
        if not candidates:
            return set()
        if self.classify_references is not None:
            try:
                selected: set[str] = set()
                for start in range(0, len(candidates), 40):
                    selected.update(await self.classify_references(goal, candidates[start:start + 40]))
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
        config = {
            "seed_sources": int(metadata.get("research_seed_sources") or 12),
            "depth_passes": int(metadata.get("research_depth_passes") or 3),
            "max_sources": int(metadata.get("research_max_sources") or 80),
            "references_per_source": int(metadata.get("research_references_per_source") or 12),
            "source_max_characters": int(metadata.get("research_source_max_characters") or 30000),
        }
        ledger_path, sources_dir = self._paths(str(task["id"]))
        try:
            ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, TypeError):
            ledger = self._new_ledger(str(task["definition"]["goal"]), config)

        await self.manager.ensure_started(["local.web-research"])
        active_seed_count = sum(
            source["depth"] == 0 and source.get("status") != "dropped"
            for source in ledger["sources"]
        )
        if active_seed_count < config["seed_sources"]:
            await self._seed(ledger, config["seed_sources"])
            _atomic_json(ledger_path, ledger)
        active_seed_count = sum(
            source["depth"] == 0 and source.get("status") != "dropped"
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
                ledger, sources_dir, ledger_path, depth, config["references_per_source"],
                config["source_max_characters"]
            )
            if depth == 0:
                readable_seeds = [
                    source for source in ledger["sources"]
                    if source["depth"] == 0 and source["status"] == "fetched"
                ]
                if len(readable_seeds) < config["seed_sources"]:
                    for source in ledger["sources"]:
                        if (source["depth"] == 0 and source.get("status") == "failed" and
                                int(source.get("fetch_attempts") or 0) >= 2):
                            source["status"] = "dropped"
                    await self._seed(ledger, config["seed_sources"])
                    _atomic_json(ledger_path, ledger)
                    return WorkItemOutcome(
                        outcome="retry",
                        summary=(f"Read {len(readable_seeds)} of {config['seed_sources']} required seed sources; "
                                 "failed candidates were checkpointed and replacements queued."),
                        wait_seconds=30,
                    )
            novel = [candidate for candidate in candidates if not candidate["url"] or candidate["url"] not in known]
            capacity = max(0, config["max_sources"] - len(ledger["sources"]))
            selected_ids = (
                await self._relevant(ledger["goal"], novel)
                if depth < config["depth_passes"] and capacity > 0 else set()
            )
            selected = [candidate for candidate in novel if candidate["id"] in selected_ids]
            resolved = await self._resolve(selected, capacity)
            accepted_by_url = {
                candidate["url"]: candidate for candidate in resolved
                if candidate["url"] and candidate["url"] not in known
            }
            accepted = list(accepted_by_url.values())[:capacity]
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
            if len(ledger["sources"]) >= config["max_sources"]:
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
        fetched = [source for source in ledger["sources"] if source["status"] == "fetched"]
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
