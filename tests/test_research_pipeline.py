import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from local_model_app.research_pipeline import (
    CandidateSelection,
    ResearchDiscoveryExecutor,
    ResearchNotesExecutor,
    ResearchSourceProcessor,
    canonical_url,
    evidence_role,
    extract_reference_candidates,
    source_type_hint,
    split_source_sections,
)


class FakeResearchManager:
    def __init__(self, seeds, pages, metadata=None):
        self.seeds = seeds
        self.pages = pages
        self.metadata = metadata or {}
        self.calls = []
        self.tools = [
            SimpleNamespace(native_name="search_web", exposed_name="web__search_web"),
            SimpleNamespace(native_name="fetch_url", exposed_name="web__fetch_url"),
            SimpleNamespace(native_name="fetch_scholarly_metadata", exposed_name="web__fetch_scholarly_metadata"),
        ]

    async def ensure_started(self, plugin_ids):
        self.calls.append(("start", tuple(plugin_ids)))

    def tools_for_plugins(self, plugin_ids):
        return self.tools

    async def call_tool(self, exposed_name, arguments):
        self.calls.append((exposed_name, arguments))
        if exposed_name == "web__search_web":
            return {"structuredContent": {"results": self.seeds}}
        if exposed_name == "web__fetch_scholarly_metadata":
            return {"structuredContent": self.metadata.get(arguments["url"], {})}
        content = self.pages[arguments["url"]]
        if len(content.split()) < 120:
            content += "\n\n" + " ".join(["substantive research evidence"] * 120)
        return {"structuredContent": {
            "url": arguments["url"],
            "content": content,
            "complete": True,
        }}


class ResearchPipelineTests(unittest.TestCase):
    def test_source_type_hints_separate_document_role_from_relevance(self):
        policy = {
            "core_types": ["scholarly_article", "scholarly_candidate"],
            "supplemental_types": ["tertiary_overview", "reference_work"],
            "disallowed_types": ["topic_index", "retail"],
            "unknown_role": "needs_metadata",
        }
        self.assertEqual(source_type_hint({
            "url": "https://www.ebsco.com/research-starters/health/physiology",
            "title": "Physiology of memory - EBSCO",
        }), "tertiary_overview")
        self.assertEqual(source_type_hint({
            "url": "https://www.nature.com/subjects/long-term-memory/neuro",
            "title": "Long-term memory",
        }), "topic_index")
        self.assertEqual(source_type_hint({
            "url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC123",
            "title": "Memory mechanisms",
        }), "scholarly_candidate")
        self.assertEqual(evidence_role("tertiary_overview", policy), "supplemental")
        self.assertEqual(evidence_role("topic_index", policy), "disallowed")

    def test_supplemental_seed_is_processed_for_leads_but_does_not_fill_core_quota(self):
        tertiary = "https://www.ebsco.com/research-starters/health/physiology"
        article = "https://journal.test/article/memory"
        manager = FakeResearchManager(
            [
                {"title": "Physiology of memory - EBSCO", "href": tertiary, "body": "Memory overview."},
                {"title": "Memory mechanism study", "href": article, "body": "Molecular memory evidence."},
            ],
            {
                tertiary: "# RESEARCH STARTER\n\n" + " ".join(["general memory overview"] * 120),
                article: "# Abstract\n\n" + " ".join(["experimental memory evidence"] * 120),
            },
            {article: {
                "title": "Memory mechanism study", "authors": ["A. Scientist"],
                "year": 2025, "venue": "Journal of Memory",
            }},
        )
        seen_roles = []

        async def keep_all(goal, candidates):
            seen_roles.extend(candidate["evidence_role"] for candidate in candidates)
            return CandidateSelection(
                keep_numbers={candidate["number"] for candidate in candidates},
                needs_abstract_numbers=set(),
            )

        policy = {
            "core_types": ["scholarly_article", "scholarly_candidate"],
            "supplemental_types": ["tertiary_overview", "reference_work"],
            "disallowed_types": ["topic_index", "retail"],
            "unknown_role": "needs_metadata",
            "unresolved_role": "disallowed",
            "max_supplemental_sources": 3,
            "follow_supplemental_references": True,
        }
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temporary:
            root = Path(temporary)
            executor = ResearchDiscoveryExecutor(manager, root, keep_all)
            task = {"id": "task-policy", "definition": {"goal": "memory", "metadata": {
                "research_seed_sources": 1, "research_depth_passes": 0, "source_policy": policy,
            }}}
            outcome = asyncio.run(executor.execute_work_item(task, {}, []))
            ledger = json.loads((root / "research" / "task-policy" / "source-ledger.json").read_text())

        self.assertEqual(outcome.outcome, "completed", f"{outcome.summary}: {ledger}")
        self.assertIn("supplemental", seen_roles)
        self.assertIn("needs_metadata", seen_roles)
        by_url = {source["url"]: source for source in ledger["sources"]}
        self.assertEqual(by_url[tertiary]["status"], "supplemental")
        self.assertEqual(by_url[tertiary]["evidence_role"], "supplemental")
        self.assertEqual(by_url[article]["status"], "fetched")
        self.assertEqual(by_url[article]["evidence_role"], "core")
        self.assertIn("seed_sources=1", outcome.completion_evidence)

    def test_source_quality_gate_rejects_interstitials_and_undersized_text(self):
        interstitial = """JavaScript is disabled in your browser.
Please enable JavaScript to proceed. A required part of this site couldn’t load."""
        self.assertIn(
            "interstitial",
            ResearchDiscoveryExecutor._source_rejection_reason(
                "https://www.nature.com/articles/example", interstitial
            ),
        )
        self.assertIn(
            "too small",
            ResearchDiscoveryExecutor._source_rejection_reason(
                "https://journal.test/paper", "A short publisher message without usable evidence."
            ),
        )
        substantive = " ".join(["memory mechanism evidence"] * 120)
        self.assertIsNone(ResearchDiscoveryExecutor._source_rejection_reason(
            "https://journal.test/paper", substantive
        ))

    def test_long_source_sections_are_bounded_and_preserve_text(self):
        text = "# Intro\n\n" + " ".join(f"word{i}" for i in range(1300))
        sections = split_source_sections(text, lambda value: len(value.split()), 512)
        self.assertGreaterEqual(len(sections), 3)
        self.assertTrue(all(len(section["content"].split()) <= 512 for section in sections))
        self.assertIn("word1299", sections[-1]["content"])

    def test_source_processor_uses_whole_or_section_mode_and_resumes(self):
        section_calls = []
        dossier_calls = []

        async def analyze(messages):
            payload = json.loads(messages[-1]["content"])
            section_calls.append(payload["segment"]["label"])
            return f"notes for {payload['segment']['label']} https://papers.test/a"

        async def dossier(messages):
            dossier_calls.append(messages[-1]["content"])
            return "source dossier https://papers.test/a"

        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temporary:
            root = Path(temporary)
            processor = ResearchSourceProcessor(analyze, dossier, lambda value: len(value.split()), root)
            source = {"id": "a", "title": "A", "url": "https://papers.test/a", "depth": 0}
            task = {"id": "task-source", "definition": {"goal": "memory", "metadata": {
                "research_whole_source_max_tokens": 512,
                "research_section_input_tokens": 512,
            }}}
            content = "# Intro\n\n" + " ".join(f"word{i}" for i in range(1200))
            first = asyncio.run(processor.process(task, source, content))
            call_count = len(section_calls) + len(dossier_calls)
            second = asyncio.run(processor.process(task, source, content))

        self.assertEqual(first["mode"], "sections")
        self.assertEqual(first["status"], "completed")
        self.assertGreaterEqual(len(first["sections"]), 3)
        self.assertEqual(second["dossier"], first["dossier"])
        self.assertEqual(len(section_calls) + len(dossier_calls), call_count)

    def test_discovery_fetches_a_long_source_in_exact_url_chunks(self):
        url = "https://papers.test/long"
        document = ("memory mechanism evidence supports consolidation and maintenance. " * 1400)[:65_000]

        class ChunkManager(FakeResearchManager):
            async def call_tool(self, exposed_name, arguments):
                self.calls.append((exposed_name, arguments))
                if exposed_name == "web__search_web":
                    return {"structuredContent": {"results": self.seeds}}
                start = int(arguments.get("start_index") or 0)
                length = int(arguments.get("max_length") or 30_000)
                end = min(len(document), start + length)
                return {"structuredContent": {
                    "url": url,
                    "content": document[start:end],
                    "complete": end >= len(document),
                    "next_start": end if end < len(document) else None,
                }}

        async def keep_all(goal, candidates):
            return CandidateSelection(
                keep_numbers={candidate["number"] for candidate in candidates},
                needs_abstract_numbers=set(),
            )

        manager = ChunkManager([{"title": "Long paper", "href": url, "body": "memory"}], {url: document})
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temporary:
            root = Path(temporary)
            executor = ResearchDiscoveryExecutor(manager, root, keep_all)
            task = {"id": "task-chunks", "definition": {"goal": "memory", "metadata": {
                "research_seed_sources": 1,
                "research_depth_passes": 0,
                "research_source_max_characters": 80_000,
            }}}
            outcome = asyncio.run(executor.execute_work_item(task, {}, []))
            saved = (root / "research" / "task-chunks" / "sources").glob("*.md")
            content = next(saved).read_text(encoding="utf-8")

        fetch_calls = [args for name, args in manager.calls if name == "web__fetch_url"]
        self.assertEqual(outcome.outcome, "completed")
        self.assertEqual([call["start_index"] for call in fetch_calls], [0, 30000, 60000])
        self.assertGreaterEqual(len(content), len(document))

    def test_unusable_seed_is_dropped_and_refilled(self):
        bad = "https://www.nature.com/articles/blocked"
        good = "https://papers.test/replacement"

        class RefillManager(FakeResearchManager):
            async def call_tool(self, exposed_name, arguments):
                self.calls.append((exposed_name, arguments))
                if exposed_name == "web__search_web":
                    page = int(arguments.get("page") or 1)
                    row = ({"title": "Memory", "href": bad, "body": "memory"} if page == 1 else
                           {"title": "Replacement paper", "href": good, "body": "memory"})
                    return {"structuredContent": {"results": [row]}}
                content = (
                    "JavaScript is disabled in your browser. Please enable JavaScript to proceed. "
                    "A required part of this site couldn’t load." if arguments["url"] == bad else
                    " ".join(["scholarly memory mechanism evidence"] * 120)
                )
                return {"structuredContent": {"url": arguments["url"], "content": content, "complete": True}}

        async def keep_all(goal, candidates):
            return CandidateSelection(
                keep_numbers={candidate["number"] for candidate in candidates},
                needs_abstract_numbers=set(),
            )

        manager = RefillManager([], {})
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temporary:
            root = Path(temporary)
            executor = ResearchDiscoveryExecutor(manager, root, keep_all)
            task = {"id": "task-refill", "definition": {"goal": "memory", "metadata": {
                "research_seed_sources": 1, "research_depth_passes": 0,
            }}}
            first = asyncio.run(executor.execute_work_item(task, {}, []))
            second = asyncio.run(executor.execute_work_item(task, {}, []))
            ledger = json.loads((root / "research" / "task-refill" / "source-ledger.json").read_text())

        self.assertEqual(first.outcome, "retry")
        self.assertEqual(second.outcome, "completed")
        self.assertEqual(next(source for source in ledger["sources"] if source["url"] == bad)["status"], "dropped")
        self.assertEqual(next(source for source in ledger["sources"] if source["url"] == good)["status"], "fetched")

    def test_canonical_url_removes_tracking_and_fragment(self):
        self.assertEqual(
            canonical_url("https://EXAMPLE.com/paper/?utm_source=x&id=2#results"),
            "https://example.com/paper?id=2",
        )

    def test_reference_extraction_prefers_reference_section_and_deduplicates_dois(self):
        text = """[Navigation](https://example.com/menu)

## References
1. [Memory paper](https://journal.test/paper?utm_source=x). doi:10.1234/ABC.7
2. The same DOI https://doi.org/10.1234/abc.7
3. Example, A. (2021). A sufficiently detailed plain-text memory consolidation citation. Journal 4, 20-30.
"""
        references = extract_reference_candidates(text, "https://origin.test/article")
        urls = {item["url"] for item in references}
        self.assertNotIn("https://example.com/menu", urls)
        self.assertIn("https://journal.test/paper", urls)
        self.assertIn("https://doi.org/10.1234/abc.7", urls)
        self.assertTrue(any(not item["url"] and item["id"].startswith("citation-") for item in references))

    def test_discovery_enforces_seed_then_expands_references_to_depth_limit(self):
        seeds = [
            {"title": "Seed A", "href": "https://papers.test/a", "body": "memory"},
            {"title": "Seed B", "href": "https://papers.test/b", "body": "memory"},
        ]
        pages = {
            "https://papers.test/a": "## References\n[Child C memory](https://papers.test/c)",
            "https://papers.test/b": "## References\n[Child D memory](https://papers.test/d)",
            "https://papers.test/c": "## References\n[Grandchild E memory](https://papers.test/e)",
            "https://papers.test/d": "## References\n[Grandchild E memory](https://papers.test/e)",
            "https://papers.test/e": "## References\n[Further F memory](https://papers.test/f)",
        }

        async def classify(goal, candidates):
            self.assertTrue(all("url" not in candidate for candidate in candidates))
            return CandidateSelection(
                keep_numbers={candidate["number"] for candidate in candidates},
                needs_abstract_numbers=set(),
            )

        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temporary:
            root = Path(temporary)
            executor = ResearchDiscoveryExecutor(FakeResearchManager(seeds, pages), root, classify)
            task = {
                "id": "task-graph",
                "definition": {
                    "goal": "biological memory mechanisms",
                    "metadata": {
                        "research_seed_sources": 2,
                        "research_depth_passes": 2,
                        "research_max_sources": 10,
                        "research_references_per_source": 5,
                    },
                },
            }
            outcome = asyncio.run(executor.execute_work_item(task, {}, []))
            ledger = json.loads((root / "research" / "task-graph" / "source-ledger.json").read_text())

        self.assertEqual(outcome.outcome, "completed")
        self.assertEqual(ledger["stop_reason"], "depth_limit_reached")
        self.assertEqual([row["depth"] for row in ledger["passes"]], [0, 1, 2])
        self.assertEqual(len([row for row in ledger["sources"] if row["depth"] == 0]), 2)
        self.assertEqual(len(ledger["sources"]), 5)
        grandchild = next(row for row in ledger["sources"] if row["url"] == "https://papers.test/e")
        self.assertEqual(len(grandchild["parents"]), 2)

    def test_discovery_stops_when_a_pass_has_no_novel_relevant_references(self):
        seeds = [{"title": "Seed", "href": "https://papers.test/a", "body": "memory"}]
        pages = {"https://papers.test/a": "## References\n[Unrelated](https://papers.test/z)"}

        selection_calls = 0

        async def reject_all(goal, candidates):
            nonlocal selection_calls
            selection_calls += 1
            return CandidateSelection(
                keep_numbers=(
                    {candidate["number"] for candidate in candidates} if selection_calls == 1 else set()
                ),
                needs_abstract_numbers=set(),
            )

        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temporary:
            root = Path(temporary)
            executor = ResearchDiscoveryExecutor(FakeResearchManager(seeds, pages), root, reject_all)
            task = {
                "id": "task-stop",
                "definition": {"goal": "memory", "metadata": {
                    "research_seed_sources": 1, "research_depth_passes": 5,
                }},
            }
            outcome = asyncio.run(executor.execute_work_item(task, {}, []))
            ledger = json.loads((root / "research" / "task-stop" / "source-ledger.json").read_text())

        self.assertEqual(outcome.outcome, "completed")
        self.assertEqual(ledger["stop_reason"], "no_novel_relevant_references")
        self.assertEqual(len(ledger["passes"]), 1)

    def test_source_cap_still_fetches_sources_admitted_to_the_frontier(self):
        seeds = [{"title": "Seed", "href": "https://papers.test/a", "body": "memory"}]
        pages = {
            "https://papers.test/a": (
                "## References\n[Child B memory](https://papers.test/b)\n"
                "[Child C memory](https://papers.test/c)\n[Child D memory](https://papers.test/d)"
            ),
            "https://papers.test/b": "No references.",
            "https://papers.test/c": "No references.",
        }

        async def classify(goal, candidates):
            return CandidateSelection(
                keep_numbers={candidate["number"] for candidate in candidates},
                needs_abstract_numbers=set(),
            )

        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temporary:
            root = Path(temporary)
            executor = ResearchDiscoveryExecutor(FakeResearchManager(seeds, pages), root, classify)
            task = {"id": "task-cap", "definition": {"goal": "memory", "metadata": {
                "research_seed_sources": 1,
                "research_depth_passes": 5,
                "research_max_sources": 3,
                "research_references_per_source": 5,
            }}}
            outcome = asyncio.run(executor.execute_work_item(task, {}, []))
            ledger = json.loads((root / "research" / "task-cap" / "source-ledger.json").read_text())

        self.assertEqual(outcome.outcome, "completed")
        self.assertEqual(ledger["stop_reason"], "source_limit_reached")
        self.assertEqual(len(ledger["sources"]), 3)
        self.assertTrue(all(source["status"] == "fetched" for source in ledger["sources"]))

    def test_ambiguous_number_gets_metadata_then_a_final_numbered_decision(self):
        url = "https://papers.test/a"
        manager = FakeResearchManager(
            [{"title": "Memory dynamics", "href": url, "body": "An ambiguous result snippet."}],
            {url: "No references."},
            {url: {
                "title": "Memory dynamics after learning",
                "authors": ["Ada Example"],
                "year": 2025,
                "venue": "Journal of Memory",
                "abstract": "Protein synthesis supports long-term memory consolidation.",
                "abstract_source": "citation_abstract",
            }},
        )
        calls = []

        async def classify(goal, candidates):
            calls.append(candidates)
            if len(calls) == 1:
                self.assertEqual(candidates[0]["context_kind"], "search_snippet")
                self.assertNotIn("abstract", candidates[0])
                return CandidateSelection(keep_numbers=set(), needs_abstract_numbers={1})
            self.assertEqual(candidates[0]["authors"], ["Ada Example"])
            self.assertEqual(candidates[0]["abstract_source"], "citation_abstract")
            self.assertNotIn("url", candidates[0])
            return CandidateSelection(keep_numbers={1}, needs_abstract_numbers=set())

        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temporary:
            root = Path(temporary)
            executor = ResearchDiscoveryExecutor(manager, root, classify)
            task = {"id": "task-metadata", "definition": {"goal": "memory", "metadata": {
                "research_seed_sources": 1, "research_depth_passes": 0,
            }}}
            outcome = asyncio.run(executor.execute_work_item(task, {}, []))

        self.assertEqual(outcome.outcome, "completed")
        self.assertEqual(len(calls), 2)
        self.assertTrue(any(name == "web__fetch_scholarly_metadata" for name, _ in manager.calls))

    def test_generic_page_description_is_not_presented_as_an_abstract(self):
        url = "https://papers.test/a"
        manager = FakeResearchManager(
            [{"title": "Memory dynamics", "href": url, "body": "Ambiguous search text."}],
            {url: "No references."},
            {url: {
                "title": "Memory dynamics",
                "description": "A publisher page description, not an abstract.",
                "description_source": "og:description",
            }},
        )
        calls = []

        async def classify(goal, candidates):
            calls.append(candidates)
            if len(calls) == 1:
                return CandidateSelection(keep_numbers=set(), needs_abstract_numbers={1})
            self.assertIsNone(candidates[0]["abstract"])
            self.assertEqual(candidates[0]["context_kind"], "page_description")
            self.assertEqual(candidates[0]["context"], "A publisher page description, not an abstract.")
            return CandidateSelection(keep_numbers=set(), needs_abstract_numbers=set())

        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temporary:
            root = Path(temporary)
            executor = ResearchDiscoveryExecutor(manager, root, classify)
            task = {"id": "task-description", "definition": {"goal": "memory", "metadata": {
                "research_seed_sources": 1, "research_depth_passes": 0,
            }}}
            outcome = asyncio.run(executor.execute_work_item(task, {}, []))

        self.assertEqual(outcome.outcome, "retry")
        self.assertEqual(len(calls), 2)

    def test_notes_are_batched_and_persisted_for_restart(self):
        calls = []

        async def generate(messages):
            payload = json.loads(messages[-1]["content"])
            calls.append([source["id"] for source in payload["sources"]])
            return "\n".join(f"{source['id']} {source['url']} notes" for source in payload["sources"])

        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temporary:
            root = Path(temporary)
            research = root / "research" / "task-notes"
            (research / "sources").mkdir(parents=True)
            sources = []
            for index in range(5):
                source = {"id": f"s{index}", "title": f"Source {index}", "url": f"https://x.test/{index}",
                          "depth": 0, "status": "fetched", "content_file": f"s{index}.md"}
                sources.append(source)
                (research / "sources" / source["content_file"]).write_text(f"content {index}")
            (research / "source-ledger.json").write_text(json.dumps({"sources": sources}))
            executor = ResearchNotesExecutor(generate, root)
            task = {"id": "task-notes", "definition": {"goal": "memory", "metadata": {
                "research_notes_batch_size": 2,
            }}}
            item = {"instructions": "Extract evidence."}
            first = asyncio.run(executor.execute_work_item(task, item, []))
            second = asyncio.run(executor.execute_work_item(task, item, []))

        self.assertEqual(first.outcome, "completed")
        self.assertEqual(second.outcome, "completed")
        self.assertEqual([len(batch) for batch in calls], [2, 2, 1])


if __name__ == "__main__":
    unittest.main()
