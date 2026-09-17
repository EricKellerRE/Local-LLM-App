import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from local_model_app.research_pipeline import (
    ResearchDiscoveryExecutor,
    ResearchNotesExecutor,
    canonical_url,
    extract_reference_candidates,
)


class FakeResearchManager:
    def __init__(self, seeds, pages):
        self.seeds = seeds
        self.pages = pages
        self.calls = []
        self.tools = [
            SimpleNamespace(native_name="search_web", exposed_name="web__search_web"),
            SimpleNamespace(native_name="fetch_url", exposed_name="web__fetch_url"),
        ]

    async def ensure_started(self, plugin_ids):
        self.calls.append(("start", tuple(plugin_ids)))

    def tools_for_plugins(self, plugin_ids):
        return self.tools

    async def call_tool(self, exposed_name, arguments):
        self.calls.append((exposed_name, arguments))
        if exposed_name == "web__search_web":
            return {"structuredContent": {"results": self.seeds}}
        return {"structuredContent": {
            "url": arguments["url"],
            "content": self.pages[arguments["url"]],
            "complete": True,
        }}


class ResearchPipelineTests(unittest.TestCase):
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
            return {candidate["id"] for candidate in candidates}

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

        async def reject_all(goal, candidates):
            return set()

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
            return {candidate["id"] for candidate in candidates}

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
