import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from local_model_app.durable_tools import DurableSectionExecutor, DurableSynthesisExecutor, DurableToolExecutor


class DurableToolExecutorTests(unittest.TestCase):
    def test_background_tool_turn_uses_shared_inference_lock(self) -> None:
        lock = asyncio.Lock()
        observed = []

        class FakeCoordinator:
            def __init__(self, *args, **kwargs):
                pass

            async def respond(self, prompt, history, plugin_ids):
                observed.append(lock.locked())
                return "Evidence: https://example.com/source"

        model = SimpleNamespace(settings=SimpleNamespace(max_tool_calls_per_step=256))
        with tempfile.TemporaryDirectory() as temporary:
            executor = DurableToolExecutor(
                model,
                SimpleNamespace(),
                SimpleNamespace(),
                Path(temporary),
                SimpleNamespace(),
                lock,
            )
            task = {
                "id": "task-1",
                "definition": {
                    "goal": "Research a topic",
                    "metadata": {"plugin_ids": ["local.web-research"], "chat_id": "chat-1"},
                },
            }
            item = {
                "id": "item-1",
                "title": "Collect sources",
                "instructions": "Search and preserve URLs.",
                "completion_check": "Sources are recorded.",
                "result": None,
            }
            with patch("local_model_app.durable_tools.ToolCoordinator", FakeCoordinator):
                outcome = asyncio.run(executor.execute_work_item(task, item, []))

        self.assertEqual(observed, [True])
        self.assertFalse(lock.locked())
        self.assertEqual(outcome.outcome, "completed")

    def test_section_executor_retries_a_shallow_draft(self) -> None:
        async def generate(messages):
            return "Too short. https://example.com/source"

        executor = DurableSectionExecutor(generate)
        task = {"id": "task-1", "definition": {"goal": "Research memory"}}
        item = {
            "title": "Mechanisms",
            "instructions": "Explain the mechanisms.",
            "completion_check": "At least 600 substantive words with exact source URLs.",
            "depends_on": ["evidence"],
            "attempts": 1,
        }
        completed = [{
            "key": "evidence",
            "title": "Evidence",
            "result": {"result": "Finding https://example.com/source", "completion_evidence": []},
        }]

        outcome = asyncio.run(executor.execute_work_item(task, item, completed))

        self.assertEqual(outcome.outcome, "retry")
        self.assertIn("did not meet", outcome.summary)

    def test_section_executor_accumulates_checkpointed_segments_until_minimum(self) -> None:
        calls = []

        async def generate(messages):
            calls.append(messages)
            marker = "first" if len(calls) == 1 else "second"
            return " ".join([f"{marker} causal mechanism limitation https://example.com/{marker}"] * 50)

        with tempfile.TemporaryDirectory() as temporary:
            executor = DurableSectionExecutor(generate, Path(temporary))
            task = {"id": "task-1", "definition": {"goal": "Research memory", "metadata": {}}}
            item = {
                "id": "item-1",
                "key": "section-test",
                "title": "Mechanisms",
                "instructions": "Explain the mechanisms.",
                "completion_check": "At least 600 substantive words with exact source URLs.",
                "depends_on": ["evidence"],
                "attempts": 1,
            }
            completed = [{
                "key": "evidence",
                "title": "Evidence",
                "result": {"result": "Finding https://example.com/source", "completion_evidence": []},
            }]

            first = asyncio.run(executor.execute_work_item(task, item, completed))
            item["attempts"] = 2
            second = asyncio.run(executor.execute_work_item(task, item, completed))

        self.assertEqual(first.outcome, "retry")
        self.assertEqual(second.outcome, "completed")
        self.assertIn("first causal", second.result)
        self.assertIn("second causal", second.result)
        self.assertEqual(len(calls), 2)

    def test_research_synthesis_exports_a_docx_artifact_without_recompressing_sections(self) -> None:
        async def unused_generate(messages):
            raise AssertionError("Research assembly should not invoke the model again")

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            research_root = root / "research" / "task-1"
            (research_root / "source-analysis").mkdir(parents=True)
            (research_root / "source-analysis" / "paper.json").write_text("{}", encoding="utf-8")
            (research_root / "sections").mkdir()
            (research_root / "sections" / "section.json").write_text("{}", encoding="utf-8")
            (research_root / "paper-reports").mkdir()
            (research_root / "paper-reports" / "paper.md").write_text(
                "# Paper\n\nSource: https://example.com/paper\n\nDurable report.", encoding="utf-8"
            )
            (research_root / "sources").mkdir()
            (research_root / "sources" / "paper.md").write_text("Source text", encoding="utf-8")
            (research_root / "originals").mkdir()
            (research_root / "originals" / "paper.pdf").write_bytes(b"%PDF fixture")
            (research_root / "source-ledger.json").write_text(json.dumps({
                "sources": [{
                    "id": "paper",
                    "title": "Paper",
                    "url": "https://example.com/paper",
                    "depth": 0,
                    "status": "analyzed",
                    "analysis_file": "research/task-1/source-analysis/paper.json",
                    "paper_report_file": "research/task-1/paper-reports/paper.md",
                }],
            }), encoding="utf-8")
            executor = DurableSynthesisExecutor(unused_generate, root)
            task = {
                "id": "task-1",
                "definition": {
                    "title": "Memory mechanisms",
                    "goal": "Research memory",
                    "metadata": {
                        "mode": "durable_tools",
                        "report_min_words": 10,
                        "report_min_sources": 1,
                    },
                },
            }
            completed = [{
                "kind": "section",
                "sequence": 1,
                "title": "Molecular mechanisms",
                "result": {
                    "result": (
                        "## Molecular mechanisms\n\nLong-term memory depends on durable synaptic change "
                        "supported by transcription and translation. https://example.com/paper"
                    )
                },
            }]
            item = {"title": "Assemble report", "instructions": "Export it."}

            outcome = asyncio.run(executor.execute_work_item(task, item, completed))

            self.assertEqual(outcome.outcome, "completed")
            self.assertEqual(len(outcome.artifacts), 1)
            path = root / "artifacts" / outcome.artifacts[0].relative_path
            self.assertTrue(path.is_file())
            self.assertGreater(path.stat().st_size, 1000)
            markdown = path.with_name("report.md").read_text(encoding="utf-8")
            self.assertIn("durable synaptic change", markdown)
            self.assertFalse((research_root / "source-analysis").exists())
            self.assertFalse((research_root / "sections").exists())
            self.assertTrue((research_root / "paper-reports" / "paper.md").is_file())
            self.assertTrue((research_root / "sources" / "paper.md").is_file())
            self.assertTrue((research_root / "originals" / "paper.pdf").is_file())
            ledger = json.loads((research_root / "source-ledger.json").read_text(encoding="utf-8"))
            self.assertNotIn("analysis_file", ledger["sources"][0])
            self.assertTrue(ledger["sources"][0]["analysis_notes_deleted"])
            self.assertEqual(
                ledger["intermediate_note_cleanup"]["removed_directories"],
                ["source-analysis", "sections"],
            )


if __name__ == "__main__":
    unittest.main()
