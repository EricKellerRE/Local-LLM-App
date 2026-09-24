import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

from pydantic import ValidationError

from local_model_app.server import Runtime, SettingsUpdateRequest
from local_model_app.workflow_skills import WorkflowSkillRegistry


class RuntimePluginRequestTests(unittest.TestCase):
    def test_source_classifier_distinguishes_a_primary_paper_from_a_research_summary(self) -> None:
        captured = {}

        class FakeModel:
            settings = SimpleNamespace(
                research_classifier_max_new_tokens=256,
                max_new_tokens=512,
            )

            def generate(self, messages, **kwargs):
                captured["messages"] = messages
                captured["kwargs"] = kwargs
                return '{"keep_numbers":[],"needs_abstract_numbers":[],"assessments":[]}'

        runtime = Runtime.__new__(Runtime)
        runtime.model = FakeModel()
        runtime._inference_lock = asyncio.Lock()
        runtime._active_operations = {}
        runtime.ensure_loaded = AsyncMock()

        asyncio.run(runtime.classify_references("memory", [{"number": 1, "title": "Research summary"}]))

        prompt = captured["messages"][0]["content"]
        self.assertIn("genre of the document actually supplied", prompt)
        self.assertIn("university research profile", prompt)
        self.assertIn("merely describes original research", prompt)
        self.assertIn("missing venue/publication identity", prompt)
        self.assertIn("decisive genre clues", prompt)
        self.assertEqual(captured["kwargs"]["generation_class"], "research_relevance")

    def test_task_planning_and_report_synthesis_use_separate_budgets(self) -> None:
        calls = []

        class FakeModel:
            settings = SimpleNamespace(
                max_new_tokens=8192,
                tool_action_max_new_tokens=1024,
                tool_temperature=0.0,
                temperature=0.7,
                section_max_new_tokens=3072,
            )

            def generate(self, messages, **kwargs):
                calls.append(kwargs)
                return "ok"

        runtime = Runtime.__new__(Runtime)
        runtime.model = FakeModel()
        runtime._inference_lock = asyncio.Lock()
        runtime._active_operations = {}
        runtime.ensure_loaded = AsyncMock()

        asyncio.run(runtime.task_generate([{"role": "user", "content": "plan"}]))
        asyncio.run(runtime.task_write_section([{"role": "user", "content": "section"}]))
        asyncio.run(runtime.task_synthesize([{"role": "user", "content": "report"}]))

        self.assertEqual(calls[0], {
            "max_new_tokens": 1024,
            "temperature": 0.0,
            "generation_class": "task_planning",
        })
        self.assertEqual(calls[1], {
            "max_new_tokens": 3072,
            "temperature": 0.7,
            "generation_class": "report_section",
        })
        self.assertEqual(calls[2], {
            "max_new_tokens": 8192,
            "temperature": 0.7,
            "generation_class": "final_synthesis",
        })

    def test_specific_mcp_name_is_detected_in_chat_request(self) -> None:
        runtime = Runtime.__new__(Runtime)
        manifest = SimpleNamespace(
            id="grid-workshop.powerworld",
            name="Grid Workshop PowerWorld",
            tool_namespace="grid",
        )
        runtime.mcp = SimpleNamespace(
            registry=SimpleNamespace(plugins=[SimpleNamespace(manifest=manifest)])
        )

        self.assertEqual(
            runtime.requested_plugins("Please use the PowerWorld MCP server for this."),
            ["grid-workshop.powerworld"],
        )
        self.assertEqual(runtime.requested_plugins("Just explain this paragraph."), [])

    def test_settings_reject_impossible_token_budgets(self) -> None:
        common = {
            "data_directory": "C:/data",
            "models_directory": "C:/models",
        }
        with self.assertRaisesRegex(ValidationError, "smaller than the context window"):
            SettingsUpdateRequest(**common, context_window=512, max_new_tokens=512)
        with self.assertRaisesRegex(ValidationError, "cannot exceed the response budget"):
            SettingsUpdateRequest(**common, max_new_tokens=512, reasoning_budget=513)
        with self.assertRaisesRegex(ValidationError, "source cap"):
            SettingsUpdateRequest(**common, research_seed_sources=20, research_max_sources=10)

    def test_research_and_project_optimization_are_durable_requests(self) -> None:
        self.assertTrue(Runtime._durable_request("Research the literature and write a report", False))
        self.assertTrue(Runtime._durable_request("Optimize this grid until the metric improves", True))
        self.assertFalse(Runtime._durable_request("Explain voltage stability", True))

    def test_explicit_primary_source_policy_keeps_reviews_only_for_reference_mining(self) -> None:
        policy = Runtime._explicit_source_policy(
            "Use primary sources. Literature reviews can be kept for reference mining but not analyzed as primary sources."
        )
        self.assertEqual(policy["core_types"], ["primary_study"])
        self.assertEqual(policy["supplemental_types"], ["systematic_review", "scholarly_review"])
        self.assertTrue(policy["follow_supplemental_references"])
        self.assertFalse(policy["supplemental_counts_toward_target"])
        self.assertIsNone(Runtime._explicit_source_policy("Write a broad report using appropriate sources."))

    def test_durable_research_chat_instantiates_the_workflow_skill(self) -> None:
        captured = {}

        def create_task(request, definition):
            captured["request"] = request
            captured["definition"] = definition
            return {"id": "task-1"}

        runtime = Runtime.__new__(Runtime)
        runtime.workflow_skills = WorkflowSkillRegistry.default()
        runtime.model = SimpleNamespace(settings=SimpleNamespace(
            research_seed_sources=12,
            research_depth_passes=3,
            research_max_sources=80,
            research_references_per_source=12,
            research_notes_batch_size=6,
        ))
        runtime.tasks = SimpleNamespace(create_task=create_task, start_task=Mock())
        runtime.task_engine = SimpleNamespace(wake=Mock())
        runtime.store = SimpleNamespace(append_exchange=Mock())
        request = (
            "Produce a comprehensive research report on biological mechanisms of long-term memory formation and "
            "maintenance. Search deeply, follow references, and export the complete document."
        )

        asyncio.run(runtime._start_durable_chat_task(
            {"id": "chat-1", "project_id": None},
            request,
            ["local.web-research"],
        ))

        definition = captured["definition"]
        self.assertEqual(definition.goal, "biological mechanisms of long-term memory formation and maintenance")
        self.assertEqual(definition.metadata["workflow_skill_id"], "scholarly-research-report")
        self.assertEqual(definition.metadata["skill_inputs"]["topic"], definition.goal)
        runtime.tasks.start_task.assert_called_once_with("task-1")
        runtime.task_engine.wake.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
