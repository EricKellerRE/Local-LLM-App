import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock

from pydantic import ValidationError

from local_model_app.server import Runtime, SettingsUpdateRequest


class RuntimePluginRequestTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
