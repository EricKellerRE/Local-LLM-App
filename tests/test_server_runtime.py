import unittest
from types import SimpleNamespace

from pydantic import ValidationError

from local_model_app.server import Runtime, SettingsUpdateRequest


class RuntimePluginRequestTests(unittest.TestCase):
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

    def test_research_and_project_optimization_are_durable_requests(self) -> None:
        self.assertTrue(Runtime._durable_request("Research the literature and write a report", False))
        self.assertTrue(Runtime._durable_request("Optimize this grid until the metric improves", True))
        self.assertFalse(Runtime._durable_request("Explain voltage stability", True))


if __name__ == "__main__":
    unittest.main()
