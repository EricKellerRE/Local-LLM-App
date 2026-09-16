import unittest
from types import SimpleNamespace

from local_model_app.server import Runtime


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


if __name__ == "__main__":
    unittest.main()
