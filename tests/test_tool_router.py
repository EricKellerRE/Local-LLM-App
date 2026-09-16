import unittest

from local_model_app.mcp_plugins import DiscoveredTool
from local_model_app.tool_router import ToolRouter


def make_tool(name: str, description: str) -> DiscoveredTool:
    return DiscoveredTool(
        exposed_name=f"demo__{name}",
        plugin_id="demo.plugin",
        server_id="demo",
        native_name=name,
        title=None,
        description=description,
        input_schema={"type": "object", "properties": {"secret_schema_field": {"type": "string"}}},
        output_schema=None,
        annotations={},
    )


class ToolRouterTests(unittest.TestCase):
    def test_ranks_metadata_and_limits_visible_schemas(self) -> None:
        tools = [
            make_tool("weather_forecast", "Forecast weather for a city"),
            make_tool("calendar_events", "Find meetings and calendar events"),
            make_tool("invoice_lookup", "Find accounting invoices"),
            make_tool("repository_search", "Search source code"),
            make_tool("send_email", "Send an email message"),
            make_tool("customer_record", "Read a customer record"),
        ]

        routed = ToolRouter().route("Will it rain? Show the city weather forecast.", tools)

        self.assertEqual(routed.visible[0].native_name, "weather_forecast")
        self.assertEqual(len(routed.visible), 4)
        self.assertTrue(routed.has_more)

    def test_compact_catalog_does_not_include_json_schemas(self) -> None:
        catalog = ToolRouter.compact_catalog([make_tool("lookup", "Look up an item")])

        self.assertIn("demo__lookup", catalog)
        self.assertNotIn("secret_schema_field", catalog)

    def test_low_confidence_route_expands_in_bounded_batches(self) -> None:
        tools = [make_tool(f"tool_{index}", f"Capability {index}") for index in range(10)]
        routed = ToolRouter().route("unrelated request", tools)

        self.assertTrue(routed.low_confidence)
        self.assertEqual(len(routed.visible), 4)
        self.assertEqual(len(routed.expanded().visible), 8)

    def test_semantic_encoder_recovers_paraphrase_and_is_cached(self) -> None:
        class FakeEncoder:
            name = "fake-semantic"

            def __init__(self) -> None:
                self.catalog_calls = 0

            def encode(self, texts):
                if len(texts) > 1:
                    self.catalog_calls += 1
                return [
                    [1.0, 0.0] if ("generator" in text or "power plant" in text) else [0.0, 1.0]
                    for text in texts
                ]

        encoder = FakeEncoder()
        router = ToolRouter(encoder)
        tools = [
            make_tool("top_generators", "Rank generating units by active output"),
            make_tool("calendar_events", "Find meetings and calendar events"),
        ]

        first = router.route("Which power plants are biggest?", tools)
        second = router.route("Compare power plant capacity", tools)

        self.assertEqual(first.visible[0].native_name, "top_generators")
        self.assertEqual(second.semantic_backend, "fake-semantic")
        self.assertEqual(encoder.catalog_calls, 1)

    def test_semantic_failure_falls_back_to_lexical_routing(self) -> None:
        class BrokenEncoder:
            name = "broken"

            def encode(self, texts):
                raise RuntimeError("weights unavailable")

        tools = [
            make_tool("weather_forecast", "Forecast weather for a city"),
            make_tool("calendar_events", "Find meetings and calendar events"),
        ]

        routed = ToolRouter(BrokenEncoder()).route("Show the weather forecast", tools)

        self.assertEqual(routed.visible[0].native_name, "weather_forecast")
        self.assertIn("weights unavailable", routed.semantic_error)

    def test_self_contained_writing_request_keeps_tool_selection_low_confidence(self) -> None:
        tools = [
            make_tool("job_status", "Read the current status of an external job"),
            make_tool("send_email", "Send an email message"),
        ]

        routed = ToolRouter().route("Write a friendly project status update for my manager.", tools)

        self.assertTrue(routed.no_tool_hint)
        self.assertTrue(routed.low_confidence)

    def test_broad_destructive_request_forces_low_confidence(self) -> None:
        tools = [
            make_tool("delete_record", "Delete one selected database record"),
            make_tool("list_models", "List available models"),
        ]

        routed = ToolRouter().route("Delete every model and erase the workstation.", tools)

        self.assertTrue(routed.no_tool_hint)
        self.assertTrue(routed.low_confidence)


if __name__ == "__main__":
    unittest.main()
