import asyncio
import json
import unittest
from unittest.mock import patch
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

from local_model_app.mcp_plugins import DiscoveredTool
from local_model_app.model import AssistantReply
from local_model_app.scratchpad import Scratchpad
from local_model_app.tool_coordinator import ToolCoordinator, _compact_result


def make_tool(native_name: str, *, required: list[str] | None = None) -> DiscoveredTool:
    return DiscoveredTool(
        exposed_name=f"grid__{native_name}", plugin_id="grid.plugin", server_id="grid",
        native_name=native_name, title=None, description=f"Test tool {native_name}",
        input_schema={"type": "object", "properties": {name: {"type": "string"} for name in (required or [])}, "required": required or []},
        output_schema=None, annotations={},
    )


class FakeModel:
    def __init__(self, replies: list[AssistantReply] | None = None) -> None:
        self.replies = list(replies or [])
        self.chat_calls = []

    def generate(self, messages, **kwargs):
        return "Route, discover, inspect the schema, then execute only when required inputs are known."

    def chat(self, messages, *, tools, **kwargs):
        self.chat_calls.append({"messages": messages, "tools": tools, **kwargs})
        return self.replies.pop(0)


class FakeManager:
    def __init__(self, tools: list[DiscoveredTool], results: dict[str, dict]) -> None:
        policy = SimpleNamespace(max_calls_per_turn=8)
        manifest = SimpleNamespace(policy=policy)
        self.registry = SimpleNamespace(get_plugin=lambda plugin_id: SimpleNamespace(manifest=manifest))
        self.tools = tools
        self.results = results
        self.calls = []

    async def ensure_started(self, plugin_ids):
        self.started = list(plugin_ids)

    def tools_for_plugins(self, plugin_ids):
        return self.tools

    async def call_tool(self, name, arguments):
        self.calls.append((name, arguments))
        return self.results[name]


class ToolCoordinatorTests(unittest.TestCase):
    def test_compact_result_prefers_structured_content(self) -> None:
        result = {
            "isError": False,
            "content": [{"type": "text", "text": "duplicate and verbose"}],
            "structuredContent": {"ok": True, "capability": "scenario"},
        }
        self.assertEqual(_compact_result(result), {"isError": False, "data": {"ok": True, "capability": "scenario"}})

    def test_compact_result_omits_binary_payloads_and_bounds_large_data(self) -> None:
        binary = _compact_result({
            "isError": False,
            "structuredContent": {
                "contents": [{"uri": "file://plot.png", "mimeType": "image/png", "blob": "A" * 1000}],
            },
        })
        large = _compact_result({
            "isError": False,
            "structuredContent": {"rows": ["x" * 1000 for _ in range(100)]},
        })

        content = binary["data"]["contents"][0]
        self.assertNotIn("blob", content)
        self.assertTrue(content["payloadOmitted"])
        self.assertTrue(large["data"]["truncated"])
        self.assertEqual(large["truncation"]["strategy"], "deterministic_json_preview")

    def test_copied_windows_path_underscore_escape_is_repaired_only_if_it_exists(self) -> None:
        copied = r"E:\Data\Tornado\_Tracks\_1.csv"
        with patch("local_model_app.tool_coordinator.Path.exists", return_value=True):
            self.assertEqual(ToolCoordinator._path_for_extension(copied, "csv"), r"E:\Data\Tornado_Tracks_1.csv")
        with patch("local_model_app.tool_coordinator.Path.exists", return_value=False):
            self.assertEqual(ToolCoordinator._path_for_extension(copied, "csv"), copied)

    def test_staged_protocol_reaches_missing_input_without_extra_model_turns(self) -> None:
        route = make_tool("powerworld_route_request", required=["request_text"])
        list_tools = make_tool("powerworld_list_tools", required=["capability"])
        get_schema = make_tool("powerworld_get_tool_schema", required=["tool_name"])
        model = FakeModel()
        manager = FakeManager(
            [route, list_tools, get_schema],
            {
                route.exposed_name: {"isError": False, "structuredContent": {"ok": True, "capability": "tornado", "next_step": "Call powerworld_list_tools for this capability."}},
                list_tools.exposed_name: {"isError": False, "structuredContent": {"ok": True, "capability": "tornado", "tools": [{"name": "tornado.build_scenario"}]}},
                get_schema.exposed_name: {"isError": False, "structuredContent": {"ok": True, "tool": {"name": "tornado.build_scenario", "input_schema": {"type": "object", "required": ["paths_csv"], "properties": {"paths_csv": {"type": "string"}}}}}},
            },
        )

        with TemporaryDirectory(dir=Path.cwd()) as directory:
            root = Path(directory)
            activity = root / "activity.jsonl"
            coordinator = ToolCoordinator(model, manager, Scratchpad(root / "scratchpad.jsonl"), activity)
            answer = asyncio.run(coordinator.respond(r"Run one tornado on C:\cases\study.pwb", [], ["grid.plugin"]))

            self.assertIn("`paths_csv`", answer)
            self.assertEqual([name for name, _ in manager.calls], [route.exposed_name, list_tools.exposed_name, get_schema.exposed_name])
            self.assertEqual(model.chat_calls, [])
            events = [json.loads(line)["event"] for line in activity.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(events, ["plan", "tool_routing", "tool_call", "tool_result", "tool_call", "tool_result", "tool_call", "tool_result", "waiting_for_input"])

    def test_generic_tool_call_is_observed_before_answer(self) -> None:
        tool = make_tool("lookup", required=["query"])
        model = FakeModel([
            AssistantReply(content="", tool_calls=[{"id": "call_1", "type": "function", "function": {"name": tool.exposed_name, "arguments": '{"query":"cases"}'}}], reasoning_content="Use lookup."),
            AssistantReply(content="The lookup completed.", tool_calls=[]),
        ])
        model.settings = SimpleNamespace(max_new_tokens=12_000)
        manager = FakeManager([tool], {tool.exposed_name: {"isError": False, "structuredContent": {"ok": True}}})

        with TemporaryDirectory(dir=Path.cwd()) as directory:
            root = Path(directory)
            coordinator = ToolCoordinator(model, manager, Scratchpad(root / "scratchpad.jsonl"), root / "activity.jsonl")
            answer = asyncio.run(coordinator.respond("look up cases", [], ["grid.plugin"]))

        self.assertEqual(answer, "The lookup completed.")
        self.assertEqual(manager.calls, [(tool.exposed_name, {"query": "cases"})])
        self.assertEqual(model.chat_calls[0]["max_new_tokens"], 192)
        self.assertEqual(model.chat_calls[1]["max_new_tokens"], 12_000)
        tool_message = next(message for message in model.chat_calls[1]["messages"] if message["role"] == "tool")
        self.assertNotIn("duplicate", tool_message["content"])

    def test_post_tool_decision_has_an_independent_budget_and_turn_class(self) -> None:
        tool = make_tool("lookup", required=["query"])
        model = FakeModel([
            AssistantReply(content="", tool_calls=[{
                "id": "call_1", "type": "function",
                "function": {"name": tool.exposed_name, "arguments": '{"query":"memory"}'},
            }]),
            AssistantReply(content="Evidence recorded.", tool_calls=[]),
        ])
        model.settings = SimpleNamespace(
            max_new_tokens=8192,
            tool_action_max_new_tokens=384,
            post_tool_decision_max_new_tokens=1536,
        )
        manager = FakeManager([tool], {tool.exposed_name: {
            "isError": False, "structuredContent": {"results": ["source"]},
        }})

        with TemporaryDirectory(dir=Path.cwd()) as directory:
            root = Path(directory)
            coordinator = ToolCoordinator(
                model,
                manager,
                Scratchpad(root / "scratchpad.jsonl"),
                root / "activity.jsonl",
                telemetry_context={"task_id": "task-1", "work_item_id": "item-1"},
            )
            answer = asyncio.run(coordinator.respond("Research memory", [], ["grid.plugin"]))

        self.assertEqual(answer, "Evidence recorded.")
        self.assertEqual(model.chat_calls[0]["max_new_tokens"], 384)
        self.assertEqual(model.chat_calls[0]["generation_class"], "tool_action")
        self.assertEqual(model.chat_calls[1]["max_new_tokens"], 1536)
        self.assertEqual(model.chat_calls[1]["generation_class"], "post_tool_decision")
        self.assertEqual(model.chat_calls[1]["telemetry_context"]["work_item_id"], "item-1")

    def test_chunked_fetch_continues_deterministically_without_model_url_rewrite(self) -> None:
        url = (
            "https://example.org/paper?comparison=early-versus-late&"
            "source=exact-long-url"
        )
        fetch = make_tool("fetch_url", required=["url"])
        model = FakeModel([
            AssistantReply(content="", tool_calls=[{
                "id": "call_1", "type": "function",
                "function": {"name": fetch.exposed_name, "arguments": {"url": url}},
            }]),
            AssistantReply(content="Source fully read.", tool_calls=[]),
        ])
        manager = FakeManager([fetch], {fetch.exposed_name: {
            "isError": False,
            "structuredContent": {"url": url, "content": "chunk", "next_start": 30000, "complete": False},
        }})
        # The deterministic continuation sees a terminal result on the second call.
        calls = 0

        async def call_tool(exposed_name, arguments, **kwargs):
            nonlocal calls
            calls += 1
            manager.calls.append((exposed_name, arguments))
            if calls == 1:
                return {"isError": False, "structuredContent": {
                    "url": url, "content": "first", "next_start": 30000, "complete": False,
                }}
            return {"isError": False, "structuredContent": {
                "url": url, "content": "second", "next_start": None, "complete": True,
            }}

        manager.call_tool = call_tool
        with TemporaryDirectory(dir=Path.cwd()) as directory:
            root = Path(directory)
            coordinator = ToolCoordinator(
                model, manager, Scratchpad(root / "scratchpad.jsonl"), root / "activity.jsonl",
            )
            answer = asyncio.run(coordinator.respond("Read the complete source", [], ["grid.plugin"]))

        self.assertEqual(answer, "Source fully read.")
        self.assertEqual(manager.calls[1], (fetch.exposed_name, {"url": url, "start_index": 30000}))
        self.assertEqual(len(model.chat_calls), 2)

    def test_tool_checkpoint_interval_compacts_and_continues(self) -> None:
        tool = make_tool("lookup", required=["query"])
        model = FakeModel([
            AssistantReply(content="", tool_calls=[{
                "id": "call_1",
                "type": "function",
                "function": {"name": tool.exposed_name, "arguments": '{"query":"first"}'},
            }]),
            AssistantReply(content="", tool_calls=[{
                "id": "call_2",
                "type": "function",
                "function": {"name": tool.exposed_name, "arguments": '{"query":"second"}'},
            }]),
            AssistantReply(content="Both lookups completed.", tool_calls=[]),
        ])
        model.settings = SimpleNamespace(max_new_tokens=12_000)
        manager = FakeManager([tool], {tool.exposed_name: {
            "isError": False, "structuredContent": {"ok": True},
        }})

        with TemporaryDirectory(dir=Path.cwd()) as directory:
            root = Path(directory)
            activity = root / "activity.jsonl"
            coordinator = ToolCoordinator(
                model,
                manager,
                Scratchpad(root / "scratchpad.jsonl"),
                activity,
                max_calls=1,
            )
            answer = asyncio.run(coordinator.respond("Perform both lookups", [], ["grid.plugin"]))
            events = [
                json.loads(line)["event"]
                for line in activity.read_text(encoding="utf-8").splitlines()
            ]

        self.assertEqual(answer, "Both lookups completed.")
        self.assertEqual(
            manager.calls,
            [
                (tool.exposed_name, {"query": "first"}),
                (tool.exposed_name, {"query": "second"}),
            ],
        )
        self.assertGreaterEqual(events.count("tool_checkpoint"), 1)

    def test_generic_router_exposes_only_top_schema_batch(self) -> None:
        weather = make_tool("weather_forecast", required=["city"])
        tools = [weather] + [make_tool(name) for name in (
            "calendar_events", "invoice_lookup", "repository_search", "send_email", "customer_record",
        )]
        model = FakeModel([
            AssistantReply(content="", tool_calls=[{
                "id": "call_weather", "type": "function",
                "function": {"name": weather.exposed_name, "arguments": '{"city":"Chicago"}'},
            }]),
            AssistantReply(content="Rain is expected.", tool_calls=[]),
        ])
        manager = FakeManager(tools, {weather.exposed_name: {
            "isError": False, "structuredContent": {"forecast": "rain"},
        }})

        with TemporaryDirectory(dir=Path.cwd()) as directory:
            root = Path(directory)
            coordinator = ToolCoordinator(
                model, manager, Scratchpad(root / "scratchpad.jsonl"), root / "activity.jsonl",
            )
            answer = asyncio.run(coordinator.respond("Show the Chicago weather forecast", [], ["grid.plugin"]))

        first_specs = model.chat_calls[0]["tools"]
        self.assertEqual(answer, "Rain is expected.")
        self.assertEqual(len(first_specs), 4)
        self.assertIn(weather.exposed_name, [item["function"]["name"] for item in first_specs])

    def test_supplied_missing_input_resumes_pending_backend_without_rerouting(self) -> None:
        route = make_tool("powerworld_route_request", required=["request_text"])
        get_schema = make_tool("powerworld_get_tool_schema", required=["tool_name"])
        gateway = make_tool("powerworld_call_tool", required=["tool_name", "arguments"])
        model = FakeModel([
            AssistantReply(content="", tool_calls=[{
                "id": "call_resume", "type": "function",
                "function": {
                    "name": gateway.exposed_name,
                    "arguments": '{"tool_name":"tornado.build_scenario","arguments":{"paths_csv":"E:\\\\tracks.csv","case_path":"C:\\\\study.pwb"}}',
                },
            }]),
        ])
        manager = FakeManager(
            [route, get_schema, gateway],
            {
                get_schema.exposed_name: {"isError": False, "structuredContent": {"ok": True, "tool": {
                    "name": "tornado.build_scenario",
                    "input_schema": {"type": "object", "required": ["paths_csv"], "properties": {
                        "paths_csv": {"type": "string"}, "case_path": {"type": ["string", "null"]},
                    }},
                }}},
                gateway.exposed_name: {"isError": True, "error": "approval_required", "message": "Approval required."},
            },
        )

        with TemporaryDirectory(dir=Path.cwd()) as directory:
            root = Path(directory)
            activity = root / "activity.jsonl"
            state = activity.with_suffix(".state.json")
            state.write_text(json.dumps({
                "status": "waiting_for_input",
                "backend_tool": "tornado.build_scenario",
                "required_inputs": ["paths_csv"],
            }), encoding="utf-8")
            coordinator = ToolCoordinator(model, manager, Scratchpad(root / "scratchpad.jsonl"), activity)
            answer = asyncio.run(coordinator.respond(r"paths_csv is E:\tracks.csv", [
                {"role": "user", "content": r"Use C:\study.pwb"},
                {"role": "assistant", "content": "Please provide paths_csv."},
            ], ["grid.plugin"]))

            self.assertIn("Approval is required", answer)
            self.assertEqual([name for name, _ in manager.calls], [get_schema.exposed_name, gateway.exposed_name])
            self.assertEqual(model.chat_calls, [])
            gateway_arguments = manager.calls[1][1]
            self.assertEqual(gateway_arguments["tool_name"], "tornado.build_scenario")
            self.assertEqual(gateway_arguments["arguments"]["paths_csv"], r"E:\tracks.csv")
            self.assertEqual(gateway_arguments["arguments"]["case_path"], r"C:\study.pwb")
            saved = json.loads(state.read_text(encoding="utf-8"))
            self.assertEqual(saved["status"], "waiting_for_approval")


if __name__ == "__main__":
    unittest.main()
