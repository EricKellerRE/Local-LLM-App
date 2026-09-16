import asyncio
import unittest
from types import SimpleNamespace

from mcp import types

from local_model_app.mcp_manager import ActivePlugin, McpPluginManager
from local_model_app.mcp_plugins import DiscoveredTool


class FakeTaskSession:
    def __init__(self) -> None:
        self.methods: list[str] = []

    async def send_request(self, request, result_type):
        self.methods.append(request.method)
        if request.method == "tools/call":
            return types.CreateTaskResult(task=types.Task(
                taskId="11111111-1111-1111-1111-111111111111",
                status="working",
                statusMessage="started",
                createdAt="2026-09-16T00:00:00Z",
                lastUpdatedAt="2026-09-16T00:00:00Z",
                ttl=604800000,
                pollInterval=1,
            ))
        if request.method == "tasks/get":
            return types.GetTaskResult(
                taskId="11111111-1111-1111-1111-111111111111",
                status="completed",
                statusMessage="done",
                createdAt="2026-09-16T00:00:00Z",
                lastUpdatedAt="2026-09-16T00:01:00Z",
                ttl=604800000,
                pollInterval=1,
            )
        return types.CallToolResult(
            content=[types.TextContent(type="text", text='{"ok": true}')],
            structuredContent={"ok": True, "artifact": "result.aux"},
            isError=False,
        )


class McpDurableTaskTests(unittest.TestCase):
    def test_required_tool_uses_task_protocol_and_checkpoints_handle(self) -> None:
        policy = SimpleNamespace(
            deny_tools=[], allow_tools=["contingency.solve_all"], default_access="ask"
        )
        registry = SimpleNamespace(
            get_plugin=lambda plugin_id: SimpleNamespace(
                manifest=SimpleNamespace(policy=policy)
            )
        )
        manager = McpPluginManager(registry)
        tool = DiscoveredTool(
            exposed_name="grid__contingency_solve_all",
            plugin_id="grid-workshop.powerworld",
            server_id="powerworld-aux",
            native_name="contingency.solve_all",
            title=None,
            description="Run a durable contingency solve.",
            input_schema={"type": "object", "properties": {}},
            output_schema={"type": "object"},
            annotations={},
            execution={"taskSupport": "required"},
        )
        session = FakeTaskSession()
        active = ActivePlugin(
            clients={"powerworld-aux": SimpleNamespace(session=session)},
            tools=[tool],
        )
        active.ready_event.set()
        manager._active[tool.plugin_id] = active
        handles: list[dict] = []

        async def checkpoint(handle: dict) -> None:
            handles.append(handle)

        result = asyncio.run(manager.call_tool(
            tool.exposed_name,
            {},
            task_checkpoint=checkpoint,
        ))

        self.assertEqual(
            session.methods, ["tools/call", "tasks/get", "tasks/result"]
        )
        self.assertEqual(handles[0]["task_id"], "11111111-1111-1111-1111-111111111111")
        self.assertEqual(result["structuredContent"]["artifact"], "result.aux")


if __name__ == "__main__":
    unittest.main()
