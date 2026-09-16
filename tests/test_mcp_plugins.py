import asyncio
import json
import os
import socket
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

from local_model_app.mcp_plugins import (
    CompanionProcessConfig,
    DiscoveredTool,
    McpPluginRegistry,
    PluginConfigurationError,
    _safe_tool_name,
    normalize_tool_result,
)


class CompanionProcessTests(unittest.IsolatedAsyncioTestCase):
    async def test_owned_companion_starts_becomes_ready_and_stops(self) -> None:
        with TemporaryDirectory(dir=Path.cwd()) as directory:
            with socket.socket() as reservation:
                reservation.bind(("127.0.0.1", 0))
                port = reservation.getsockname()[1]
            companion = CompanionProcessConfig(
                id="test-http",
                command=os.sys.executable,
                args=["-m", "http.server", str(port), "--bind", "127.0.0.1"],
                cwd=directory,
                ready_url=f"http://127.0.0.1:{port}",
                timeout_seconds=10,
            )
            registry = McpPluginRegistry.__new__(McpPluginRegistry)

            processes = await registry._start_companions([companion])
            self.assertTrue(await asyncio.to_thread(registry._url_is_ready, companion.ready_url))
            await registry._stop_companions(processes)

            self.assertIsNotNone(processes[0].returncode)


class McpPluginRegistryTests(unittest.TestCase):
    def test_loads_and_expands_stdio_manifest(self) -> None:
        with TemporaryDirectory(dir=Path.cwd()) as directory:
            root = Path(directory)
            config = root / "mcp.d"
            config.mkdir()
            manifest = {
                "manifest_version": 1,
                "id": "example.tools",
                "tool_namespace": "example",
                "name": "Example",
                "version": "1.0.0",
                "servers": [
                    {
                        "id": "local",
                        "transport": "stdio",
                        "command": "${PROJECT_ROOT}/python.exe",
                        "args": ["server.py"],
                        "cwd": "${SERVER_ROOT}",
                        "companions": [{
                            "id": "backend",
                            "command": "${PYTHON_EXECUTABLE}",
                            "args": ["-m", "demo.backend"],
                            "cwd": "${SERVER_ROOT}",
                            "ready_url": "http://127.0.0.1:9000/health"
                        }]
                    }
                ]
            }
            (config / "example.json").write_text(json.dumps(manifest), encoding="utf-8")
            registry = McpPluginRegistry(config, project_root=root, variables={"SERVER_ROOT": "C:/example"})
            server = registry._expanded_server(registry.plugins[0].manifest.servers[0])
            self.assertEqual(server.command, f"{root.resolve()}/python.exe")
            self.assertEqual(server.cwd, "C:/example")
            self.assertEqual(server.companions[0].command, os.sys.executable)
            self.assertEqual(server.companions[0].cwd, "C:/example")

    def test_missing_variable_is_rejected(self) -> None:
        with TemporaryDirectory(dir=Path.cwd()) as directory:
            root = Path(directory)
            config = root / "mcp.d"
            config.mkdir()
            (config / "bad.json").write_text(
                json.dumps({
                    "manifest_version": 1,
                    "id": "bad.plugin",
                    "tool_namespace": "bad",
                    "name": "Bad",
                    "version": "1",
                    "servers": [{"id": "one", "transport": "stdio", "command": "${MISSING}/server"}]
                }),
                encoding="utf-8",
            )
            registry = McpPluginRegistry(config, project_root=root, variables={})
            with self.assertRaises(PluginConfigurationError):
                registry._expanded_server(registry.plugins[0].manifest.servers[0])

    def test_tool_names_are_namespaced_and_bounded(self) -> None:
        name = _safe_tool_name("grid", "comparison.compare_case_folder_objects")
        self.assertTrue(name.startswith("grid__comparison_compare_case"))
        self.assertLessEqual(len(name), 64)

    def test_default_ask_policy_blocks_unapproved_calls(self) -> None:
        with TemporaryDirectory(dir=Path.cwd()) as directory:
            root = Path(directory)
            config = root / "mcp.d"
            config.mkdir()
            (config / "guarded.json").write_text(
                json.dumps({
                    "manifest_version": 1,
                    "id": "guarded.plugin",
                    "tool_namespace": "guarded",
                    "name": "Guarded",
                    "version": "1",
                    "servers": [{"id": "one", "transport": "stdio", "command": "does-not-run"}],
                    "policy": {"default_access": "ask"}
                }),
                encoding="utf-8",
            )
            registry = McpPluginRegistry(config, project_root=root)
            tool = DiscoveredTool(
                exposed_name="guarded__change",
                plugin_id="guarded.plugin",
                server_id="one",
                native_name="change",
                title=None,
                description="",
                input_schema={"type": "object", "properties": {}},
                output_schema=None,
                annotations={"readOnlyHint": True},
            )
            import asyncio

            with self.assertRaisesRegex(PermissionError, "requires user approval"):
                asyncio.run(registry.call_tool(tool, {}))

    def test_annotations_use_mcp_alias_names(self) -> None:
        from mcp import types
        import asyncio

        advertised = SimpleNamespace(
            name="inspect", title=None, description="Inspect safely",
            input_schema={"type": "object", "properties": {}}, output_schema=None,
            annotations=types.ToolAnnotations(readOnlyHint=True, destructiveHint=False),
        )
        client = SimpleNamespace(list_tools=lambda **kwargs: None)

        async def list_tools(**kwargs):
            return SimpleNamespace(tools=[advertised], next_cursor=None)

        client.list_tools = list_tools
        registry = McpPluginRegistry.__new__(McpPluginRegistry)
        plugin = SimpleNamespace(manifest=SimpleNamespace(id="demo.plugin", tool_namespace="demo"))
        server = SimpleNamespace(id="demo")
        discovered = asyncio.run(registry.discover_connected_tools(plugin, server, client))

        self.assertEqual(discovered[0].annotations["readOnlyHint"], True)
        self.assertNotIn("read_only_hint", discovered[0].annotations)

    def test_output_schema_failure_is_quarantined(self) -> None:
        class Dumpable:
            def model_dump(self, **kwargs):
                return {"type": "text", "text": "raw result"}

        tool = DiscoveredTool(
            exposed_name="demo__count", plugin_id="demo.plugin", server_id="demo",
            native_name="count", title=None, description="Count",
            input_schema={"type": "object"},
            output_schema={"$schema": "https://json-schema.org/draft/2020-12/schema", "type": "object", "required": ["count"], "properties": {"count": {"type": "integer"}}},
            annotations={},
        )
        result = SimpleNamespace(content=[Dumpable()], structured_content={"count": "three"}, is_error=False)

        normalized = normalize_tool_result(tool, result)

        self.assertTrue(normalized["isError"])
        self.assertEqual(normalized["error"], "output_validation_failed")
        self.assertIsNone(normalized["structuredContent"])
        self.assertEqual(normalized["rawResult"]["structuredContent"], {"count": "three"})

    def test_guidance_is_selected_by_routed_capability_and_versioned(self) -> None:
        with TemporaryDirectory(dir=Path.cwd()) as directory:
            root = Path(directory)
            config = root / "mcp.d"
            config.mkdir()
            guide = root / "case-guide.md"
            guide.write_text("Case-specific operating guidance.", encoding="utf-8")
            (config / "demo.json").write_text(json.dumps({
                "manifest_version": 1,
                "id": "demo.plugin",
                "tool_namespace": "demo",
                "name": "Demo",
                "version": "1",
                "servers": [{"id": "one", "transport": "stdio", "command": "does-not-run"}],
                "guidance": [{"path": "${GUIDE_PATH}", "capability": "case"}],
            }), encoding="utf-8")
            registry = McpPluginRegistry(config, project_root=root, variables={"GUIDE_PATH": str(guide)})
            case_tool = DiscoveredTool(
                exposed_name="demo__case_overview", plugin_id="demo.plugin", server_id="one",
                native_name="case_overview", title=None, description="Summarize a case",
                input_schema={"type": "object"}, output_schema=None, annotations={},
            )
            unrelated = DiscoveredTool(
                exposed_name="demo__weather", plugin_id="demo.plugin", server_id="one",
                native_name="weather", title=None, description="Get weather",
                input_schema={"type": "object"}, output_schema=None, annotations={},
            )

            selected = registry.guidance_for_tools([unrelated, case_tool])

            self.assertEqual(len(selected), 1)
            self.assertEqual(selected[0].capability, "case")
            self.assertTrue(selected[0].version.startswith("sha256:"))


if __name__ == "__main__":
    unittest.main()
