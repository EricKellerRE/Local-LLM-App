import json
import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from local_model_app.mcp_plugins import DiscoveredTool, McpPluginRegistry, PluginConfigurationError, _safe_tool_name


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
                        "cwd": "${SERVER_ROOT}"
                    }
                ]
            }
            (config / "example.json").write_text(json.dumps(manifest), encoding="utf-8")
            registry = McpPluginRegistry(config, project_root=root, variables={"SERVER_ROOT": "C:/example"})
            server = registry._expanded_server(registry.plugins[0].manifest.servers[0])
            self.assertEqual(server.command, f"{root.resolve()}/python.exe")
            self.assertEqual(server.cwd, "C:/example")

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


if __name__ == "__main__":
    unittest.main()
