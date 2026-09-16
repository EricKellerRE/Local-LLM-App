from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from local_model_app.config import _load_dotenv
from local_model_app.mcp_plugins import McpPluginRegistry


async def run(command: str, *, tool_name: str | None = None, arguments: str = "{}") -> int:
    _load_dotenv(ROOT / ".env")
    registry = McpPluginRegistry(ROOT / "config" / "mcp.d", project_root=ROOT)
    if command == "list":
        print(json.dumps([plugin.manifest.model_dump() for plugin in registry.plugins], indent=2))
        return 0
    tools = await registry.discover_tools()
    if command == "call":
        matches = [tool for tool in tools if tool.exposed_name == tool_name or tool.native_name == tool_name]
        if len(matches) != 1:
            raise SystemExit(f"Expected one matching tool for {tool_name!r}; found {len(matches)}")
        result = await registry.call_tool(matches[0], json.loads(arguments))
        print(json.dumps({"tool": matches[0].exposed_name, "result": result}, indent=2))
        return 0
    print(
        json.dumps(
            {
                "plugins": len(registry.enabled_plugins()),
                "capabilities": len(tools),
                "catalog": [
                    {
                        "exposed_name": tool.exposed_name,
                        "native_name": tool.native_name,
                        "kind": tool.kind,
                        "plugin": tool.plugin_id,
                        "server": tool.server_id,
                    }
                    for tool in tools
                ],
            },
            indent=2,
        )
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect and validate Local Model MCP plugins.")
    parser.add_argument("command", choices=("list", "validate", "call"))
    parser.add_argument("--tool")
    parser.add_argument("--arguments", default="{}")
    args = parser.parse_args()
    return asyncio.run(run(args.command, tool_name=args.tool, arguments=args.arguments))


if __name__ == "__main__":
    raise SystemExit(main())
