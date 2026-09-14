"""Manual smoke test for the real staged Grid MCP protocol without loading model weights."""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path

from local_model_app.config import _load_dotenv
from local_model_app.mcp_manager import McpPluginManager
from local_model_app.mcp_plugins import McpPluginRegistry
from local_model_app.model import AssistantReply
from local_model_app.scratchpad import Scratchpad
from local_model_app.tool_coordinator import ToolCoordinator


ROOT = Path(__file__).resolve().parents[1]


class ScriptedPlanner:
    """Select the expected single-track tool; protocol plumbing remains real."""

    def generate(self, messages, **kwargs):
        return "Route the request, select the single-tornado tool, inspect its schema, and stop for missing inputs."

    def chat(self, messages, *, tools, **kwargs):
        schema_tool = tools[0]["function"]["name"]
        return AssistantReply(
            content="",
            tool_calls=[{
                "id": "call_select_tornado",
                "type": "function",
                "function": {
                    "name": schema_tool,
                    "arguments": {"tool_name": "tornado.build_scenario"},
                },
            }],
        )


async def main() -> None:
    _load_dotenv(ROOT / ".env")
    manager = McpPluginManager(McpPluginRegistry(ROOT / "config" / "mcp.d", project_root=ROOT))
    try:
        with tempfile.TemporaryDirectory(dir=ROOT / "data") as directory:
            temporary = Path(directory)
            coordinator = ToolCoordinator(
                ScriptedPlanner(),
                manager,
                Scratchpad(temporary / "scratchpad.jsonl"),
                temporary / "activity.jsonl",
            )
            result = await coordinator.respond(
                r"Run one tornado on C:\cases\Texas2k\study.pwb and preserve the original.",
                [],
                ["grid-workshop.powerworld"],
            )
            print(result)
            if "`paths_csv`" not in result:
                raise RuntimeError("Coordinator did not stop for the required tornado path CSV.")
    finally:
        await manager.stop_all()


if __name__ == "__main__":
    asyncio.run(main())
