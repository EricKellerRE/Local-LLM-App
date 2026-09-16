"""Manual smoke test for the real standard Grid MCP catalog without model weights."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from local_model_app.config import _load_dotenv
from local_model_app.mcp_manager import McpPluginManager
from local_model_app.mcp_plugins import McpPluginRegistry
from local_model_app.model import AssistantReply
from local_model_app.scratchpad import Scratchpad
from local_model_app.tool_coordinator import ToolCoordinator

SAFE_EXPOSED_TOOL = "grid__regulatory_list_tests"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the standard PowerWorld MCP state-machine smoke test.")
    parser.add_argument("--powerworld-repo", type=Path)
    return parser.parse_args()


class ScriptedObserver:
    def __init__(self) -> None:
        self.schema_counts: list[int] = []
        self.observed = False

    def generate(self, messages, **kwargs):
        return "List formal regulatory profiles with the safe catalog tool, then summarize the observation."

    def chat(self, messages, *, tools, **kwargs):
        self.schema_counts.append(len(tools))
        observation = next((message for message in reversed(messages) if message.get("role") == "tool"), None)
        if observation:
            self.observed = True
            payload = json.loads(observation["content"])
            data = payload.get("data") if isinstance(payload, dict) else None
            profiles = data.get("profiles", []) if isinstance(data, dict) else []
            profile_count = (
                data.get("profile_count", len(profiles))
                if isinstance(data, dict)
                else 0
            )
            return AssistantReply(content=f"Observed: {profile_count} profiles returned.", tool_calls=[])
        selected = next(item for item in tools if item["function"]["name"] == SAFE_EXPOSED_TOOL)
        return AssistantReply(content="", tool_calls=[{
            "id": "call_safe_regulatory_list",
            "type": "function",
            "function": {"name": selected["function"]["name"], "arguments": {}},
        }])


async def main(args: argparse.Namespace) -> None:
    _load_dotenv(ROOT / ".env")
    variables = {}
    if args.powerworld_repo:
        variables["GRID_WORKSHOP_ROOT"] = str(args.powerworld_repo.resolve().parent)
    manager = McpPluginManager(
        McpPluginRegistry(ROOT / "config" / "mcp.d", project_root=ROOT, variables=variables)
    )
    try:
        with tempfile.TemporaryDirectory(dir=ROOT / "data") as directory:
            temporary = Path(directory)
            observer = ScriptedObserver()
            coordinator = ToolCoordinator(
                observer,
                manager,
                Scratchpad(temporary / "scratchpad.jsonl"),
                temporary / "activity.jsonl",
            )
            result = await coordinator.respond(
                "Which formal regulatory test profiles are available? Do not run them.",
                [],
                ["grid-workshop.powerworld"],
            )
            if max(observer.schema_counts) > 4:
                raise RuntimeError(f"Router disclosed too many schemas: {observer.schema_counts}")
            if not observer.observed:
                raise RuntimeError("Final answer was composed without observing the tool result.")
            if "Approval is required" in result:
                raise RuntimeError("Safe regulatory listing unexpectedly requires approval.")
            print(result)
    finally:
        await manager.stop_all()


if __name__ == "__main__":
    asyncio.run(main(parse_args()))
