from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from local_model_app.config import _load_dotenv
from local_model_app.mcp_manager import McpPluginManager
from local_model_app.mcp_plugins import McpPluginRegistry
from local_model_app.model import AssistantReply
from local_model_app.scratchpad import Scratchpad
from local_model_app.tool_coordinator import ToolCoordinator

SAFE_NATIVE_TOOL = "regulatory.list_tests"
SAFE_EXPOSED_TOOL = "grid__regulatory_list_tests"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a safe standard-catalog PowerWorld MCP smoke test.")
    parser.add_argument("--powerworld-repo", type=Path)
    parser.add_argument("--transcript", type=Path, default=Path("data/powerworld-tool-smoke.jsonl"))
    return parser.parse_args()


class ObservingPlanner:
    """Exercise schema disclosure and result observation without loading model weights."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def generate(self, messages, **kwargs):
        return "Use the safe regulatory catalog listing and report the observed profile count."

    def chat(self, messages, *, tools, **kwargs):
        self.calls.append({"messages": messages, "tools": tools})
        if len(tools) > 4:
            raise RuntimeError(f"Progressive disclosure exposed {len(tools)} schemas; expected at most four.")
        tool_message = next((message for message in reversed(messages) if message.get("role") == "tool"), None)
        if tool_message is not None:
            observation = json.loads(tool_message["content"])
            if "data" not in observation and not observation.get("isError"):
                raise RuntimeError("The tool result was not observed before final composition.")
            data = observation.get("data") if isinstance(observation, dict) else None
            profiles = data.get("profiles", []) if isinstance(data, dict) else []
            profile_count = (
                data.get("profile_count", len(profiles))
                if isinstance(data, dict)
                else 0
            )
            return AssistantReply(
                content=f"Observed regulatory catalog result: {profile_count} profiles returned.",
                tool_calls=[],
            )
        match = next(
            (item for item in tools if item["function"]["name"] == SAFE_EXPOSED_TOOL),
            None,
        )
        if match is None:
            raise RuntimeError("The safe regulatory listing was not in the router's top four schemas.")
        return AssistantReply(content="", tool_calls=[{
            "id": "call_regulatory_catalog",
            "type": "function",
            "function": {"name": match["function"]["name"], "arguments": {}},
        }])


async def run(args: argparse.Namespace) -> str:
    _load_dotenv(ROOT / ".env")
    variables = {}
    if args.powerworld_repo:
        variables["GRID_WORKSHOP_ROOT"] = str(args.powerworld_repo.resolve().parent)
    registry = McpPluginRegistry(ROOT / "config" / "mcp.d", project_root=ROOT, variables=variables)
    manager = McpPluginManager(registry)
    transcript = args.transcript.resolve()
    transcript.parent.mkdir(parents=True, exist_ok=True)
    try:
        planner = ObservingPlanner()
        coordinator = ToolCoordinator(
            planner,
            manager,
            Scratchpad(transcript.with_suffix(".scratchpad.jsonl")),
            transcript,
        )
        answer = await coordinator.respond(
            "List the available formal regulatory study profiles without running a study.",
            [],
            ["grid-workshop.powerworld"],
        )
        if "Approval is required" in answer:
            raise RuntimeError("The conservative regulatory listing is not allowed by the host manifest.")
        if not any(message.get("role") == "tool" for message in planner.calls[-1]["messages"]):
            raise RuntimeError("The planner composed an answer before observing the tool result.")
        return answer
    finally:
        await manager.stop_all()


def main() -> int:
    args = parse_args()
    try:
        answer = asyncio.run(run(args))
    except Exception as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, indent=2))
        return 1
    print(json.dumps({"ok": True, "answer": answer, "transcript": str(args.transcript.resolve())}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
