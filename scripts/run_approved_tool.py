from __future__ import annotations

import argparse
import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from local_model_app.config import _load_dotenv
from local_model_app.mcp_manager import McpPluginManager
from local_model_app.mcp_plugins import McpPluginRegistry


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def log(activity_path: Path, event: str, **values: Any) -> None:
    with activity_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"timestamp": now(), "event": event, **values}, ensure_ascii=False) + "\n")


async def run(root: Path, chat_id: str, plugin_id: str) -> dict[str, Any]:
    activity_path = root / "data" / "tool_activity" / f"{chat_id}.jsonl"
    state_path = activity_path.with_suffix(".state.json")
    state = json.loads(state_path.read_text(encoding="utf-8"))
    if state.get("status") != "waiting_for_approval":
        raise RuntimeError(f"Chat is not waiting for approval (status={state.get('status')!r}).")

    exposed_tool = str(state["exposed_tool"])
    arguments = state["arguments"]
    consuming = {**state, "status": "executing_approved_call", "approved_at": now()}
    write_json(state_path, consuming)
    log(activity_path, "approved_tool_execution_started", tool=exposed_tool, arguments=arguments)

    manager = McpPluginManager(McpPluginRegistry(root / "config" / "mcp.d", project_root=root))
    try:
        await manager.ensure_started([plugin_id])
        result = await manager.call_tool(exposed_tool, arguments, approved=True)
        terminal = "failed" if result.get("isError") else "complete"
        write_json(state_path, {
            "status": terminal,
            "approved_at": consuming["approved_at"],
            "completed_at": now(),
            "exposed_tool": exposed_tool,
            "arguments": arguments,
            "result": result,
        })
        log(activity_path, "approved_tool_execution_finished", tool=exposed_tool, result=result)
        return result
    except Exception as exc:
        write_json(state_path, {**consuming, "status": "failed", "completed_at": now(), "error": str(exc)})
        log(activity_path, "approved_tool_execution_failed", tool=exposed_tool, error=type(exc).__name__, message=str(exc))
        raise
    finally:
        await manager.stop_all()


def main() -> int:
    parser = argparse.ArgumentParser(description="Consume one saved MCP approval exactly once.")
    parser.add_argument("chat_id")
    parser.add_argument("--plugin-id", default="grid-workshop.powerworld")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    _load_dotenv(root / ".env")
    result = asyncio.run(run(root, args.chat_id, args.plugin_id))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 1 if result.get("isError") else 0


if __name__ == "__main__":
    raise SystemExit(main())
