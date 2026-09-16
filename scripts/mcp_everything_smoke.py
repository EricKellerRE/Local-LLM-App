from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from local_model_app.config import Settings, _load_dotenv
from local_model_app.mcp_manager import McpPluginManager
from local_model_app.mcp_plugins import McpPluginRegistry
from local_model_app.tool_router import LocalEmbeddingEncoder, ToolRouter


PLUGIN_ID = "mcp.reference.everything"


async def run() -> dict:
    _load_dotenv(ROOT / ".env")
    settings = Settings.from_environment(ROOT)
    registry = McpPluginRegistry(ROOT / "config" / "mcp.d", project_root=ROOT)
    manager = McpPluginManager(registry)
    try:
        await manager.ensure_started([PLUGIN_ID])
        tools = manager.tools_for_plugins([PLUGIN_ID])
        resources = manager.resources_for_plugins([PLUGIN_ID])
        prompts = manager.prompts_for_plugins([PLUGIN_ID])
        if not tools or not resources or not prompts:
            raise RuntimeError(
                "The reference server must expose at least one tool, resource, and prompt; "
                f"discovered tools={len(tools)}, resources={len(resources)}, prompts={len(prompts)}."
            )

        encoder = (
            LocalEmbeddingEncoder(settings.router_model_id, device=settings.router_device)
            if settings.router_model_id
            else None
        )
        routed = ToolRouter(encoder, semantic_weight=settings.router_semantic_weight).route(
            "Echo the words interoperability check back to me.",
            tools,
        )
        echo = next((tool for tool in routed.visible if tool.native_name == "echo"), None)
        if echo is None:
            raise RuntimeError("Semantic routing did not place the reference echo tool in the top four schemas.")
        result = await manager.call_tool(
            echo.exposed_name,
            {"message": "interoperability check"},
            approved=True,
        )
        if result.get("isError"):
            raise RuntimeError(f"Reference echo call failed: {result}")
        return {
            "ok": True,
            "tools": len(tools),
            "resources": len(resources),
            "prompts": len(prompts),
            "routed_top_four": [tool.native_name for tool in routed.visible],
            "semantic_backend": routed.semantic_backend,
            "echo_result": result,
        }
    finally:
        await manager.stop_all()


def main() -> int:
    print(json.dumps(asyncio.run(run()), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
