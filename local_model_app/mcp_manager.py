from __future__ import annotations

import asyncio
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from typing import Any

from local_model_app.mcp_plugins import DiscoveredTool, McpPluginManifest, McpPluginRegistry


@dataclass
class ActivePlugin:
    stop_event: asyncio.Event = field(default_factory=asyncio.Event)
    ready_event: asyncio.Event = field(default_factory=asyncio.Event)
    task: asyncio.Task[None] | None = None
    clients: dict[str, Any] = field(default_factory=dict)
    tools: list[DiscoveredTool] = field(default_factory=list)
    error: str | None = None


class McpPluginManager:
    def __init__(self, registry: McpPluginRegistry) -> None:
        self.registry = registry
        self._active: dict[str, ActivePlugin] = {}
        self._errors: dict[str, str] = {}
        self._lock = asyncio.Lock()

    def status(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for plugin in self.registry.plugins:
            active = self._active.get(plugin.manifest.id)
            running = bool(active and active.ready_event.is_set() and not active.error)
            status = "running" if running else ("starting" if active else ("error" if plugin.manifest.id in self._errors else "stopped"))
            rows.append({
                "id": plugin.manifest.id,
                "name": plugin.manifest.name,
                "version": plugin.manifest.version,
                "description": plugin.manifest.description,
                "tool_namespace": plugin.manifest.tool_namespace,
                "installed": True,
                "running": running,
                "status": status,
                "error": self._errors.get(plugin.manifest.id),
                "server_count": len(plugin.manifest.servers),
                "tool_count": len(active.tools) if active else 0,
                "transports": sorted({server.transport for server in plugin.manifest.servers}),
                "default_access": plugin.manifest.policy.default_access,
            })
        return rows

    async def start(self, plugin_id: str) -> dict[str, Any]:
        async with self._lock:
            if plugin_id in self._active:
                return next(row for row in self.status() if row["id"] == plugin_id)
            plugin = self.registry.get_plugin(plugin_id)
            if not plugin.manifest.enabled:
                raise RuntimeError("Plugin is disabled in its manifest.")
            active = ActivePlugin()
            self._active[plugin_id] = active
            active.task = asyncio.create_task(self._run_plugin(plugin_id, active))
            await active.ready_event.wait()
            if active.error:
                self._active.pop(plugin_id, None)
                self._errors[plugin_id] = active.error
                if active.task:
                    await active.task
                raise RuntimeError(active.error)
            self._errors.pop(plugin_id, None)
            return next(row for row in self.status() if row["id"] == plugin_id)

    async def _run_plugin(self, plugin_id: str, active: ActivePlugin) -> None:
        plugin = self.registry.get_plugin(plugin_id)
        try:
            async with AsyncExitStack() as stack:
                for server in plugin.manifest.servers:
                    client = await stack.enter_async_context(self.registry.connect(server))
                    active.clients[server.id] = client
                    active.tools.extend(await self.registry.discover_connected_tools(plugin, server, client))
                active.ready_event.set()
                await active.stop_event.wait()
        except Exception as exc:
            active.error = str(exc)
            active.ready_event.set()

    async def stop(self, plugin_id: str) -> dict[str, Any]:
        async with self._lock:
            active = self._active.pop(plugin_id, None)
            if active:
                active.stop_event.set()
                if active.task:
                    await active.task
            self._errors.pop(plugin_id, None)
            self.registry.get_plugin(plugin_id)
            return next(row for row in self.status() if row["id"] == plugin_id)

    async def stop_all(self) -> None:
        for plugin_id in list(self._active):
            await self.stop(plugin_id)

    def active_tools(self) -> list[DiscoveredTool]:
        return [tool for plugin in self._active.values() for tool in plugin.tools]

    def tools_for_plugins(self, plugin_ids: list[str]) -> list[DiscoveredTool]:
        return [
            tool
            for plugin_id in plugin_ids
            for tool in (self._active.get(plugin_id).tools if self._active.get(plugin_id) else [])
        ]

    async def ensure_started(self, plugin_ids: list[str]) -> None:
        for plugin_id in plugin_ids:
            if plugin_id not in self._active:
                await self.start(plugin_id)

    async def call_tool(
        self,
        exposed_name: str,
        arguments: dict[str, Any],
        *,
        approved: bool = False,
    ) -> dict[str, Any]:
        from jsonschema import validate

        matches = [tool for tool in self.active_tools() if tool.exposed_name == exposed_name]
        if len(matches) != 1:
            raise KeyError(exposed_name)
        tool = matches[0]
        validate(arguments, tool.input_schema)
        plugin = self.registry.get_plugin(tool.plugin_id)
        policy = plugin.manifest.policy
        if tool.native_name in policy.deny_tools:
            raise PermissionError(f"Tool is denied by plugin policy: {tool.native_name}")
        explicitly_allowed = tool.native_name in policy.allow_tools
        if not explicitly_allowed and policy.default_access == "deny":
            raise PermissionError(f"Tool is not allowed by plugin policy: {tool.native_name}")
        if not explicitly_allowed and policy.default_access == "ask" and not approved:
            raise PermissionError(f"Tool requires user approval: {tool.native_name}")
        active = self._active[tool.plugin_id]
        client = active.clients[tool.server_id]
        result = await client.call_tool(tool.native_name, arguments)
        return {
            "content": [item.model_dump(exclude_none=True) for item in result.content],
            "structuredContent": result.structured_content,
            "isError": bool(result.is_error),
        }

    async def install(self, manifest_data: dict[str, Any]) -> dict[str, Any]:
        manifest = McpPluginManifest.model_validate(manifest_data)
        destination = self.registry.config_dir / f"{manifest.id}.json"
        if destination.exists():
            raise FileExistsError(f"Plugin is already installed: {manifest.id}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(".json.tmp")
        temporary.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
        temporary.replace(destination)
        self.registry.reload()
        return next(row for row in self.status() if row["id"] == manifest.id)
