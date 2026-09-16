from __future__ import annotations

import asyncio
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from local_model_app.mcp_plugins import (
    DiscoveredTool,
    McpPluginManifest,
    McpPluginRegistry,
    PluginConfigurationError,
    normalize_tool_result,
)


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

    @staticmethod
    def _add_discovered(active: ActivePlugin, discovered: list[DiscoveredTool]) -> None:
        names = {item.exposed_name for item in active.tools}
        for item in discovered:
            if item.exposed_name in names:
                raise PluginConfigurationError(f"Exposed MCP capability-name collision: {item.exposed_name}")
            names.add(item.exposed_name)
            active.tools.append(item)

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
                "tool_count": len([item for item in active.tools if item.kind == "tool"]) if active else 0,
                "resource_count": len([item for item in active.tools if item.kind.startswith("resource")]) if active else 0,
                "prompt_count": len([item for item in active.tools if item.kind == "prompt"]) if active else 0,
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
                    capabilities = getattr(client, "server_capabilities", None)
                    if capabilities is None or getattr(capabilities, "tools", None) is not None:
                        self._add_discovered(
                            active, await self.registry.discover_connected_tools(plugin, server, client)
                        )
                    if capabilities is not None and getattr(capabilities, "resources", None) is not None:
                        self._add_discovered(
                            active, await self.registry.discover_connected_resources(plugin, server, client)
                        )
                    if capabilities is not None and getattr(capabilities, "prompts", None) is not None:
                        self._add_discovered(
                            active, await self.registry.discover_connected_prompts(plugin, server, client)
                        )
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
            if tool.kind == "tool"
        ]

    def resources_for_plugins(self, plugin_ids: list[str]) -> list[DiscoveredTool]:
        return [
            item
            for plugin_id in plugin_ids
            for item in (self._active.get(plugin_id).tools if self._active.get(plugin_id) else [])
            if item.kind in {"resource", "resource_template"}
        ]

    def prompts_for_plugins(self, plugin_ids: list[str]) -> list[DiscoveredTool]:
        return [
            item
            for plugin_id in plugin_ids
            for item in (self._active.get(plugin_id).tools if self._active.get(plugin_id) else [])
            if item.kind == "prompt"
        ]

    async def ensure_started(self, plugin_ids: list[str]) -> None:
        for plugin_id in plugin_ids:
            if plugin_id not in self._active:
                await self.start(plugin_id)

    @staticmethod
    def _requires_durable_task(tool: DiscoveredTool) -> bool:
        execution = tool.execution or {}
        return execution.get("taskSupport") == "required"

    async def _wait_for_tool_task(
        self,
        client: Any,
        tool: DiscoveredTool,
        task_id: str,
    ) -> dict[str, Any]:
        from mcp import types

        while True:
            status = await client.session.send_request(
                types.GetTaskRequest(params=types.GetTaskRequestParams(taskId=task_id)),
                types.GetTaskResult,
            )
            if status.status == "completed":
                result = await client.session.send_request(
                    types.GetTaskPayloadRequest(
                        params=types.GetTaskPayloadRequestParams(taskId=task_id)
                    ),
                    types.CallToolResult,
                )
                return normalize_tool_result(tool, result)
            if status.status in {"failed", "cancelled"}:
                detail = status.status_message or f"MCP task {task_id} {status.status}."
                raise RuntimeError(detail)
            poll_seconds = max(0.25, min(float(status.poll_interval or 2000) / 1000.0, 30.0))
            await asyncio.sleep(poll_seconds)

    async def resume_tool_task(self, handle: dict[str, Any]) -> dict[str, Any]:
        """Resume polling a previously checkpointed MCP task without launching it again."""
        plugin_id = str(handle["plugin_id"])
        server_id = str(handle["server_id"])
        exposed_name = str(handle["exposed_name"])
        task_id = str(handle["task_id"])
        await self.ensure_started([plugin_id])
        active = self._active[plugin_id]
        tool = next(
            item
            for item in active.tools
            if item.server_id == server_id and item.exposed_name == exposed_name
        )
        return await self._wait_for_tool_task(active.clients[server_id], tool, task_id)

    async def call_tool(
        self,
        exposed_name: str,
        arguments: dict[str, Any],
        *,
        approved: bool = False,
        task_checkpoint: Callable[[dict[str, Any]], Awaitable[None]] | None = None,
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
        if tool.kind in {"resource", "resource_template"}:
            uri = tool.target if tool.kind == "resource" else arguments["uri"]
            result = await client.read_resource(uri)
            return {
                "content": [],
                "structuredContent": {
                    "uri": uri,
                    "contents": [
                        item.model_dump(mode="json", by_alias=True, exclude_none=True)
                        for item in result.contents
                    ],
                },
                "isError": False,
            }
        if tool.kind == "prompt":
            result = await client.get_prompt(tool.target or "", arguments or None)
            return {
                "content": [],
                "structuredContent": {
                    "description": result.description,
                    "messages": [
                        item.model_dump(mode="json", by_alias=True, exclude_none=True)
                        for item in result.messages
                    ],
                },
                "isError": False,
            }
        if self._requires_durable_task(tool):
            from mcp import types

            created = await client.session.send_request(
                types.CallToolRequest(params=types.CallToolRequestParams(
                    name=tool.native_name,
                    arguments=arguments,
                    task=types.TaskMetadata(ttl=7 * 24 * 60 * 60 * 1000),
                )),
                types.CreateTaskResult,
            )
            handle = {
                "plugin_id": tool.plugin_id,
                "server_id": tool.server_id,
                "exposed_name": tool.exposed_name,
                "native_name": tool.native_name,
                "task_id": created.task.task_id,
                "created_at": created.task.created_at,
            }
            if task_checkpoint is not None:
                await task_checkpoint(handle)
            return await self._wait_for_tool_task(client, tool, created.task.task_id)

        result = await client.call_tool(tool.native_name, arguments)
        return normalize_tool_result(tool, result)

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
