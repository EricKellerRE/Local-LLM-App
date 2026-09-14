from __future__ import annotations

import json
import os
import re
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, AsyncIterator, Literal

from pydantic import BaseModel, Field, model_validator


VARIABLE_PATTERN = re.compile(r"\$\{([A-Z][A-Z0-9_]*)\}")
SAFE_NAME_PATTERN = re.compile(r"[^a-zA-Z0-9_]+")


class PluginConfigurationError(ValueError):
    pass


class McpServerConfig(BaseModel):
    id: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]*$")
    transport: Literal["stdio", "streamable_http"]
    command: str | None = None
    args: list[str] = Field(default_factory=list)
    cwd: str | None = None
    url: str | None = None
    env: dict[str, str] = Field(default_factory=dict)
    headers: dict[str, str] = Field(default_factory=dict)
    timeout_seconds: float = Field(default=300, gt=0)

    @model_validator(mode="after")
    def validate_transport_fields(self) -> "McpServerConfig":
        if self.transport == "stdio" and not self.command:
            raise ValueError("stdio servers require command")
        if self.transport == "streamable_http" and not self.url:
            raise ValueError("streamable_http servers require url")
        if self.transport == "stdio" and self.url:
            raise ValueError("stdio servers cannot define url")
        if self.transport == "streamable_http" and (self.command or self.args or self.cwd or self.env):
            raise ValueError("streamable_http servers cannot define stdio process fields")
        return self


class PluginPolicy(BaseModel):
    default_access: Literal["deny", "ask", "allow"] = "ask"
    allow_tools: list[str] = Field(default_factory=list)
    deny_tools: list[str] = Field(default_factory=list)
    max_calls_per_turn: int = Field(default=8, ge=1, le=100)


class GuidanceDocument(BaseModel):
    path: str
    capability: str | None = None


class McpPluginManifest(BaseModel):
    manifest_version: Literal[1]
    id: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]*$")
    tool_namespace: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    name: str
    version: str
    description: str = ""
    enabled: bool = True
    servers: list[McpServerConfig] = Field(min_length=1)
    guidance: list[GuidanceDocument] = Field(default_factory=list)
    policy: PluginPolicy = Field(default_factory=PluginPolicy)

    @model_validator(mode="after")
    def unique_server_ids(self) -> "McpPluginManifest":
        ids = [server.id for server in self.servers]
        if len(ids) != len(set(ids)):
            raise ValueError("server ids must be unique within a plugin")
        return self


@dataclass(frozen=True)
class LoadedPlugin:
    manifest: McpPluginManifest
    manifest_path: Path


@dataclass(frozen=True)
class DiscoveredTool:
    exposed_name: str
    plugin_id: str
    server_id: str
    native_name: str
    title: str | None
    description: str
    input_schema: dict[str, Any]
    output_schema: dict[str, Any] | None
    annotations: dict[str, Any]

    def openai_spec(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.exposed_name,
                "description": self.description,
                "parameters": self.input_schema,
            },
        }


def _expand_string(value: str, variables: dict[str, str]) -> str:
    def replace(match: re.Match[str]) -> str:
        name = match.group(1)
        if name not in variables:
            raise PluginConfigurationError(f"Missing configuration variable: {name}")
        return variables[name]

    return VARIABLE_PATTERN.sub(replace, value)


def _safe_tool_name(namespace: str, tool_name: str) -> str:
    parts = [namespace, tool_name]
    value = "__".join(SAFE_NAME_PATTERN.sub("_", part).strip("_").lower() for part in parts)
    if len(value) <= 64:
        return value
    import hashlib

    suffix = hashlib.sha256(value.encode("utf-8")).hexdigest()[:10]
    return f"{value[:53]}_{suffix}"


class McpPluginRegistry:
    def __init__(self, config_dir: Path, *, project_root: Path, variables: dict[str, str] | None = None) -> None:
        self.config_dir = config_dir
        self.project_root = project_root.resolve()
        self._extra_variables = dict(variables or {})
        self.variables = {**os.environ, "PROJECT_ROOT": str(self.project_root), **self._extra_variables}
        self.plugins = self._load_plugins()

    def reload(self) -> None:
        self.variables = {**os.environ, "PROJECT_ROOT": str(self.project_root), **self._extra_variables}
        self.plugins = self._load_plugins()

    def get_plugin(self, plugin_id: str) -> LoadedPlugin:
        try:
            return next(plugin for plugin in self.plugins if plugin.manifest.id == plugin_id)
        except StopIteration as exc:
            raise KeyError(plugin_id) from exc

    def _load_plugins(self) -> list[LoadedPlugin]:
        loaded: list[LoadedPlugin] = []
        plugin_ids: set[str] = set()
        namespaces: set[str] = set()
        for path in sorted(self.config_dir.glob("*.json")):
            try:
                manifest = McpPluginManifest.model_validate_json(path.read_text(encoding="utf-8"))
            except Exception as exc:
                raise PluginConfigurationError(f"Invalid MCP plugin manifest {path}: {exc}") from exc
            if manifest.id in plugin_ids:
                raise PluginConfigurationError(f"Duplicate MCP plugin id: {manifest.id}")
            if manifest.tool_namespace in namespaces:
                raise PluginConfigurationError(f"Duplicate MCP tool namespace: {manifest.tool_namespace}")
            plugin_ids.add(manifest.id)
            namespaces.add(manifest.tool_namespace)
            loaded.append(LoadedPlugin(manifest=manifest, manifest_path=path.resolve()))
        return loaded

    def enabled_plugins(self) -> list[LoadedPlugin]:
        return [plugin for plugin in self.plugins if plugin.manifest.enabled]

    def _expanded_server(self, server: McpServerConfig) -> McpServerConfig:
        data = server.model_dump()
        for key in ("command", "cwd", "url"):
            if data.get(key):
                data[key] = _expand_string(data[key], self.variables)
        data["args"] = [_expand_string(item, self.variables) for item in data["args"]]
        data["env"] = {name: _expand_string(value, self.variables) for name, value in data["env"].items()}
        data["headers"] = {name: _expand_string(value, self.variables) for name, value in data["headers"].items()}
        return McpServerConfig.model_validate(data)

    @asynccontextmanager
    async def connect(self, server: McpServerConfig) -> AsyncIterator[Any]:
        from mcp import Client, StdioServerParameters

        selected = self._expanded_server(server)
        if selected.transport == "stdio":
            parameters = StdioServerParameters(
                command=selected.command or "",
                args=selected.args,
                cwd=selected.cwd,
                env=selected.env or None,
            )
            async with Client(parameters, read_timeout_seconds=selected.timeout_seconds) as client:
                yield client
            return

        if selected.headers:
            import httpx2
            from mcp.client.streamable_http import streamable_http_client

            async with httpx2.AsyncClient(headers=selected.headers) as http_client:
                transport = streamable_http_client(selected.url or "", http_client=http_client)
                async with Client(transport, read_timeout_seconds=selected.timeout_seconds) as client:
                    yield client
        else:
            async with Client(selected.url or "", read_timeout_seconds=selected.timeout_seconds) as client:
                yield client

    async def discover_tools(self) -> list[DiscoveredTool]:
        discovered: list[DiscoveredTool] = []
        names: set[str] = set()
        for plugin in self.enabled_plugins():
            for server in plugin.manifest.servers:
                async with self.connect(server) as client:
                    server_tools = await self.discover_connected_tools(plugin, server, client)
                    for tool in server_tools:
                        if tool.exposed_name in names:
                            raise PluginConfigurationError(f"Exposed MCP tool-name collision: {tool.exposed_name}")
                        names.add(tool.exposed_name)
                        discovered.append(tool)
        return discovered

    async def discover_connected_tools(
        self, plugin: LoadedPlugin, server: McpServerConfig, client: Any
    ) -> list[DiscoveredTool]:
        tools: list[DiscoveredTool] = []
        cursor: str | None = None
        while True:
            result = await client.list_tools(cursor=cursor)
            for tool in result.tools:
                tools.append(
                    DiscoveredTool(
                        exposed_name=_safe_tool_name(plugin.manifest.tool_namespace, tool.name),
                        plugin_id=plugin.manifest.id,
                        server_id=server.id,
                        native_name=tool.name,
                        title=tool.title,
                        description=tool.description or "",
                        input_schema=tool.input_schema,
                        output_schema=tool.output_schema,
                        annotations=tool.annotations.model_dump(exclude_none=True) if tool.annotations else {},
                    )
                )
            cursor = result.next_cursor
            if not cursor:
                return tools

    async def call_tool(
        self,
        tool: DiscoveredTool,
        arguments: dict[str, Any],
        *,
        approved: bool = False,
    ) -> dict[str, Any]:
        from jsonschema import validate

        validate(arguments, tool.input_schema)
        plugin = next(item for item in self.enabled_plugins() if item.manifest.id == tool.plugin_id)
        policy = plugin.manifest.policy
        if tool.native_name in policy.deny_tools:
            raise PermissionError(f"Tool is denied by plugin policy: {tool.native_name}")
        explicitly_allowed = tool.native_name in policy.allow_tools
        if not explicitly_allowed and policy.default_access == "deny":
            raise PermissionError(f"Tool is not allowed by plugin policy: {tool.native_name}")
        if not explicitly_allowed and policy.default_access == "ask" and not approved:
            raise PermissionError(f"Tool requires user approval: {tool.native_name}")
        server = next(item for item in plugin.manifest.servers if item.id == tool.server_id)
        async with self.connect(server) as client:
            result = await client.call_tool(tool.native_name, arguments)
        return {
            "content": [item.model_dump(exclude_none=True) for item in result.content],
            "structuredContent": result.structured_content,
            "isError": bool(result.is_error),
        }
