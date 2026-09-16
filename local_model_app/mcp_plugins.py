from __future__ import annotations

import asyncio
import json
import os
import re
import hashlib
import subprocess
import sys
import urllib.error
import urllib.request
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, AsyncIterator, Literal

from pydantic import BaseModel, Field, model_validator


VARIABLE_PATTERN = re.compile(r"\$\{([A-Z][A-Z0-9_]*)\}")
SAFE_NAME_PATTERN = re.compile(r"[^a-zA-Z0-9_]+")


class PluginConfigurationError(ValueError):
    pass


class CompanionProcessConfig(BaseModel):
    id: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]*$")
    command: str
    args: list[str] = Field(default_factory=list)
    cwd: str | None = None
    env: dict[str, str] = Field(default_factory=dict)
    ready_url: str | None = None
    timeout_seconds: float = Field(default=30, gt=0)


class McpServerConfig(BaseModel):
    id: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]*$")
    transport: Literal["stdio", "streamable_http"]
    command: str | None = None
    args: list[str] = Field(default_factory=list)
    cwd: str | None = None
    url: str | None = None
    env: dict[str, str] = Field(default_factory=dict)
    headers: dict[str, str] = Field(default_factory=dict)
    companions: list[CompanionProcessConfig] = Field(default_factory=list)
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
        companion_ids = [companion.id for companion in self.companions]
        if len(companion_ids) != len(set(companion_ids)):
            raise ValueError("companion ids must be unique within a server")
        return self


class PluginPolicy(BaseModel):
    default_access: Literal["deny", "ask", "allow"] = "ask"
    allow_tools: list[str] = Field(default_factory=list)
    deny_tools: list[str] = Field(default_factory=list)
    max_calls_per_turn: int = Field(default=256, ge=1, le=4096)


class GuidanceDocument(BaseModel):
    path: str
    capability: str | None = None


@dataclass(frozen=True)
class LoadedGuidance:
    plugin_id: str
    capability: str | None
    path: str
    version: str
    content: str
    truncated: bool


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
    execution: dict[str, Any] | None = None
    kind: Literal["tool", "resource", "resource_template", "prompt"] = "tool"
    target: str | None = None

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


def _uri_template_pattern(template: str) -> str:
    """Build a conservative JSON Schema pattern for common RFC 6570 templates."""
    pattern = "^"
    for part in re.split(r"(\{[^{}]+\})", template):
        if not part:
            continue
        if not part.startswith("{"):
            pattern += re.escape(part)
            continue
        expression = part[1:-1]
        operator = expression[:1]
        if operator == "?":
            pattern += r"(?:\?[^#]*)?"
        elif operator == "&":
            pattern += r"(?:&[^#]*)?"
        elif operator == "#":
            pattern += r"(?:#.*)?"
        else:
            pattern += r".+?"
    return pattern + "$"


def normalize_tool_result(tool: DiscoveredTool, result: Any) -> dict[str, Any]:
    """Serialize a native result and quarantine malformed structured output."""
    content = [item.model_dump(mode="json", by_alias=True, exclude_none=True) for item in result.content]
    structured = result.structured_content
    serialized = {
        "content": content,
        "structuredContent": structured,
        "isError": bool(result.is_error),
    }
    if tool.output_schema is None:
        return serialized

    from jsonschema import Draft202012Validator, SchemaError

    try:
        Draft202012Validator.check_schema(tool.output_schema)
        errors = list(Draft202012Validator(tool.output_schema).iter_errors(structured))
    except SchemaError as exc:
        return {
            "content": [],
            "structuredContent": None,
            "isError": True,
            "error": "invalid_output_schema",
            "message": f"The MCP server advertised an invalid output schema: {exc.message[:500]}",
            "rawResult": serialized,
        }
    if not errors:
        return serialized
    details = [
        {
            "path": "/".join(str(part) for part in error.absolute_path),
            "message": error.message[:500],
        }
        for error in errors[:5]
    ]
    return {
        "content": [],
        "structuredContent": None,
        "isError": True,
        "error": "output_validation_failed",
        "message": "The MCP tool returned structured content that does not match its output schema.",
        "validationErrors": details,
        "rawResult": serialized,
    }


class McpPluginRegistry:
    def __init__(self, config_dir: Path, *, project_root: Path, variables: dict[str, str] | None = None) -> None:
        self.config_dir = config_dir
        self.project_root = project_root.resolve()
        self._extra_variables = dict(variables or {})
        self.variables = {
            **os.environ,
            "PROJECT_ROOT": str(self.project_root),
            "PYTHON_EXECUTABLE": sys.executable,
            **self._extra_variables,
        }
        self.plugins = self._load_plugins()

    def reload(self) -> None:
        self.variables = {
            **os.environ,
            "PROJECT_ROOT": str(self.project_root),
            "PYTHON_EXECUTABLE": sys.executable,
            **self._extra_variables,
        }
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
        for companion in data["companions"]:
            for key in ("command", "cwd", "ready_url"):
                if companion.get(key):
                    companion[key] = _expand_string(companion[key], self.variables)
            companion["args"] = [_expand_string(item, self.variables) for item in companion["args"]]
            companion["env"] = {
                name: _expand_string(value, self.variables)
                for name, value in companion["env"].items()
            }
        return McpServerConfig.model_validate(data)

    @staticmethod
    def _url_is_ready(url: str) -> bool:
        try:
            with urllib.request.urlopen(url, timeout=0.5):
                return True
        except (OSError, urllib.error.URLError):
            return False

    async def _start_companions(
        self, companions: list[CompanionProcessConfig]
    ) -> list[asyncio.subprocess.Process]:
        processes: list[asyncio.subprocess.Process] = []
        try:
            for companion in companions:
                if companion.ready_url and await asyncio.to_thread(self._url_is_ready, companion.ready_url):
                    raise RuntimeError(
                        f"Cannot start companion '{companion.id}': its address is already in use."
                    )
                process = await asyncio.create_subprocess_exec(
                    companion.command,
                    *companion.args,
                    cwd=companion.cwd,
                    env={**os.environ, **companion.env},
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                processes.append(process)
                if not companion.ready_url:
                    continue
                deadline = asyncio.get_running_loop().time() + companion.timeout_seconds
                while asyncio.get_running_loop().time() < deadline:
                    if process.returncode is not None:
                        raise RuntimeError(
                            f"Companion '{companion.id}' stopped during startup (exit {process.returncode})."
                        )
                    if await asyncio.to_thread(self._url_is_ready, companion.ready_url):
                        break
                    await asyncio.sleep(0.1)
                else:
                    raise RuntimeError(f"Companion '{companion.id}' did not become ready in time.")
            return processes
        except Exception:
            await self._stop_companions(processes)
            raise

    @staticmethod
    async def _stop_companions(processes: list[asyncio.subprocess.Process]) -> None:
        for process in reversed(processes):
            if process.returncode is None:
                process.terminate()
        for process in reversed(processes):
            if process.returncode is not None:
                continue
            try:
                await asyncio.wait_for(process.wait(), timeout=5)
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()

    @asynccontextmanager
    async def connect(self, server: McpServerConfig) -> AsyncIterator[Any]:
        from mcp import Client, StdioServerParameters

        selected = self._expanded_server(server)
        companions = await self._start_companions(selected.companions)
        try:
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
        finally:
            await self._stop_companions(companions)

    async def discover_tools(self) -> list[DiscoveredTool]:
        discovered: list[DiscoveredTool] = []
        names: set[str] = set()
        for plugin in self.enabled_plugins():
            for server in plugin.manifest.servers:
                async with self.connect(server) as client:
                    capabilities = getattr(client, "server_capabilities", None)
                    server_items: list[DiscoveredTool] = []
                    if capabilities is None or getattr(capabilities, "tools", None) is not None:
                        server_items.extend(await self.discover_connected_tools(plugin, server, client))
                    if capabilities is not None and getattr(capabilities, "resources", None) is not None:
                        server_items.extend(await self.discover_connected_resources(plugin, server, client))
                    if capabilities is not None and getattr(capabilities, "prompts", None) is not None:
                        server_items.extend(await self.discover_connected_prompts(plugin, server, client))
                    for tool in server_items:
                        if tool.exposed_name in names:
                            raise PluginConfigurationError(f"Exposed MCP capability-name collision: {tool.exposed_name}")
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
                        annotations=(
                            tool.annotations.model_dump(by_alias=True, exclude_none=True)
                            if tool.annotations else {}
                        ),
                        execution=(
                            tool.execution.model_dump(by_alias=True, exclude_none=True)
                            if getattr(tool, "execution", None) else None
                        ),
                    )
                )
            cursor = result.next_cursor
            if not cursor:
                return tools

    async def discover_connected_resources(
        self, plugin: LoadedPlugin, server: McpServerConfig, client: Any
    ) -> list[DiscoveredTool]:
        resources: list[DiscoveredTool] = []
        cursor: str | None = None
        while True:
            result = await client.list_resources(cursor=cursor)
            for resource in result.resources:
                native_name = f"resource:{resource.name}"
                resources.append(DiscoveredTool(
                    exposed_name=_safe_tool_name(plugin.manifest.tool_namespace, native_name),
                    plugin_id=plugin.manifest.id,
                    server_id=server.id,
                    native_name=native_name,
                    title=resource.title or resource.name,
                    description=resource.description or f"Read MCP resource {resource.uri}",
                    input_schema={"type": "object", "properties": {}, "additionalProperties": False},
                    output_schema=None,
                    annotations={"readOnlyHint": True},
                    kind="resource",
                    target=str(resource.uri),
                ))
            cursor = result.next_cursor
            if not cursor:
                break

        cursor = None
        while True:
            result = await client.list_resource_templates(cursor=cursor)
            for template in result.resource_templates:
                native_name = f"resource_template:{template.name}"
                resources.append(DiscoveredTool(
                    exposed_name=_safe_tool_name(plugin.manifest.tool_namespace, native_name),
                    plugin_id=plugin.manifest.id,
                    server_id=server.id,
                    native_name=native_name,
                    title=template.title or template.name,
                    description=(template.description or "Read an MCP resource") + f" URI template: {template.uri_template}",
                    input_schema={
                        "type": "object",
                        "properties": {
                            "uri": {
                                "type": "string",
                                "pattern": _uri_template_pattern(str(template.uri_template)),
                                "description": f"URI matching {template.uri_template}",
                            },
                        },
                        "required": ["uri"],
                        "additionalProperties": False,
                    },
                    output_schema=None,
                    annotations={"readOnlyHint": True},
                    kind="resource_template",
                    target=str(template.uri_template),
                ))
            cursor = result.next_cursor
            if not cursor:
                return resources

    async def discover_connected_prompts(
        self, plugin: LoadedPlugin, server: McpServerConfig, client: Any
    ) -> list[DiscoveredTool]:
        prompts: list[DiscoveredTool] = []
        cursor: str | None = None
        while True:
            result = await client.list_prompts(cursor=cursor)
            for prompt in result.prompts:
                properties: dict[str, Any] = {}
                required: list[str] = []
                for argument in prompt.arguments or []:
                    properties[argument.name] = {
                        "type": "string",
                        "description": argument.description or argument.title or "Prompt argument",
                    }
                    if argument.required:
                        required.append(argument.name)
                native_name = f"prompt:{prompt.name}"
                schema: dict[str, Any] = {
                    "type": "object",
                    "properties": properties,
                    "additionalProperties": False,
                }
                if required:
                    schema["required"] = required
                prompts.append(DiscoveredTool(
                    exposed_name=_safe_tool_name(plugin.manifest.tool_namespace, native_name),
                    plugin_id=plugin.manifest.id,
                    server_id=server.id,
                    native_name=native_name,
                    title=prompt.title or prompt.name,
                    description=prompt.description or f"Render MCP prompt {prompt.name}",
                    input_schema=schema,
                    output_schema=None,
                    annotations={"readOnlyHint": True},
                    kind="prompt",
                    target=prompt.name,
                ))
            cursor = result.next_cursor
            if not cursor:
                return prompts

    def guidance_for_tools(
        self,
        tools: list[DiscoveredTool],
        *,
        max_documents: int = 2,
        max_chars_per_document: int = 6000,
    ) -> list[LoadedGuidance]:
        selected: list[LoadedGuidance] = []
        seen_paths: set[Path] = set()
        for tool in tools:
            plugin = self.get_plugin(tool.plugin_id)
            searchable = re.sub(r"[^a-z0-9]+", " ", f"{tool.native_name} {tool.description}".lower()).split()
            for document in plugin.manifest.guidance:
                capability = document.capability.lower() if document.capability else None
                if capability and capability not in searchable:
                    continue
                expanded = Path(_expand_string(document.path, self.variables)).resolve()
                if expanded in seen_paths or not expanded.is_file():
                    continue
                raw = expanded.read_text(encoding="utf-8")
                digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:12]
                selected.append(LoadedGuidance(
                    plugin_id=tool.plugin_id,
                    capability=document.capability,
                    path=str(expanded),
                    version=f"sha256:{digest}",
                    content=raw[:max_chars_per_document],
                    truncated=len(raw) > max_chars_per_document,
                ))
                seen_paths.add(expanded)
                if len(selected) >= max_documents:
                    return selected
        return selected


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
            result = await client.call_tool(tool.native_name, arguments)
        return normalize_tool_result(tool, result)
